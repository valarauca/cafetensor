use cafetensor_testkit::{Sampler, load_profiles, test_bytes, test_seed};
use general_backend::codebook::{Codebook, EXP_BITS, ExpAlias, SM_BITS, SmMode};
use general_backend::rans::{
    EncTables, LANES, RANS_L, WAYS, encode_bound, encode_scratch_len, way_len,
};
use general_backend::{Format, OpError, Operations, available};

fn format_of(dtype: &str) -> Option<Format> {
    match dtype {
        "BF16" => Some(Format::BF16),
        "F32" => Some(Format::F32),
        "F16" => Some(Format::F16),
        "F8_E4M3" => Some(Format::E4M3),
        "F8_E5M2" => Some(Format::E5M2),
        _ => None,
    }
}

fn portable() -> &'static dyn Operations {
    *available().last().unwrap()
}

fn planes(fmt: Format, bytes: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let n = bytes.len() / fmt.width;
    let (mut e, mut h, mut l) = (vec![0; n], vec![0; n], vec![0; n * fmt.low_bytes()]);
    portable()
        .split_planes(fmt, bytes, &mut e, &mut h, &mut l)
        .unwrap();
    (e, h)
}

fn codebook(e: &[u8], h: &[u8], mode: SmMode) -> Codebook {
    let mut joint = Box::new([0u64; 65536]);
    for (&x, &y) in e.iter().zip(h) {
        joint[(x as usize) << 8 | y as usize] += 1;
    }
    Codebook::build(&joint, mode).expect("codebook")
}

fn global(w: usize, j: usize) -> usize {
    ((j / LANES) * WAYS + w) * LANES + j % LANES
}

struct Words<'a> {
    w: &'a [u8],
    pos: usize,
}

impl Words<'_> {
    fn next(&mut self) -> u32 {
        let v = u16::from_le_bytes([self.w[2 * self.pos], self.w[2 * self.pos + 1]]) as u32;
        self.pos += 1;
        v
    }
}

/// Decode `len` symbols of one way with `entry(slot, j) -> (sym, freq, rank)`, lane by lane.
fn ref_decode(
    words: &[u8],
    len: usize,
    bits: u32,
    mut entry: impl FnMut(u32, usize) -> (u32, u32, u32),
) -> Vec<u32> {
    let mut r = Words { w: words, pos: 0 };
    let mut st = [0u32; LANES];
    for x in st.iter_mut() {
        let hi = r.next();
        *x = hi << 16 | r.next();
    }
    let mut out = Vec::with_capacity(len);
    for j in 0..len {
        let l = j % LANES;
        let (sym, freq, rank) = entry(st[l] & ((1 << bits) - 1), j);
        let mut x = freq * (st[l] >> bits) + rank;
        if x < RANS_L {
            x = x << 16 | r.next();
        }
        st[l] = x;
        out.push(sym);
    }
    assert!(st.iter().all(|&x| x == RANS_L), "final states");
    assert_eq!(r.pos * 2, words.len(), "all words consumed");
    out
}

fn sm_entry(freqs: &[u16; 256], slot: u32) -> (u32, u32, u32) {
    let mut start = 0u32;
    for (s, &f) in freqs.iter().enumerate() {
        if slot < start + f as u32 {
            return (s as u32, f as u32, slot - start);
        }
        start += f as u32;
    }
    unreachable!("slot beyond table")
}

fn check_round_trip(
    cb: &Codebook,
    e: &[u8],
    h: &[u8],
    out: &[u8],
    info: &general_backend::rans::ChunkInfo,
) {
    let alias = ExpAlias::new(&cb.exp_freqs, cb.tier).unwrap();
    let mut at = 0usize;
    for w in 0..WAYS {
        let len = way_len(e.len(), w);
        let [ew, sw, esc] = info.index[w].map(|v| v as usize);
        let exp_words = &out[at..at + 2 * ew];
        at += 2 * ew;
        let codes = ref_decode(exp_words, len, EXP_BITS, |slot, _| {
            let v = alias.flat(slot);
            (v & 0xFF, ((v >> 8) & 0xFFF) + 1, v >> 20)
        });
        let mut real = Vec::new();
        for j in 0..len {
            let x = e[global(w, j)];
            if cb.exp_freqs[x as usize] > 0 {
                assert_eq!(codes[j], x as u32, "way {w} element {j}");
            } else {
                assert_eq!(Some(codes[j] as u8), cb.esc, "way {w} element {j} escapes");
                real.push(x);
            }
        }
        if let Some(sm) = &cb.sm {
            let sm_words = &out[at..at + 2 * sw];
            at += 2 * sw;
            let syms = ref_decode(sm_words, len, SM_BITS, |slot, j| {
                sm_entry(
                    &sm.freqs[sm.ctx_of[e[global(w, j)] as usize] as usize],
                    slot,
                )
            });
            for j in 0..len {
                assert_eq!(syms[j], h[global(w, j)] as u32, "way {w} sm element {j}");
            }
        } else {
            assert_eq!(sw, 0);
        }
        assert_eq!(&out[at..at + esc], &real[..], "way {w} escape bytes");
        at += esc;
    }
    assert_eq!(at, info.len);
}

#[test]
fn every_tier_matches_portable_and_round_trips() {
    for p in load_profiles() {
        for d in &p.dtypes {
            let Some(fmt) = format_of(&d.dtype) else {
                continue;
            };
            let seed = test_seed(d.seed);
            let (e, h) = planes(fmt, &Sampler::new(d, seed).sample(test_bytes(4 << 20)));
            for mode in [SmMode::Raw, SmMode::Coded] {
                if mode == SmMode::Coded && fmt.hi_bits() != 8 {
                    continue;
                }
                let cb = codebook(&e, &h, mode);
                let t = Box::new(EncTables::new(&cb).unwrap());
                for n in [
                    1usize,
                    15,
                    16,
                    17,
                    63,
                    64,
                    65,
                    1000,
                    4096 + 7,
                    e.len().min(1 << 20),
                ] {
                    let (ce, ch) = (&e[..n], &h[..n]);
                    let mut scratch = vec![0u16; encode_scratch_len(n)];
                    let mut want = vec![0u8; encode_bound(n, t.coded)];
                    let wi = portable()
                        .encode_chunk(&t, ce, ch, &mut scratch, &mut want)
                        .unwrap();
                    check_round_trip(&cb, ce, ch, &want, &wi);
                    for tier in available() {
                        let mut got = vec![0u8; encode_bound(n, t.coded)];
                        let gi = tier
                            .encode_chunk(&t, ce, ch, &mut scratch, &mut got)
                            .unwrap();
                        assert_eq!(
                            gi,
                            wi,
                            "{} {} {} n={n} seed={seed}",
                            tier.tier_name(),
                            p.source.file,
                            d.dtype
                        );
                        assert!(
                            got[..gi.len] == want[..wi.len],
                            "{} {} n={n} seed={seed}",
                            tier.tier_name(),
                            d.dtype
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn escapes_round_trip() {
    let n = 50_000;
    let mut s = 17u64;
    let e: Vec<u8> = (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            if s.is_multiple_of(20) {
                (s >> 20) as u8 % 120
            } else {
                60 + (s >> 30) as u8 % 8
            }
        })
        .collect();
    let h: Vec<u8> = (0..n).map(|i| (i * 31 % 256) as u8).collect();
    for mode in [SmMode::Raw, SmMode::Coded] {
        let cb = codebook(&e, &h, mode);
        assert!(cb.esc.is_some());
        let t = Box::new(EncTables::new(&cb).unwrap());
        let mut scratch = vec![0u16; encode_scratch_len(n)];
        let mut out = vec![0u8; encode_bound(n, t.coded)];
        let info = portable()
            .encode_chunk(&t, &e, &h, &mut scratch, &mut out)
            .unwrap();
        check_round_trip(&cb, &e, &h, &out, &info);
        for tier in available() {
            let mut got = vec![0u8; encode_bound(n, t.coded)];
            assert_eq!(
                tier.encode_chunk(&t, &e, &h, &mut scratch, &mut got)
                    .unwrap(),
                info
            );
            assert!(got[..info.len] == out[..info.len], "{}", tier.tier_name());
        }
    }
}

#[test]
fn errors() {
    let e = vec![60u8; 100];
    let h = vec![1u8; 100];
    let cb = codebook(&e, &h, SmMode::Coded);
    let t = Box::new(EncTables::new(&cb).unwrap());
    for tier in available() {
        let mut scratch = vec![0u16; encode_scratch_len(100)];
        let mut out = vec![0u8; encode_bound(100, true)];
        assert_eq!(
            tier.encode_chunk(&t, &e, &h[..99], &mut scratch, &mut out),
            Err(OpError::Corrupt)
        );
        assert_eq!(
            tier.encode_chunk(&t, &e, &h, &mut scratch[..10], &mut out),
            Err(OpError::OutputTooSmall)
        );
        assert_eq!(
            tier.encode_chunk(&t, &e, &h, &mut scratch, &mut out[..10]),
            Err(OpError::OutputTooSmall)
        );
        let mut other = e.clone();
        other[50] = 61;
        assert_eq!(
            tier.encode_chunk(&t, &other, &h, &mut scratch, &mut out),
            Err(OpError::Corrupt)
        );
        let mut bad_h = h.clone();
        bad_h[3] = 2;
        assert_eq!(
            tier.encode_chunk(&t, &e, &bad_h, &mut scratch, &mut out),
            Err(OpError::Corrupt)
        );
        let info = tier
            .encode_chunk(&t, &[], &[], &mut scratch, &mut out)
            .unwrap();
        assert_eq!(info.len, WAYS * 2 * 2 * 2 * LANES);
    }
}
