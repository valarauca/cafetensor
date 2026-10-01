use cafetensor_testkit::{Sampler, load_profiles, test_bytes, test_seed};
use general_backend::codebook::{Codebook, SmMode};
use general_backend::decode::{ChunkRef, DecTables};
use general_backend::rans::{ChunkInfo, EncTables, WAYS, encode_bound, encode_scratch_len};
use general_backend::{Format, OpError, Operations, available, plane_len};

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

/// One encoded chunk with its raw planes, as the container stores it.
struct Encoded {
    elems: usize,
    streams: Vec<u8>,
    info: ChunkInfo,
    hi: Vec<u8>,
    bits: Vec<u8>,
    lo: Vec<u8>,
}

impl Encoded {
    fn chunk(&self) -> ChunkRef<'_> {
        let mut exp: [&[u8]; WAYS] = [&[]; WAYS];
        let mut sm: [&[u8]; WAYS] = [&[]; WAYS];
        let mut esc: [&[u8]; WAYS] = [&[]; WAYS];
        let mut at = 0;
        for w in 0..WAYS {
            let [e, s, x] = self.info.index[w].map(|v| v as usize);
            exp[w] = &self.streams[at..at + 2 * e];
            at += 2 * e;
            sm[w] = &self.streams[at..at + 2 * s];
            at += 2 * s;
            esc[w] = &self.streams[at..at + x];
            at += x;
        }
        ChunkRef {
            elems: self.elems,
            exp,
            sm,
            esc,
            hi: &self.hi,
            bits: &self.bits,
            lo: &self.lo,
        }
    }
}

struct Tensor {
    fmt: Format,
    dec: Box<DecTables>,
    chunks: Vec<Encoded>,
    chunk_elems: usize,
}

fn encode(fmt: Format, bytes: &[u8], mode: SmMode, chunk_elems: usize) -> Tensor {
    let t = portable();
    let n = bytes.len() / fmt.width;
    let (mut e, mut h, mut l) = (vec![0; n], vec![0; n], vec![0; n * fmt.low_bytes()]);
    t.split_planes(fmt, bytes, &mut e, &mut h, &mut l).unwrap();
    let mut joint = Box::new([0u64; 65536]);
    for (&x, &y) in e.iter().zip(&h) {
        joint[(x as usize) << 8 | y as usize] += 1;
    }
    let mode = if fmt.hi_bits() == 8 {
        mode
    } else {
        SmMode::Raw
    };
    let cb = Codebook::build(&joint, mode).unwrap();
    let enc = Box::new(EncTables::new(&cb).unwrap());
    let dec = Box::new(DecTables::new(&cb).unwrap());
    let mut chunks = Vec::new();
    let low = fmt.low_bytes();
    for start in (0..n).step_by(chunk_elems) {
        let end = (start + chunk_elems).min(n);
        let k = end - start;
        let mut scratch = vec![0u16; encode_scratch_len(k)];
        let mut streams = vec![0u8; encode_bound(k, enc.coded)];
        let info = t
            .encode_chunk(
                &enc,
                &e[start..end],
                &h[start..end],
                &mut scratch,
                &mut streams,
            )
            .unwrap();
        streams.truncate(info.len);
        let (hi, bits) = match (enc.coded, fmt.hi_bits()) {
            (true, _) => (Vec::new(), Vec::new()),
            (false, 8) => (h[start..end].to_vec(), Vec::new()),
            (false, b) => {
                let mut bits = vec![0u8; b as usize * plane_len(k)];
                t.pack_bitplanes(&h[start..end], b, &mut bits).unwrap();
                (Vec::new(), bits)
            }
        };
        chunks.push(Encoded {
            elems: k,
            streams,
            info,
            hi,
            bits,
            lo: l[start * low..end * low].to_vec(),
        });
    }
    Tensor {
        fmt,
        dec,
        chunks,
        chunk_elems,
    }
}

fn decode(tier: &dyn Operations, t: &Tensor, out: &mut [u8]) -> Result<(), OpError> {
    let w = t.fmt.width;
    for (c, dst) in t.chunks.iter().zip(out.chunks_mut(t.chunk_elems * w)) {
        tier.decode_chunk(&t.dec, t.fmt, c.chunk(), dst)?;
    }
    Ok(())
}

#[test]
fn round_trip_on_every_tier() {
    for p in load_profiles() {
        for d in &p.dtypes {
            let Some(fmt) = format_of(&d.dtype) else {
                continue;
            };
            let seed = test_seed(d.seed);
            let sample = Sampler::new(d, seed).sample(test_bytes(4 << 20));
            for mode in [SmMode::Raw, SmMode::Coded] {
                for chunk in [1 << 20, 4096 + 16, 1000] {
                    let t = encode(fmt, &sample, mode, chunk);
                    for tier in available() {
                        let mut back = vec![0xA5u8; sample.len() + 64];
                        for off in [0usize, 1, 2, 31] {
                            let out = &mut back[off..off + sample.len()];
                            decode(tier, &t, out).unwrap_or_else(|e| {
                                panic!(
                                    "{} {} {mode:?} chunk={chunk}: {e:?}",
                                    tier.tier_name(),
                                    d.dtype
                                )
                            });
                            assert!(
                                out == &sample[..],
                                "{} {} {} {mode:?} chunk={chunk} off={off} seed={seed}",
                                tier.tier_name(),
                                p.source.file,
                                d.dtype
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn small_chunks_and_escapes() {
    let mut s = 23u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    for fmt in [
        Format::BF16,
        Format::F32,
        Format::F16,
        Format::E4M3,
        Format::E5M2,
    ] {
        for n in [1usize, 15, 16, 17, 63, 64, 65, 127, 128, 129, 1000] {
            for wide in [false, true] {
                let bytes: Vec<u8> = (0..n * fmt.width)
                    .map(|i| {
                        if wide || i % fmt.width != fmt.width - 1 {
                            next() as u8
                        } else {
                            0x3F + (next() % 3) as u8
                        }
                    })
                    .collect();
                for mode in [SmMode::Raw, SmMode::Coded] {
                    let t = encode(fmt, &bytes, mode, 64);
                    for tier in available() {
                        let mut out = vec![0u8; bytes.len()];
                        decode(tier, &t, &mut out).unwrap();
                        assert_eq!(
                            out,
                            bytes,
                            "{} {fmt:?} n={n} wide={wide} {mode:?}",
                            tier.tier_name()
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn corruption_is_rejected_without_panicking() {
    let p = &load_profiles()[0];
    let d = &p.dtypes[0];
    let fmt = format_of(&d.dtype).unwrap();
    let sample = Sampler::new(d, test_seed(d.seed)).sample(1 << 18);
    for mode in [SmMode::Raw, SmMode::Coded] {
        let clean = encode(fmt, &sample, mode, 1 << 15);
        let mut s = 99u64;
        let mut detected = 0;
        let trials = 300;
        for _ in 0..trials {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let mut t = encode(fmt, &sample, mode, 1 << 15);
            let c = (s % t.chunks.len() as u64) as usize;
            let len = t.chunks[c].streams.len();
            t.chunks[c].streams[(s >> 16) as usize % len] ^= 1 << ((s >> 40) % 8);
            for tier in available() {
                let mut out = vec![0u8; sample.len()];
                if decode(tier, &t, &mut out).is_err() {
                    detected += 1;
                }
            }
        }
        let total = trials * available().len();
        assert!(
            detected * 10 > total * 9,
            "{mode:?}: only {detected}/{total} stream corruptions detected"
        );
        let mut out = vec![0u8; sample.len()];
        decode(portable(), &clean, &mut out).unwrap();
        assert_eq!(out, sample);
        let short = &mut vec![0u8; clean.chunk_elems * fmt.width - 1];
        assert_eq!(
            portable().decode_chunk(&clean.dec, fmt, clean.chunks[0].chunk(), short),
            Err(OpError::OutputTooSmall)
        );
    }
}
