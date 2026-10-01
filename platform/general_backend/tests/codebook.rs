use cafetensor_testkit::{Sampler, load_profiles, test_bytes, test_seed};
use general_backend::codebook::{CODEBOOK_MAX_BYTES, Codebook, ExpAlias, SmMode, Tier, normalize};
use general_backend::{Format, OpError, available};

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

fn planes(fmt: Format, bytes: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let n = bytes.len() / fmt.width;
    let (mut e, mut h, mut l) = (vec![0; n], vec![0; n], vec![0; n * fmt.low_bytes()]);
    available()
        .last()
        .unwrap()
        .split_planes(fmt, bytes, &mut e, &mut h, &mut l)
        .unwrap();
    (e, h)
}

#[test]
fn histogram_matches_reference_on_every_tier() {
    for p in load_profiles() {
        for d in &p.dtypes {
            let Some(fmt) = format_of(&d.dtype) else {
                continue;
            };
            let seed = test_seed(d.seed);
            let (e, h) = planes(fmt, &Sampler::new(d, seed).sample(test_bytes(4 << 20)));
            let mut want = vec![0u32; 65536];
            for (&x, &y) in e.iter().zip(&h) {
                want[(x as usize) << 8 | y as usize] += 1;
            }
            for t in available() {
                let mut got = Box::new([0u32; 65536]);
                t.histogram(&e, &h, &mut got).unwrap();
                assert!(
                    got[..] == want[..],
                    "{} {} {} seed={seed}",
                    t.tier_name(),
                    p.source.file,
                    d.dtype
                );
            }
        }
    }
    for t in available() {
        let mut j = Box::new([0u32; 65536]);
        assert_eq!(t.histogram(&[1, 2], &[3], &mut j), Err(OpError::Corrupt));
        assert_eq!(t.histogram(&[], &[], &mut j), Ok(()));
    }
}

#[test]
fn codebooks_build_and_round_trip() {
    for p in load_profiles() {
        for d in &p.dtypes {
            let Some(fmt) = format_of(&d.dtype) else {
                continue;
            };
            let seed = test_seed(d.seed);
            let (e, h) = planes(fmt, &Sampler::new(d, seed).sample(test_bytes(4 << 20)));
            let mut joint = Box::new([0u64; 65536]);
            for (&x, &y) in e.iter().zip(&h) {
                joint[(x as usize) << 8 | y as usize] += 1;
            }
            for mode in [SmMode::Raw, SmMode::Coded] {
                let mode = if fmt.hi_bits() == 8 {
                    mode
                } else {
                    SmMode::Raw
                };
                let cb = Codebook::build(&joint, mode)
                    .unwrap_or_else(|| panic!("{} {} {mode:?}", p.source.file, d.dtype));
                assert_eq!(cb.exp_freqs.iter().map(|&f| f as u32).sum::<u32>(), 4096);
                let mut buf = vec![0u8; CODEBOOK_MAX_BYTES];
                let n = cb.write(&mut buf).unwrap();
                let (back, used) = Codebook::read(&buf[..n]).unwrap();
                assert_eq!(
                    (back, used),
                    (cb.clone(), n),
                    "{} {} {mode:?}",
                    p.source.file,
                    d.dtype
                );
                assert_eq!(cb.write(&mut buf[..n - 1]), Err(OpError::OutputTooSmall));
                for cut in [0, 1, 32, n / 2, n - 1] {
                    assert!(Codebook::read(&buf[..cut]).is_err(), "truncated at {cut}");
                }
                let mut s = seed | 1;
                for _ in 0..200 {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    let mut bad = buf[..n].to_vec();
                    bad[(s % n as u64) as usize] ^= 1 << ((s >> 32) % 8);
                    let _ = Codebook::read(&bad);
                }
            }
        }
    }
    assert!(Codebook::build(&Box::new([0u64; 65536]), SmMode::Raw).is_none());
}

#[test]
fn alias_layout_is_a_bijection() {
    let mut s = 5u64;
    for trial in 0..300 {
        let k = 1 + (trial % 64);
        let mut counts = [0u64; 256];
        for _ in 0..k {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            counts[(s % 256) as usize] += 1 + (s >> 40) % 1000;
        }
        let f = normalize(&counts, 12);
        let alphabet = f.iter().filter(|&&v| v > 0).count();
        let tier = Tier::fitting(alphabet).unwrap();
        let a = ExpAlias::new(&f, tier).expect("alias");
        let mut seen = vec![[false; 4096]; 256];
        for slot in 0..4096 {
            let e = a.flat(slot);
            let (sym, freq, rank) = ((e & 0xFF) as usize, ((e >> 8) & 0xFFF) + 1, e >> 20);
            assert_eq!(freq, f[sym] as u32);
            assert!(rank < freq && !seen[sym][rank as usize]);
            seen[sym][rank as usize] = true;
        }
    }
}
