use cafetensor_testkit::{Sampler, load_profiles, test_bytes, test_seed};
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

const ALL: [Format; 5] = [
    Format::BF16,
    Format::F32,
    Format::F16,
    Format::E4M3,
    Format::E5M2,
];

type Planes = (Vec<u8>, Vec<u8>, Vec<u8>);

fn reference(fmt: Format, bytes: &[u8]) -> Planes {
    let (mut e, mut h, mut l) = (Vec::new(), Vec::new(), Vec::new());
    for el in bytes.chunks_exact(fmt.width) {
        let v = el.iter().rev().fold(0u32, |a, &b| (a << 8) | b as u32);
        let (x, y, z) = fmt.split(v);
        e.push(x);
        h.push(y);
        l.extend_from_slice(&z.to_le_bytes()[..fmt.low_bytes()]);
        assert_eq!(fmt.join(x, y, z), v);
    }
    (e, h, l)
}

fn split(t: &dyn Operations, fmt: Format, bytes: &[u8]) -> Planes {
    let n = bytes.len() / fmt.width;
    let (mut e, mut h, mut l) = (
        vec![0xAA; n],
        vec![0xAA; n],
        vec![0xAA; n * fmt.low_bytes()],
    );
    t.split_planes(fmt, bytes, &mut e, &mut h, &mut l)
        .expect("split");
    (e, h, l)
}

fn ref_bitplanes(his: &[u8], bits: u32) -> Vec<u8> {
    let mut out = Vec::new();
    for j in 0..bits {
        for g in his.chunks(16) {
            let w = g
                .iter()
                .enumerate()
                .fold(0u16, |a, (l, &v)| a | ((((v >> j) & 1) as u16) << l));
            out.extend_from_slice(&w.to_le_bytes());
        }
    }
    out
}

fn noise(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s as u8
        })
        .collect()
}

#[test]
fn every_tier_matches_reference_on_profile_samples() {
    for p in load_profiles() {
        for d in &p.dtypes {
            let Some(fmt) = format_of(&d.dtype) else {
                continue;
            };
            let seed = test_seed(d.seed);
            let sample = Sampler::new(d, seed).sample(test_bytes(8 << 20));
            let want = reference(fmt, &sample);
            let bits = (fmt.hi_bits() < 8).then(|| ref_bitplanes(&want.1, fmt.hi_bits()));
            for t in available() {
                let got = split(t, fmt, &sample);
                assert!(
                    got == want,
                    "{} {} {} seed={seed}",
                    t.tier_name(),
                    p.source.file,
                    d.dtype
                );
                if let Some(bits) = &bits {
                    let mut out = vec![0u8; bits.len()];
                    t.pack_bitplanes(&got.1, fmt.hi_bits(), &mut out).unwrap();
                    assert!(
                        &out == bits,
                        "{} bitplanes {} seed={seed}",
                        t.tier_name(),
                        d.dtype
                    );
                }
            }
        }
    }
}

#[test]
fn edge_lengths_and_offsets() {
    let data = noise(4 * 1100, 5);
    for t in available() {
        for fmt in ALL {
            for n in [0usize, 1, 15, 16, 17, 63, 64, 65, 127, 128, 129, 1000] {
                for off in [0usize, 1, 7, 33, 63] {
                    let bytes = &data[off..off + n * fmt.width];
                    let want = reference(fmt, bytes);
                    assert!(
                        split(t, fmt, bytes) == want,
                        "{} {fmt:?} n={n} off={off}",
                        t.tier_name()
                    );
                    let bits = fmt.hi_bits().min(8);
                    let mut out = vec![0u8; bits as usize * plane_len(n)];
                    t.pack_bitplanes(&want.1, bits, &mut out).unwrap();
                    assert_eq!(
                        out,
                        ref_bitplanes(&want.1, bits),
                        "{} bitplanes {fmt:?} n={n}",
                        t.tier_name()
                    );
                }
            }
        }
    }
}

#[test]
fn errors() {
    for t in available() {
        let (mut e, mut h, mut l) = ([0u8; 4], [0u8; 4], [0u8; 8]);
        assert_eq!(
            t.split_planes(Format::BF16, &[0; 3], &mut e, &mut h, &mut l),
            Err(OpError::Corrupt)
        );
        assert_eq!(
            t.split_planes(Format::BF16, &[0; 10], &mut e, &mut h, &mut l),
            Err(OpError::OutputTooSmall)
        );
        assert_eq!(
            t.split_planes(Format::F32, &[0; 16], &mut e, &mut h, &mut l[..7]),
            Err(OpError::OutputTooSmall)
        );
        let bad = Format {
            width: 2,
            mant_bits: 5,
        };
        assert_eq!(
            t.split_planes(bad, &[0; 4], &mut e, &mut h, &mut l),
            Err(OpError::Corrupt)
        );
        assert_eq!(
            t.pack_bitplanes(&[0; 17], 3, &mut [0u8; 11]),
            Err(OpError::OutputTooSmall)
        );
        assert_eq!(
            t.pack_bitplanes(&[0; 17], 9, &mut [0u8; 64]),
            Err(OpError::Corrupt)
        );
        assert_eq!(
            t.split_planes(Format::F16, &[], &mut [], &mut [], &mut []),
            Ok(())
        );
    }
}
