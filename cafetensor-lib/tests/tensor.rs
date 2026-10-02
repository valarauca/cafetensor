use cafetensor_lib::SmMode;
use cafetensor_lib::tensor::{
    DEFAULT_BLOCK, DType, Options, compress_bytes, decompress_tensor, decompressed_len, dtype,
    has_crc, is_passthrough, is_raw,
};
use cafetensor_testkit::profile::entropy;
use cafetensor_testkit::{Layout, Sampler, load_profiles, test_bytes, test_seed};

const FLOATS: [DType; 5] = [
    DType::Bf16,
    DType::F32,
    DType::F16,
    DType::F8E4M3,
    DType::F8E5M2,
];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn roundtrip(dtype: DType, bytes: &[u8], opts: &Options) -> Vec<u8> {
    let packed = compress_bytes(dtype, bytes, opts).unwrap();
    assert_eq!(
        decompressed_len(&packed).unwrap() * dtype.width(),
        bytes.len()
    );
    assert_eq!(has_crc(&packed).unwrap(), opts.crc);
    let mut region = vec![0xA5u8; bytes.len() + 64];
    for off in [0usize, 1, 3, 61] {
        let out = &mut region[off..off + bytes.len()];
        decompress_tensor(&packed, out).unwrap();
        assert!(
            out == bytes,
            "{dtype:?} {opts:?} off={off} n={}",
            bytes.len()
        );
    }
    packed
}

#[test]
fn profile_samples_round_trip() {
    for p in load_profiles() {
        for d in &p.dtypes {
            let Some(dt) = DType::from_name(&d.dtype) else {
                continue;
            };
            let seed = test_seed(d.seed);
            let sample = Sampler::new(d, seed).sample(test_bytes(8 << 20));
            for mode in [SmMode::Raw, SmMode::Coded] {
                for block in [1 << 16, DEFAULT_BLOCK] {
                    let opts = Options::new(block, mode);
                    let packed = roundtrip(dt, &sample, &opts);
                    assert_eq!(dtype(&packed).unwrap(), dt);
                    assert_eq!(is_passthrough(&packed).unwrap(), dt.format().is_none());
                }
            }
        }
    }
}

/// Gate 4 ratio check. A generated sample is drawn from the pooled histogram, so its ratio is
/// compared with its own order-0 bound (coded exponent plus raw residual). For the old
/// project's ratio on the real file, the sample is compressed in pieces of the profile's mean
/// tensor size, so per-tensor overhead counts as it did there, and the old ratio is shifted by
/// the pooled minus per-tensor exponent entropy.
#[test]
fn ratio_within_profile_tolerance() {
    for p in load_profiles() {
        for d in &p.dtypes {
            let (Some(dt), Some(layout), Some(old)) = (
                DType::from_name(&d.dtype),
                Layout::of(&d.dtype),
                d.old_ratio,
            ) else {
                continue;
            };
            let seed = test_seed(d.seed);
            let sample = Sampler::new(d, seed).sample(test_bytes(16 << 20));
            let packed = compress_bytes(dt, &sample, &Options::default()).unwrap();
            let ratio = packed.len() as f64 / sample.len() as f64;
            let n = sample.len() / layout.width;
            let mut exps = [0u64; 256];
            for i in 0..n {
                exps[layout.split(layout.load(&sample, i)).0 as usize] += 1;
            }
            let bits = 8.0 * layout.width as f64;
            let residual = (layout.hi_bits() + 8 * layout.low_bytes() as u32) as f64;
            let bound = ((entropy(exps) + residual) / bits).min(1.0);
            let shifted = old + (d.entropy_exp_bits - d.entropy_exp_bits_per_tensor) / bits;
            let piece = (d.elements_total / d.tensors.max(1) as u64).max(1) as usize * layout.width;
            let pieces: usize = sample
                .chunks(piece)
                .map(|t| compress_bytes(dt, t, &Options::default()).unwrap().len())
                .sum();
            let ratio_pieces = pieces as f64 / sample.len() as f64;
            println!(
                "{} {}: ratio {ratio:.4} bound {bound:.4} pieces {ratio_pieces:.4} old {old:.4} shifted {shifted:.4}",
                p.source.file, d.dtype
            );
            assert!(
                ratio - bound <= d.ratio_tolerance && bound - ratio <= 1e-3,
                "{} {} seed={seed}: ratio {ratio:.4} vs own bound {bound:.4}",
                p.source.file,
                d.dtype
            );
            assert!(
                (ratio_pieces - shifted).abs() <= d.ratio_tolerance,
                "{} {} seed={seed}: ratio {ratio_pieces:.4} vs old {old:.4} shifted to {shifted:.4}",
                p.source.file,
                d.dtype
            );
        }
    }
}

#[test]
fn every_float_dtype_at_small_lengths() {
    let mut r = Rng(91);
    for dt in FLOATS {
        let fmt = dt.format().unwrap();
        let emax = (1u64 << fmt.exp_bits()) - 1;
        for n in (0usize..=100).chain([127, 128, 129, 1000, 4097]) {
            let bytes: Vec<u8> = (0..n)
                .flat_map(|_| {
                    let e = emax / 2 - (r.next() % 6).min(r.next() % 6);
                    let v = fmt.join(e as u8, r.next() as u8, r.next() as u32);
                    v.to_le_bytes()[..fmt.width].to_vec()
                })
                .collect();
            for mode in [SmMode::Raw, SmMode::Coded] {
                for crc in [false, true] {
                    let opts = Options {
                        crc,
                        ..Options::new(64, mode)
                    };
                    roundtrip(dt, &bytes, &opts);
                }
            }
        }
    }
}

#[test]
fn passthrough_empty_and_random_tensors() {
    let mut r = Rng(5);
    for dt in [
        DType::F64,
        DType::Bool,
        DType::U8,
        DType::I8,
        DType::U16,
        DType::I16,
        DType::U32,
        DType::I32,
        DType::U64,
        DType::I64,
    ] {
        let bytes: Vec<u8> = (0..dt.width() * 1001).map(|_| r.next() as u8).collect();
        let packed = roundtrip(dt, &bytes, &Options::default());
        assert!(is_passthrough(&packed).unwrap() && is_raw(&packed).unwrap());
    }
    for dt in FLOATS {
        let packed = roundtrip(dt, &[], &Options::default());
        assert!(is_raw(&packed).unwrap() && !is_passthrough(&packed).unwrap());
        let noise: Vec<u8> = (0..dt.width() * 50_000).map(|_| r.next() as u8).collect();
        assert!(is_raw(&roundtrip(dt, &noise, &Options::default())).unwrap());
    }
    assert!(compress_bytes(DType::F32, &[0; 6], &Options::default()).is_err());
}

#[test]
fn corruption_and_truncation_are_rejected() {
    let p = load_profiles();
    let d = p
        .iter()
        .flat_map(|p| &p.dtypes)
        .find(|d| d.dtype == "BF16")
        .unwrap();
    let sample = Sampler::new(d, test_seed(d.seed)).sample(1 << 18);
    let mut r = Rng(1234);
    let mut out = vec![0u8; sample.len()];
    for mode in [SmMode::Raw, SmMode::Coded] {
        for crc in [false, true] {
            let opts = Options {
                crc,
                ..Options::new(1 << 15, mode)
            };
            let packed = compress_bytes(DType::Bf16, &sample, &opts).unwrap();
            for cut in [0, 3, 10, packed.len() / 2, packed.len() - 1] {
                assert!(decompress_tensor(&packed[..cut], &mut out).is_err());
            }
            let mut detected = 0;
            for _ in 0..300 {
                let mut bad = packed.clone();
                for _ in 0..1 + r.next() % 3 {
                    let i = (r.next() % bad.len() as u64) as usize;
                    bad[i] ^= 1 << (r.next() % 8);
                }
                detected += usize::from(decompress_tensor(&bad, &mut out).is_err());
            }
            if crc {
                assert_eq!(detected, 300, "{mode:?}: a corruption passed the CRC");
            }
        }
    }
}
