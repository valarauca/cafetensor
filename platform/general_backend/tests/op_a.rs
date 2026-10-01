use general_backend::{OpError, Operations, available};

fn data(n: usize, seed: u64) -> Vec<u8> {
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

fn portable() -> &'static dyn Operations {
    *available().last().expect("portable is always available")
}

#[test]
fn every_tier_matches_portable() {
    let tiers = available();
    assert_eq!(portable().tier_name(), "portable");
    for half in [0usize, 1, 31, 63, 64, 65, 127, 128, 1000, 4096 + 7, 1 << 20] {
        let input = data(2 * half, half as u64 + 11);
        let mut want = vec![0u8; half];
        assert_eq!(portable().op_a(&input, &mut want), Ok(half));
        let expected: Vec<u8> = input[..half]
            .iter()
            .zip(&input[half..])
            .map(|(a, b)| a ^ b)
            .collect();
        assert_eq!(want, expected, "portable len={half}");
        for t in &tiers {
            let mut got = vec![0u8; half];
            assert_eq!(t.op_a(&input, &mut got), Ok(half), "{}", t.tier_name());
            assert_eq!(got, want, "{} len={half}", t.tier_name());
        }
    }
}

#[test]
fn misaligned_slices() {
    let backing = data(4096 + 128, 3);
    for t in available() {
        for off in 1..64 {
            let input = &backing[off..off + 2 * 1000];
            let mut out_backing = vec![0u8; 1000 + 64];
            let out = &mut out_backing[off..off + 1000];
            assert_eq!(t.op_a(input, out), Ok(1000), "{} off={off}", t.tier_name());
            for i in 0..1000 {
                assert_eq!(
                    out[i],
                    input[i] ^ input[1000 + i],
                    "{} off={off} i={i}",
                    t.tier_name()
                );
            }
        }
    }
}

#[test]
fn errors() {
    for t in available() {
        let mut out = [0u8; 4];
        assert_eq!(
            t.op_a(&[1, 2, 3], &mut out),
            Err(OpError::Corrupt),
            "{}",
            t.tier_name()
        );
        assert_eq!(
            t.op_a(&[0u8; 10], &mut out),
            Err(OpError::OutputTooSmall),
            "{}",
            t.tier_name()
        );
        assert_eq!(t.op_a(&[], &mut []), Ok(0), "{}", t.tier_name());
    }
}

#[test]
fn selection_is_best_available() {
    let tiers = available();
    let names: Vec<_> = tiers.iter().map(|t| t.tier_name()).collect();
    if std::env::var("CAFETENSOR_TIER").is_err() {
        assert_eq!(general_backend::operations().tier_name(), names[0]);
    }
    if let Ok(expect) = std::env::var("CAFETENSOR_EXPECT_TIER") {
        assert_eq!(names[0], expect, "available: {names:?}");
    }
}
