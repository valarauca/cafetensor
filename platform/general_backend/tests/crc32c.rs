use cafetensor_testkit::{Sampler, load_profiles, test_bytes, test_seed};
use general_backend::{Operations, available};

fn reference(state: u32, data: &[u8]) -> u32 {
    let mut c = state;
    for &b in data {
        c ^= b as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 {
                (c >> 1) ^ 0x82F6_3B78
            } else {
                c >> 1
            };
        }
    }
    c
}

fn portable() -> &'static dyn Operations {
    *available().last().expect("portable")
}

fn bytes(n: usize, seed: u64) -> Vec<u8> {
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
fn known_vectors() {
    for t in available() {
        assert_eq!(
            !t.crc32c_update(!0, b"123456789"),
            0xE306_9283,
            "{}",
            t.tier_name()
        );
        assert_eq!(!t.crc32c_update(!0, b""), 0, "{}", t.tier_name());
    }
}

#[test]
fn edge_lengths_offsets_and_states() {
    let data = bytes(70_000, 9);
    let lens = [
        0usize,
        1,
        7,
        8,
        9,
        63,
        64,
        65,
        255,
        256,
        257,
        511,
        512,
        1023,
        1024,
        4096 + 13,
        65_537,
    ];
    for t in available() {
        for &len in &lens {
            for off in [0usize, 1, 3, 31, 63] {
                let d = &data[off..off + len];
                for state in [!0u32, 0, 0x1234_5678] {
                    assert_eq!(
                        t.crc32c_update(state, d),
                        reference(state, d),
                        "{} len={len} off={off} state={state:x}",
                        t.tier_name()
                    );
                }
            }
        }
    }
}

#[test]
fn every_tier_matches_portable_on_profile_samples() {
    for p in load_profiles() {
        for d in &p.dtypes {
            let seed = test_seed(d.seed);
            let sample = Sampler::new(d, seed).sample(test_bytes(8 << 20));
            let want = portable().crc32c_update(!0, &sample);
            for t in available() {
                assert_eq!(
                    t.crc32c_update(!0, &sample),
                    want,
                    "{} {} {} seed={seed}",
                    t.tier_name(),
                    p.source.file,
                    d.dtype
                );
                let split = sample.len() / 3;
                let chained =
                    t.crc32c_update(t.crc32c_update(!0, &sample[..split]), &sample[split..]);
                assert_eq!(
                    chained,
                    want,
                    "{} chained {} seed={seed}",
                    t.tier_name(),
                    d.dtype
                );
            }
        }
    }
}
