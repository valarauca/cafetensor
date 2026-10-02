use cafetensor_lib::hash::{Hasher, b3sum, hash, hash_string};
use cafetensor_testkit::test_seed;

fn bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 24) as u8
        })
        .collect()
}

#[test]
fn streaming_matches_reference_for_any_split() {
    let seed = test_seed(11);
    let data = bytes(5 << 20, seed);
    let mut s = seed | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let k = 1024;
    let lens = [
        0,
        1,
        k - 1,
        k,
        k + 1,
        2 * k,
        3 * k,
        16 * k,
        255 * k + 7,
        256 * k,
        (1 << 20) + 1,
        5 << 20,
    ];
    for n in lens {
        let want = *blake3::hash(&data[..n]).as_bytes();
        assert_eq!(hash(&data[..n]), want, "len={n}");
        for trial in 0..8 {
            let mut h = Hasher::new();
            let mut at = 0;
            while at < n {
                let step = match trial % 4 {
                    0 => 1 + (next() % 100) as usize,
                    1 => k * (1 + (next() % 4) as usize),
                    2 => 1 + (next() % (3 << 20)) as usize,
                    _ => 1 + (next() % (64 * k as u64)) as usize,
                };
                let end = (at + step).min(n);
                h.update(&data[at..end]);
                at = end;
            }
            h.update(&[]);
            assert_eq!(h.finalize(), want, "len={n} trial={trial} seed={seed}");
        }
    }
}

#[test]
fn b3sum_matches_reference() {
    let dir = std::env::temp_dir().join(format!("cafetensor-hash-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for n in [0usize, 1000, (1 << 26) + 4097] {
        let data = bytes(n, 7 + n as u64);
        let path = dir.join(format!("f{n}"));
        std::fs::write(&path, &data).unwrap();
        let want = hash_string(blake3::hash(&data).as_bytes());
        assert_eq!(b3sum(&path).unwrap(), want);
        assert_eq!(want, format!("blake3-{}", blake3::hash(&data).to_hex()));
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
