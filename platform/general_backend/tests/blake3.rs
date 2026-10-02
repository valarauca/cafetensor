use blake3::hazmat::HasherExt;
use cafetensor_testkit::{test_bytes, test_seed};
use general_backend::blake3::CHUNK_LEN;
use general_backend::{Operations, available};

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

fn left_len(len: usize) -> usize {
    CHUNK_LEN << ((len - 1) / CHUNK_LEN).ilog2()
}

#[test]
fn hash_matches_reference_at_every_boundary() {
    let k = CHUNK_LEN;
    let lens = [
        0,
        1,
        63,
        64,
        65,
        127,
        128,
        k - 1,
        k,
        k + 1,
        2 * k - 1,
        2 * k,
        2 * k + 1,
        3 * k + 1,
        15 * k,
        16 * k - 1,
        16 * k,
        16 * k + 1,
        17 * k,
        31 * k + 5,
        32 * k,
        255 * k,
        256 * k - 1,
        256 * k,
        256 * k + 1,
        257 * k,
        300 * k + 7,
        512 * k,
        1 << 20,
        (1 << 20) + 1,
        3_000_017,
    ];
    let data = bytes(3_000_100, test_seed(1));
    for tier in available() {
        for n in lens {
            for off in [0, 1, 7] {
                let input = &data[off..off + n];
                assert_eq!(
                    tier.blake3_hash(input),
                    *blake3::hash(input).as_bytes(),
                    "{} len={n} off={off}",
                    tier.tier_name()
                );
            }
        }
    }
}

#[test]
fn random_lengths_match_reference() {
    let max = test_bytes(2 << 20);
    let seed = test_seed(2);
    let data = bytes(max + 64, seed);
    let mut s = seed | 1;
    for _ in 0..64 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let n = (s % (max as u64 + 1)) as usize;
        let off = (s >> 40) as usize % 64;
        let want = *blake3::hash(&data[off..off + n]).as_bytes();
        for tier in available() {
            assert_eq!(
                tier.blake3_hash(&data[off..off + n]),
                want,
                "{} len={n} off={off} seed={seed}",
                tier.tier_name()
            );
        }
    }
}

fn tree(ops: &dyn Operations, data: &[u8], counter: u64, root: bool) -> [u8; 32] {
    if data.len() <= 4 * CHUNK_LEN {
        return if root {
            ops.blake3_hash(data)
        } else {
            ops.blake3_subtree(data, counter)
        };
    }
    let left = left_len(data.len());
    let l = tree(ops, &data[..left], counter, false);
    let r = tree(
        ops,
        &data[left..],
        counter + (left / CHUNK_LEN) as u64,
        false,
    );
    ops.blake3_parent(&l, &r, root)
}

#[test]
fn subtrees_and_parents_compose_to_the_hash() {
    let data = bytes(1_100_000, test_seed(3));
    for tier in available() {
        for n in [
            4 * CHUNK_LEN + 1,
            37 * CHUNK_LEN,
            300 * CHUNK_LEN + 9,
            1_100_000,
        ] {
            assert_eq!(
                tree(tier, &data[..n], 0, true),
                *blake3::hash(&data[..n]).as_bytes(),
                "{} len={n}",
                tier.tier_name()
            );
        }
        for (chunks, counter) in [(1, 0), (1, 5), (16, 16), (64, 128), (256, 256), (512, 512)] {
            let input = &data[..chunks * CHUNK_LEN];
            let mut h = blake3::Hasher::new();
            h.set_input_offset(counter * CHUNK_LEN as u64);
            h.update(input);
            assert_eq!(
                tier.blake3_subtree(input, counter),
                h.finalize_non_root(),
                "{} chunks={chunks} counter={counter}",
                tier.tier_name()
            );
        }
    }
}
