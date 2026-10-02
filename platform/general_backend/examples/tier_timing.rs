//! Interleaved per-tier timing of one operation, to separate codegen from measurement order.
//!
//!   cargo run --release -p general_backend --example tier_timing -- [crc32c|histogram|blake3]
use std::time::Instant;

fn main() {
    let op = std::env::args().nth(1).unwrap_or_else(|| "crc32c".into());
    let n = 256usize << 20;
    let data: Vec<u8> = (0..n as u32)
        .map(|i| (i.wrapping_mul(2654435761) >> 13) as u8)
        .collect();
    let exps: Vec<u8> = data.iter().map(|&b| 100 + (b % 7).min(b / 37)).collect();
    let tiers = general_backend::available();
    let mut best = vec![f64::MAX; tiers.len()];
    let mut joint = Box::new([0u32; 65536]);
    for round in 0..6 {
        let order: Vec<usize> = if round % 2 == 0 {
            (0..tiers.len()).collect()
        } else {
            (0..tiers.len()).rev().collect()
        };
        for i in order {
            let s = Instant::now();
            match op.as_str() {
                "crc32c" => {
                    std::hint::black_box(tiers[i].crc32c_update(!0, std::hint::black_box(&data)));
                }
                "histogram" => {
                    for (e, h) in exps.chunks(1 << 20).zip(data.chunks(1 << 20)) {
                        tiers[i].histogram(e, h, &mut joint).unwrap();
                    }
                    std::hint::black_box(&joint);
                }
                "blake3" => {
                    std::hint::black_box(tiers[i].blake3_hash(std::hint::black_box(&data)));
                }
                other => panic!("unknown op {other}"),
            }
            best[i] = best[i].min(s.elapsed().as_secs_f64());
        }
    }
    if op == "blake3" {
        let mut crate_best = f64::MAX;
        for _ in 0..6 {
            let s = Instant::now();
            std::hint::black_box(blake3::hash(std::hint::black_box(&data)));
            crate_best = crate_best.min(s.elapsed().as_secs_f64());
        }
        println!(
            "{:16} {:.2} G/s",
            "blake3 crate",
            n as f64 / crate_best / 1e9
        );
    }
    for (t, b) in tiers.iter().zip(best) {
        println!("{:16} {:.2} G/s", t.tier_name(), n as f64 / b / 1e9);
    }
}
