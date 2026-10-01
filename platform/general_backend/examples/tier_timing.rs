//! Interleaved per-tier timing of one operation, to separate codegen from measurement order.
use std::time::Instant;

fn main() {
    let data: Vec<u8> = (0..256u32 << 20)
        .map(|i| (i.wrapping_mul(2654435761) >> 13) as u8)
        .collect();
    let tiers = general_backend::available();
    let mut best = vec![f64::MAX; tiers.len()];
    for round in 0..6 {
        let order: Vec<usize> = if round % 2 == 0 {
            (0..tiers.len()).collect()
        } else {
            (0..tiers.len()).rev().collect()
        };
        for i in order {
            let s = Instant::now();
            std::hint::black_box(tiers[i].crc32c_update(!0, std::hint::black_box(&data)));
            best[i] = best[i].min(s.elapsed().as_secs_f64());
        }
    }
    for (t, b) in tiers.iter().zip(best) {
        println!(
            "{:16} {:.2} GB/s",
            t.tier_name(),
            data.len() as f64 / b / 1e9
        );
    }
}
