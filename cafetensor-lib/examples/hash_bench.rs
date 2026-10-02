//! Parallel BLAKE3 throughput against the `blake3` crate on the same rayon pool.
//!
//!   cargo run --release -p cafetensor-lib --example hash_bench -- [MiB]
use std::time::Instant;

fn main() {
    let mib: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(1024);
    let data: Vec<u8> = (0..mib << 20).map(|i| (i * 7 + (i >> 13)) as u8).collect();
    let mut best = [f64::MAX; 4];
    for _ in 0..5 {
        let s = Instant::now();
        let a = cafetensor_lib::hash::hash(&data);
        best[0] = best[0].min(s.elapsed().as_secs_f64());
        let s = Instant::now();
        let mut h = blake3::Hasher::new();
        h.update_rayon(&data);
        let b = *h.finalize().as_bytes();
        best[1] = best[1].min(s.elapsed().as_secs_f64());
        let s = Instant::now();
        let c = *blake3::hash(&data).as_bytes();
        best[2] = best[2].min(s.elapsed().as_secs_f64());
        let s = Instant::now();
        let d = general_backend::operations().blake3_hash(&data);
        best[3] = best[3].min(s.elapsed().as_secs_f64());
        assert!(a == b && b == c && c == d);
    }
    let gbps = |t: f64| data.len() as f64 / t / 1e9;
    println!(
        "{} threads: cafetensor {:.2} GB/s, blake3 update_rayon {:.2} GB/s, blake3 one thread {:.2} GB/s, tier one call {:.2} GB/s",
        rayon::current_num_threads(),
        gbps(best[0]),
        gbps(best[1]),
        gbps(best[2]),
        gbps(best[3])
    );
}
