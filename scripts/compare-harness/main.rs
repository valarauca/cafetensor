//! Gate 6 harness: compares the old project with every available tier of the new one on
//! generated samples. Built by scripts/compare-old.sh in a temporary directory outside both
//! repositories, never inside them.

use std::hint::black_box;
use std::time::Instant;

use cafetensor_testkit::{Sampler, load_profiles, test_bytes, test_seed};
use general_backend::{Operations, available};

fn best<T>(reps: usize, mut f: impl FnMut() -> T) -> (T, f64) {
    let mut out = f();
    let mut t = f64::MAX;
    for _ in 0..reps {
        let s = Instant::now();
        out = black_box(f());
        t = t.min(s.elapsed().as_secs_f64());
    }
    (out, t)
}

fn gbps(bytes: usize, secs: f64) -> String {
    format!("{:.2}", bytes as f64 / secs / 1e9)
}

fn crc32c(tiers: &[&'static dyn Operations]) {
    println!("| sample | old (dispatch) GB/s | {} | equal |", tiers.iter().map(|t| format!("{} GB/s", t.tier_name())).collect::<Vec<_>>().join(" | "));
    println!("|---|---|{}---|", "---|".repeat(tiers.len()));
    for p in load_profiles() {
        for d in &p.dtypes {
            let sample = Sampler::new(d, test_seed(d.seed)).sample(test_bytes(256 << 20));
            let (want, t_old) = best(5, || tensor_compressor::crc::update(!0, &sample));
            let mut cells = Vec::new();
            let mut equal = true;
            for t in tiers {
                let (got, secs) = best(5, || t.crc32c_update(!0, &sample));
                equal &= got == want;
                cells.push(gbps(sample.len(), secs));
            }
            println!("| {} {} | {} | {} | {} |", p.source.file, d.dtype, gbps(sample.len(), t_old), cells.join(" | "), if equal { "yes" } else { "NO" });
            assert!(equal, "CRC mismatch on {} {}", p.source.file, d.dtype);
        }
    }
}

fn main() {
    let tiers = available();
    match std::env::args().nth(1).as_deref() {
        Some("crc32c") => crc32c(&tiers),
        other => panic!("unknown op {other:?}"),
    }
}
