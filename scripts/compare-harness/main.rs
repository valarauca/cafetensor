//! Gate 6 harness: compares the old project with every available tier of the new one on
//! generated samples. Built by scripts/compare-old.sh in a temporary directory outside both
//! repositories, never inside them.

use std::hint::black_box;
use std::time::Instant;

use cafetensor_testkit::{Sampler, load_profiles, test_bytes, test_seed};
use general_backend::{Format, Operations, available, plane_len};

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

fn format_of(dtype: &str) -> Option<(Format, tensor_compressor::codec::Format)> {
    let new = match dtype {
        "BF16" => Format::BF16,
        "F32" => Format::F32,
        "F16" => Format::F16,
        "F8_E4M3" => Format::E4M3,
        "F8_E5M2" => Format::E5M2,
        _ => return None,
    };
    Some((new, tensor_compressor::codec::Format { width: new.width, mant_bits: new.mant_bits }))
}

/// The old project's split loop (`compress_bytes`), single threaded.
#[inline(never)]
fn old_split(fmt: tensor_compressor::codec::Format, bytes: &[u8], e: &mut [u8], h: &mut [u8], l: &mut [u8]) {
    let low = fmt.low_bytes();
    for i in 0..e.len() {
        let (x, y, z) = fmt.split(fmt.load(bytes, i));
        e[i] = x;
        h[i] = y;
        l[i * low..(i + 1) * low].copy_from_slice(&z.to_le_bytes()[..low]);
    }
}

/// The old project's bitplane packing loop (`compress_bytes`).
#[inline(never)]
fn old_bitplanes(his: &[u8], hbits: u32, out: &mut Vec<u8>) {
    out.clear();
    for j in 0..hbits {
        for g in his.chunks(16) {
            let word = g.iter().enumerate().fold(0u16, |acc, (l, &v)| acc | ((((v >> j) & 1) as u16) << l));
            out.extend_from_slice(&word.to_le_bytes());
        }
    }
}

fn split(tiers: &[&'static dyn Operations]) {
    println!("| sample | old split GB/s | {} | equal | old bitplanes GB/s | {} |",
        tiers.iter().map(|t| format!("{} GB/s", t.tier_name())).collect::<Vec<_>>().join(" | "),
        tiers.iter().map(|t| format!("{} bitplanes", t.tier_name())).collect::<Vec<_>>().join(" | "));
    println!("|---|---|{}---|---|{}", "---|".repeat(tiers.len()), "---|".repeat(tiers.len()));
    for p in load_profiles() {
        for d in &p.dtypes {
            let Some((fmt, ofmt)) = format_of(&d.dtype) else { continue };
            let sample = Sampler::new(d, test_seed(d.seed)).sample(test_bytes(256 << 20));
            let n = sample.len() / fmt.width;
            let (mut e0, mut h0, mut l0) = (vec![0u8; n], vec![0u8; n], vec![0u8; n * fmt.low_bytes()]);
            let (_, t_old) = best(3, || old_split(ofmt, &sample, &mut e0, &mut h0, &mut l0));
            let mut cells = Vec::new();
            let mut equal = true;
            for t in tiers {
                let (mut e, mut h, mut l) = (vec![0u8; n], vec![0u8; n], vec![0u8; n * fmt.low_bytes()]);
                let (_, secs) = best(3, || t.split_planes(fmt, &sample, &mut e, &mut h, &mut l).unwrap());
                equal &= e == e0 && h == h0 && l == l0;
                cells.push(gbps(sample.len(), secs));
            }
            let mut bits = Vec::new();
            let (old_bits, new_bits) = if fmt.hi_bits() < 8 {
                let mut want = Vec::new();
                let (_, t) = best(3, || old_bitplanes(&h0, fmt.hi_bits(), &mut want));
                let mut cells = Vec::new();
                for tier in tiers {
                    let mut out = vec![0u8; fmt.hi_bits() as usize * plane_len(n)];
                    let (_, secs) = best(3, || tier.pack_bitplanes(&h0, fmt.hi_bits(), &mut out).unwrap());
                    equal &= out == want;
                    cells.push(gbps(n, secs));
                }
                bits = cells;
                (gbps(n, t), bits.join(" | "))
            } else {
                ("n/a".into(), vec!["n/a"; tiers.len()].join(" | "))
            };
            let _ = &bits;
            println!("| {} {} | {} | {} | {} | {} | {} |", p.source.file, d.dtype, gbps(sample.len(), t_old), cells.join(" | "),
                if equal { "yes" } else { "NO" }, old_bits, new_bits);
            assert!(equal, "split mismatch on {} {}", p.source.file, d.dtype);
        }
    }
}

fn main() {
    let tiers = available();
    match std::env::args().nth(1).as_deref() {
        Some("crc32c") => crc32c(&tiers),
        Some("split") => split(&tiers),
        other => panic!("unknown op {other:?}"),
    }
}
