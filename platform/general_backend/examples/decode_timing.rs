//! Interleaved per-tier decode timing on one profile sample.
//!
//!   cargo run --release -p general_backend --example decode_timing -- [dtype] [raw|coded] [file]
use std::time::Instant;

use cafetensor_testkit::{Sampler, load_profiles, test_seed};
use general_backend::codebook::{Codebook, SmMode};
use general_backend::decode::{ChunkRef, DecTables};
use general_backend::rans::{EncTables, WAYS, encode_bound, encode_scratch_len};
use general_backend::{Format, available, plane_len};

const CHUNK: usize = 1 << 20;

fn main() {
    let dtype = std::env::args().nth(1).unwrap_or_else(|| "BF16".into());
    let mode = match std::env::args().nth(2).as_deref() {
        Some("coded") => SmMode::Coded,
        _ => SmMode::Raw,
    };
    let fmt = match dtype.as_str() {
        "BF16" => Format::BF16,
        "F32" => Format::F32,
        "F16" => Format::F16,
        "F8_E4M3" => Format::E4M3,
        "F8_E5M2" => Format::E5M2,
        other => panic!("unknown dtype {other}"),
    };
    let file = std::env::args().nth(3).unwrap_or_default();
    let profiles = load_profiles();
    let d = profiles
        .iter()
        .filter(|p| p.source.file.contains(&file))
        .flat_map(|p| &p.dtypes)
        .find(|d| d.dtype == dtype)
        .unwrap();
    let sample = Sampler::new(d, test_seed(d.seed)).sample(64 << 20);
    let tiers = available();
    let ops = *tiers.last().unwrap();
    let n = sample.len() / fmt.width;
    let low = fmt.low_bytes();
    let (mut e, mut h, mut l) = (vec![0; n], vec![0; n], vec![0; n * low]);
    ops.split_planes(fmt, &sample, &mut e, &mut h, &mut l)
        .unwrap();
    let mut joint = Box::new([0u64; 65536]);
    for (&x, &y) in e.iter().zip(&h) {
        joint[(x as usize) << 8 | y as usize] += 1;
    }
    let cb = Codebook::build(&joint, mode).unwrap();
    let enc = Box::new(EncTables::new(&cb).unwrap());
    let dec = Box::new(DecTables::new(&cb).unwrap());
    let mut chunks = Vec::new();
    let mut scratch = vec![0u16; encode_scratch_len(CHUNK)];
    for start in (0..n).step_by(CHUNK) {
        let end = (start + CHUNK).min(n);
        let mut buf = vec![0u8; encode_bound(end - start, enc.coded)];
        let info = ops
            .encode_chunk(&enc, &e[start..end], &h[start..end], &mut scratch, &mut buf)
            .unwrap();
        let mut bits = vec![0u8; fmt.hi_bits() as usize * plane_len(end - start)];
        let hi = match (enc.coded, fmt.hi_bits()) {
            (true, _) => Vec::new(),
            (false, 8) => h[start..end].to_vec(),
            (false, b) => {
                ops.pack_bitplanes(&h[start..end], b, &mut bits).unwrap();
                Vec::new()
            }
        };
        if fmt.hi_bits() == 8 {
            bits.clear();
        }
        buf.truncate(info.len);
        chunks.push((
            end - start,
            buf,
            info,
            hi,
            bits,
            l[start * low..end * low].to_vec(),
        ));
    }
    let mut out = vec![0u8; sample.len()];
    let mut best = vec![f64::MAX; tiers.len()];
    for round in 0..6 {
        let order: Vec<usize> = if round % 2 == 0 {
            (0..tiers.len()).collect()
        } else {
            (0..tiers.len()).rev().collect()
        };
        for i in order {
            out.fill(0);
            let s = Instant::now();
            for ((elems, buf, info, hi, bits, lo), dst) in
                chunks.iter().zip(out.chunks_mut(CHUNK * fmt.width))
            {
                let mut parts = [[&[][..]; 3]; WAYS];
                let mut at = 0;
                for (w, p) in parts.iter_mut().enumerate() {
                    for (k, q) in p.iter_mut().enumerate() {
                        let len = info.index[w][k] as usize * if k < 2 { 2 } else { 1 };
                        *q = &buf[at..at + len];
                        at += len;
                    }
                }
                let chunk = ChunkRef {
                    elems: *elems,
                    exp: parts.map(|p| p[0]),
                    sm: parts.map(|p| p[1]),
                    esc: parts.map(|p| p[2]),
                    hi,
                    bits,
                    lo,
                };
                tiers[i].decode_chunk(&dec, fmt, chunk, dst).unwrap();
            }
            best[i] = best[i].min(s.elapsed().as_secs_f64());
            assert!(
                out == sample,
                "{} decoded wrong bytes",
                tiers[i].tier_name()
            );
        }
    }
    for (t, b) in tiers.iter().zip(best) {
        println!(
            "{:16} {:.2} GB/s",
            t.tier_name(),
            sample.len() as f64 / b / 1e9
        );
    }
}
