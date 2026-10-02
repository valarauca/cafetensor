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

fn codebook(tiers: &[&'static dyn Operations]) {
    use general_backend::codebook::{CODEBOOK_MAX_BYTES, Codebook, SmMode};
    use tensor_compressor::codec::{Codebook as OldCodebook, SmMode as OldMode};
    println!("| sample | mode | old build (2 threads) GB/s | {} | codebook bytes equal |",
        tiers.iter().map(|t| format!("{} histogram+build GB/s", t.tier_name())).collect::<Vec<_>>().join(" | "));
    println!("|---|---|---|{}---|", "---|".repeat(tiers.len()));
    for p in load_profiles() {
        for d in &p.dtypes {
            let Some((fmt, ofmt)) = format_of(&d.dtype) else { continue };
            let sample = Sampler::new(d, test_seed(d.seed)).sample(test_bytes(256 << 20));
            let n = sample.len() / fmt.width;
            let (mut e, mut h, mut l) = (vec![0u8; n], vec![0u8; n], vec![0u8; n * fmt.low_bytes()]);
            old_split(ofmt, &sample, &mut e, &mut h, &mut l);
            for (mode, omode) in [(SmMode::Raw, OldMode::Raw), (SmMode::Coded, OldMode::Coded)] {
                if mode == SmMode::Coded && fmt.hi_bits() != 8 {
                    continue;
                }
                let (old, t_old) = best(3, || {
                    let cb = OldCodebook::build(&e, &h, omode).expect("old codebook");
                    let mut out = Vec::new();
                    cb.write(&mut out);
                    out
                });
                let mut cells = Vec::new();
                let mut equal = true;
                for t in tiers {
                    let (new, secs) = best(3, || {
                        let mut joint = vec![0u64; 65536];
                        let mut part = Box::new([0u32; 65536]);
                        for (ce, ch) in e.chunks(1 << 20).zip(h.chunks(1 << 20)) {
                            part.fill(0);
                            t.histogram(ce, ch, &mut part).unwrap();
                            joint.iter_mut().zip(part.iter()).for_each(|(a, &b)| *a += b as u64);
                        }
                        let cb = Codebook::build(joint.as_slice().try_into().unwrap(), mode).expect("new codebook");
                        let mut out = vec![0u8; CODEBOOK_MAX_BYTES];
                        let k = cb.write(&mut out).unwrap();
                        out.truncate(k);
                        out
                    });
                    equal &= new == old;
                    cells.push(gbps(n, secs));
                }
                println!("| {} {} | {mode:?} | {} | {} | {} |", p.source.file, d.dtype, gbps(n, t_old), cells.join(" | "), if equal { "yes" } else { "NO" });
                assert!(equal, "codebook mismatch on {} {} {mode:?}", p.source.file, d.dtype);
            }
        }
    }
}

/// The old project's per-chunk encoder, serialized in the container's per-way order.
#[inline(never)]
fn old_encode(t: &tensor_compressor::codec::EncTables, e: &[u8], h: &[u8], cs: &mut tensor_compressor::codec::ChunkStreams, out: &mut Vec<u8>) {
    tensor_compressor::codec::encode_chunk(t, e, h, cs);
    for w in 0..4 {
        for &v in cs.exp[w].iter().chain(&cs.sm[w]) {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&cs.esc[w]);
    }
}

fn encode(tiers: &[&'static dyn Operations]) {
    use general_backend::codebook::{Codebook, SmMode};
    use general_backend::rans::{EncTables, encode_bound, encode_scratch_len};
    use tensor_compressor::codec::{ChunkStreams, Codebook as OldCodebook, EncTables as OldTables, SmMode as OldMode};
    const CHUNK: usize = 1 << 20;
    println!("| sample | mode | old G elem/s | {} | equal |", tiers.iter().map(|t| format!("{} G elem/s", t.tier_name())).collect::<Vec<_>>().join(" | "));
    println!("|---|---|---|{}---|", "---|".repeat(tiers.len()));
    for p in load_profiles() {
        for d in &p.dtypes {
            let Some((fmt, ofmt)) = format_of(&d.dtype) else { continue };
            let sample = Sampler::new(d, test_seed(d.seed)).sample(test_bytes(64 << 20));
            let n = sample.len() / fmt.width;
            let (mut e, mut h, mut l) = (vec![0u8; n], vec![0u8; n], vec![0u8; n * fmt.low_bytes()]);
            old_split(ofmt, &sample, &mut e, &mut h, &mut l);
            for (mode, omode) in [(SmMode::Raw, OldMode::Raw), (SmMode::Coded, OldMode::Coded)] {
                if mode == SmMode::Coded && fmt.hi_bits() != 8 {
                    continue;
                }
                let old_tables = OldTables::new(&OldCodebook::build(&e, &h, omode).unwrap());
                let mut joint = vec![0u64; 65536];
                for (&x, &y) in e.iter().zip(&h) {
                    joint[(x as usize) << 8 | y as usize] += 1;
                }
                let cb = Codebook::build(joint.as_slice().try_into().unwrap(), mode).unwrap();
                let tables = Box::new(EncTables::new(&cb).unwrap());
                let (old, t_old) = best(2, || {
                    let mut cs = ChunkStreams::default();
                    let mut out = Vec::new();
                    for (ce, ch) in e.chunks(CHUNK).zip(h.chunks(CHUNK)) {
                        old_encode(&old_tables, ce, ch, &mut cs, &mut out);
                    }
                    out
                });
                let mut cells = Vec::new();
                let mut equal = true;
                for t in tiers {
                    let mut scratch = vec![0u16; encode_scratch_len(CHUNK)];
                    let mut buf = vec![0u8; encode_bound(CHUNK, tables.coded)];
                    let (new, secs) = best(2, || {
                        let mut out = Vec::new();
                        for (ce, ch) in e.chunks(CHUNK).zip(h.chunks(CHUNK)) {
                            let info = t.encode_chunk(&tables, ce, ch, &mut scratch, &mut buf).unwrap();
                            out.extend_from_slice(&buf[..info.len]);
                        }
                        out
                    });
                    equal &= new == old;
                    cells.push(gbps(n, secs));
                }
                println!("| {} {} | {mode:?} | {} | {} | {} |", p.source.file, d.dtype, gbps(n, t_old), cells.join(" | "), if equal { "yes" } else { "NO" });
                assert!(equal, "encode mismatch on {} {} {mode:?}", p.source.file, d.dtype);
            }
        }
    }
}

/// One chunk encoded by the old encoder, with its raw planes, as the old container stores it.
struct OldChunk {
    elems: usize,
    exp: [Vec<u8>; 4],
    sm: [Vec<u8>; 4],
    esc: [Vec<u8>; 4],
    hi: Vec<u8>,
    bits: Vec<u8>,
    lo: Vec<u8>,
}

fn words(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// The old project's per-chunk decode (`tensor.rs`): AVX-512 lines when available, then the
/// scalar tail.
#[inline(never)]
fn old_decode(t: &tensor_compressor::codec::DecTables, fmt: tensor_compressor::codec::Format, c: &OldChunk, dst: &mut [u8]) {
    use tensor_compressor::codec::{ChunkState, Planes};
    let planes = Planes { hi: &c.hi, bits: &c.bits, lo: &c.lo };
    let exp = std::array::from_fn(|w| c.exp[w].as_slice());
    let sm = std::array::from_fn(|w| c.sm[w].as_slice());
    let esc = std::array::from_fn(|w| c.esc[w].as_slice());
    let mut st = ChunkState::new(fmt, t.mode, c.elems, exp, sm, esc, planes).unwrap();
    let first = if tensor_compressor::avx512::available() { tensor_compressor::avx512::decode_chunk(t, &mut st, dst) } else { 0 };
    st.decode_scalar(t, dst, first);
    st.finish().unwrap();
}

fn decode(tiers: &[&'static dyn Operations]) {
    use general_backend::codebook::{Codebook, SmMode};
    use general_backend::decode::{ChunkRef, DecTables};
    use tensor_compressor::codec::{ChunkStreams, Codebook as OldCodebook, DecTables as OldDec, EncTables as OldTables, SmMode as OldMode};
    const CHUNK: usize = 1 << 20;
    println!("| sample | mode | old GB/s | {} | equal |", tiers.iter().map(|t| format!("{} GB/s", t.tier_name())).collect::<Vec<_>>().join(" | "));
    println!("|---|---|---|{}---|", "---|".repeat(tiers.len()));
    for p in load_profiles() {
        for d in &p.dtypes {
            let Some((fmt, ofmt)) = format_of(&d.dtype) else { continue };
            let sample = Sampler::new(d, test_seed(d.seed)).sample(test_bytes(256 << 20));
            let n = sample.len() / fmt.width;
            let low = fmt.low_bytes();
            let (mut e, mut h, mut l) = (vec![0u8; n], vec![0u8; n], vec![0u8; n * low]);
            old_split(ofmt, &sample, &mut e, &mut h, &mut l);
            for (mode, omode) in [(SmMode::Raw, OldMode::Raw), (SmMode::Coded, OldMode::Coded)] {
                if mode == SmMode::Coded && fmt.hi_bits() != 8 {
                    continue;
                }
                let ocb = OldCodebook::build(&e, &h, omode).unwrap();
                let old_enc = OldTables::new(&ocb);
                let old_dec = OldDec::new(&ocb).unwrap();
                let mut joint = vec![0u64; 65536];
                for (&x, &y) in e.iter().zip(&h) {
                    joint[(x as usize) << 8 | y as usize] += 1;
                }
                let cb = Codebook::build(joint.as_slice().try_into().unwrap(), mode).unwrap();
                let dec = Box::new(DecTables::new(&cb).unwrap());
                let mut chunks = Vec::new();
                let mut cs = ChunkStreams::default();
                for start in (0..n).step_by(CHUNK) {
                    let end = (start + CHUNK).min(n);
                    tensor_compressor::codec::encode_chunk(&old_enc, &e[start..end], &h[start..end], &mut cs);
                    let (hi, bits) = match (mode, fmt.hi_bits()) {
                        (SmMode::Coded, _) => (Vec::new(), Vec::new()),
                        (_, 8) => (h[start..end].to_vec(), Vec::new()),
                        (_, b) => {
                            let mut bits = Vec::new();
                            old_bitplanes(&h[start..end], b, &mut bits);
                            (Vec::new(), bits)
                        }
                    };
                    chunks.push(OldChunk {
                        elems: end - start,
                        exp: std::array::from_fn(|w| words(&cs.exp[w])),
                        sm: std::array::from_fn(|w| words(&cs.sm[w])),
                        esc: cs.esc.clone(),
                        hi,
                        bits,
                        lo: l[start * low..end * low].to_vec(),
                    });
                }
                let mut out = vec![0u8; sample.len()];
                let (_, t_old) = best(5, || {
                    for (c, dst) in chunks.iter().zip(out.chunks_mut(CHUNK * fmt.width)) {
                        old_decode(&old_dec, ofmt, c, dst);
                    }
                });
                let mut equal = out == sample;
                let mut cells = Vec::new();
                for t in tiers {
                    out.fill(0);
                    let (_, secs) = best(5, || {
                        for (c, dst) in chunks.iter().zip(out.chunks_mut(CHUNK * fmt.width)) {
                            let chunk = ChunkRef {
                                elems: c.elems,
                                exp: std::array::from_fn(|w| c.exp[w].as_slice()),
                                sm: std::array::from_fn(|w| c.sm[w].as_slice()),
                                esc: std::array::from_fn(|w| c.esc[w].as_slice()),
                                hi: &c.hi,
                                bits: &c.bits,
                                lo: &c.lo,
                            };
                            t.decode_chunk(&dec, fmt, chunk, dst).unwrap();
                        }
                    });
                    equal &= out == sample;
                    cells.push(gbps(sample.len(), secs));
                }
                println!("| {} {} | {mode:?} | {} | {} | {} |", p.source.file, d.dtype, gbps(sample.len(), t_old), cells.join(" | "), if equal { "yes" } else { "NO" });
                assert!(equal, "decode mismatch on {} {} {mode:?}", p.source.file, d.dtype);
            }
        }
    }
}

fn main() {
    let tiers = available();
    match std::env::args().nth(1).as_deref() {
        Some("crc32c") => crc32c(&tiers),
        Some("split") => split(&tiers),
        Some("codebook") => codebook(&tiers),
        Some("encode") => encode(&tiers),
        Some("decode") => decode(&tiers),
        other => panic!("unknown op {other:?}"),
    }
}
