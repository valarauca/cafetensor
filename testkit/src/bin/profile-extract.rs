//! Reads local safetensors files and writes one distribution profile per source to
//! `testdata/profiles/` (or `--out DIR`). Tensor data never leaves the machine; only histograms
//! and summary statistics are written.
//!
//! ```text
//! profile-extract [--out DIR] [--old-tcz PATH] FILE.safetensors...
//! ```
//!
//! With `--old-tcz`, the old project's `tcz bench` is run on each file and its per-tensor
//! ratios are summed per dtype into `old_ratio`.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::process::Command;

use cafetensor_testkit::layout::{Layout, width};
use cafetensor_testkit::profile::{Accum, Profile, SourceInfo};
use safetensors::tensor::Metadata;

const PIECE: usize = 64 << 20;

fn main() {
    let mut args = std::env::args().skip(1);
    let mut out = cafetensor_testkit::profiles_dir();
    let mut old_tcz: Option<PathBuf> = None;
    let mut files = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--out" => out = PathBuf::from(args.next().expect("--out DIR")),
            "--old-tcz" => old_tcz = Some(PathBuf::from(args.next().expect("--old-tcz PATH"))),
            _ => files.push(PathBuf::from(a)),
        }
    }
    if files.is_empty() {
        eprintln!("usage: profile-extract [--out DIR] [--old-tcz PATH] FILE.safetensors...");
        std::process::exit(2);
    }
    std::fs::create_dir_all(&out).expect("output directory");
    for f in &files {
        let profile = extract(f, old_tcz.as_deref());
        let dest = out.join(format!("{}.toml", profile_name(f)));
        std::fs::write(&dest, profile.to_toml()).expect("write profile");
        for d in &profile.dtypes {
            eprintln!(
                "{}: {} {} tensors {} elements H(exp) {:.4} bound {:.4} old ratio {:?}",
                dest.display(),
                d.dtype,
                d.tensors,
                d.elements_total,
                d.entropy_exp_bits,
                d.entropy_bits,
                d.old_ratio
            );
        }
    }
}

/// A profile file name unique per source: the Hugging Face repo (`models--org--name` in a
/// cache path) or the parent directory, then the file stem.
fn profile_name(path: &Path) -> String {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .expect("file name");
    let repo = path
        .components()
        .filter_map(|c| c.as_os_str().to_str())
        .find_map(|c| c.strip_prefix("models--"))
        .map(str::to_string)
        .or_else(|| {
            path.parent()
                .and_then(|p| p.file_name())
                .and_then(|n| n.to_str())
                .map(str::to_string)
        })
        .unwrap_or_default();
    format!("{repo}--{stem}")
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || "-_.".contains(c) {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn extract(path: &Path, old_tcz: Option<&Path>) -> Profile {
    let mut f = File::open(path).expect("open source");
    let mut prefix = [0u8; 8];
    f.read_exact(&mut prefix).expect("header length");
    let mut json = vec![0u8; u64::from_le_bytes(prefix) as usize];
    f.read_exact(&mut json).expect("header");
    let meta: Metadata = serde_json::from_slice(&json).expect("safetensors header");
    let data_start = 8 + json.len() as u64;

    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .expect("file name")
        .to_string();
    let mut per_dtype: BTreeMap<String, (Accum, usize, u64, u64, f64, f64)> = BTreeMap::new();
    let mut buf = vec![0u8; PIECE];
    let tensors = meta.tensors();
    for info in tensors.values() {
        let dtype = format!("{:?}", info.dtype);
        let w = width(&dtype).unwrap_or_else(|| panic!("unsupported dtype {dtype}"));
        let (start, end) = info.data_offsets;
        let elems = ((end - start) / w) as u64;
        let layout = Layout::of(&dtype);
        let mut tensor = Accum::new(layout);
        let mut at = start;
        while at < end {
            let n = (end - at).min(PIECE / w * w);
            f.read_exact_at(&mut buf[..n], data_start + at as u64)
                .expect("tensor data");
            tensor.add(&buf[..n], w);
            at += n;
        }
        let (h_exp, h) = tensor.entropies();
        let entry = per_dtype
            .entry(dtype)
            .or_insert_with(|| (Accum::new(layout), 0, u64::MAX, 0, 0.0, 0.0));
        entry.0.merge(&tensor);
        entry.1 += 1;
        entry.2 = entry.2.min(elems);
        entry.3 = entry.3.max(elems);
        entry.4 += h_exp * elems as f64;
        entry.5 += h * elems as f64;
    }

    let mut hasher = blake3::Hasher::new();
    hasher.update_mmap_rayon(path).expect("hash source");
    let digest = hasher.finalize();
    let old = old_tcz.map(|t| old_ratios(t, path)).unwrap_or_default();
    let seed_base = u64::from_le_bytes(
        blake3::hash(file_name.as_bytes()).as_bytes()[..8]
            .try_into()
            .unwrap(),
    );

    let dtypes = per_dtype
        .into_iter()
        .enumerate()
        .map(|(i, (dtype, (acc, n, min, max, he, h)))| {
            let total = acc.elements.max(1) as f64;
            let per_tensor = (he / total, h / total);
            let mut p = acc.into_profile(
                &dtype,
                n,
                min,
                max,
                seed_base.wrapping_add(i as u64),
                per_tensor,
            );
            p.old_ratio = old.get(&dtype).copied();
            p
        })
        .collect();
    Profile {
        source: SourceInfo {
            file: file_name,
            blake3: format!("blake3-{}", digest.to_hex()),
            bytes: std::fs::metadata(path).expect("stat").len(),
            tensors: tensors.len(),
        },
        dtypes,
    }
}

/// Per-dtype compressed/raw ratio from the old project's `tcz bench` output, whose tensor lines
/// end in `<dtype> <raw bytes> <ratio>x`.
fn old_ratios(tcz: &Path, source: &Path) -> BTreeMap<String, f64> {
    let out = Command::new(tcz)
        .arg("bench")
        .arg(source)
        .output()
        .expect("run old tcz bench");
    assert!(
        out.status.success(),
        "old tcz bench failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let mut sums: BTreeMap<String, (f64, f64)> = BTreeMap::new();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let words: Vec<&str> = line.split_whitespace().collect();
        let [.., dtype, bytes, ratio] = words.as_slice() else {
            continue;
        };
        let (Ok(bytes), Some(Ok(ratio))) = (
            bytes.parse::<f64>(),
            ratio.strip_suffix('x').map(str::parse::<f64>),
        ) else {
            continue;
        };
        if width(dtype).is_none() {
            continue;
        }
        let e = sums.entry(dtype.to_string()).or_default();
        e.0 += bytes;
        e.1 += bytes * ratio;
    }
    sums.into_iter()
        .filter(|(_, (raw, _))| *raw > 0.0)
        .map(|(d, (raw, packed))| (d, packed / raw))
        .collect()
}
