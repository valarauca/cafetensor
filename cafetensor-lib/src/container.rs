//! The `.cafetensor` container: one or more `.safetensors` files gathered into a single
//! compressed file, byte-identical to the `tensor-compressor` project's.
//!
//! ```text
//!   u64    header length (little endian)
//!   JSON   header, padded with spaces to a multiple of 8
//!   data   source headers and compressed tensor blobs
//! ```
//!
//! The JSON follows the safetensors shape. Every tensor entry points at its compressed blob and
//! names its source file, and a `rasn_comp` entry describes the container:
//!
//! ```json
//! {
//!   "rasn_comp": {
//!     "version": "v1",
//!     "crc32c": true,
//!     "checksum": "blake3-<hex of every source file concatenated in order>",
//!     "block": 21,
//!     "sources": [
//!       { "name": "model-00001-of-00002.safetensors", "header": [0, 344],
//!         "checksum": "blake3-<hex>", "bytes": 5263851872 }
//!     ]
//!   },
//!   "model.embed_tokens.weight": {
//!     "dtype": "BF16", "shape": [151936, 5120], "data_offsets": [344, 1032000000], "source": 0
//!   }
//! }
//! ```
//!
//! `block` is the log2 of the uncompressed bytes per chunk. Each source header is stored
//! verbatim, so decompression rebuilds every source file byte for byte, which `checksum`
//! verifies. Data offsets are relative to the start of the data section, as in safetensors.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::hash::{HASH_PREFIX, Hasher, hash_string};
use crate::safetensors::Source;
use crate::tensor::{DType, Options, compress_bytes, decompress_tensor};
use crate::{Error, HugeBuf};

/// File extension of the container.
pub const EXTENSION: &str = "cafetensor";
const VERSION: &str = "v1";
const META_KEY: &str = "rasn_comp";

/// The `rasn_comp` header entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Meta {
    pub version: String,
    pub crc32c: bool,
    pub checksum: String,
    pub block: u32,
    pub sources: Vec<SourceMeta>,
}

/// One gathered source file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceMeta {
    pub name: String,
    pub header: [u64; 2],
    pub checksum: String,
    pub bytes: u64,
}

/// A tensor entry of the container header.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [u64; 2],
    pub source: usize,
}

/// A parsed container header.
#[derive(Debug, Clone)]
pub struct Header {
    pub meta: Meta,
    pub tensors: HashMap<String, Entry>,
    /// Absolute file offset of the data section.
    pub data_start: u64,
}

/// Counters and timings for one container operation.
#[derive(Debug, Clone, Default)]
pub struct Stats {
    pub tensors: usize,
    pub elements: u64,
    pub raw_bytes: u64,
    pub packed_bytes: u64,
    pub codec_time: Duration,
    pub hash_time: Duration,
    /// Pages backing the decode buffer, as `HugeBuf::backing` names them.
    pub backing: &'static str,
    pub checksum: String,
}

fn header_json(meta: &Meta, tensors: &[(String, Entry)]) -> Result<Vec<u8>, Error> {
    let json = |e: serde_json::Error| Error::Header(e.to_string());
    let mut map = serde_json::Map::new();
    map.insert(META_KEY.into(), serde_json::to_value(meta).map_err(json)?);
    for (name, e) in tensors {
        map.insert(name.clone(), serde_json::to_value(e).map_err(json)?);
    }
    serde_json::to_vec(&map).map_err(json)
}

fn file_name(p: &Path) -> Result<String, Error> {
    p.file_name()
        .and_then(|n| n.to_str())
        .map(str::to_string)
        .ok_or_else(|| Error::Layout(format!("{} has no usable file name", p.display())))
}

/// Gather the `.safetensors` files `inputs` into one container at `output`. When `verify` is
/// set every tensor is decoded again and compared before it is written.
pub fn compress(
    inputs: &[PathBuf],
    output: &Path,
    opts: &Options,
    verify: bool,
) -> Result<Stats, Error> {
    let block = opts.block_bytes.trailing_zeros();
    let sources = inputs
        .iter()
        .map(|p| Source::open(p))
        .collect::<Result<Vec<_>, _>>()?;
    let mut names = HashSet::new();
    let mut tensor_names = HashSet::new();
    let mut source_names = Vec::with_capacity(sources.len());
    for s in &sources {
        let name = file_name(&s.path)?;
        if !names.insert(name.clone()) {
            return Err(Error::Layout(format!(
                "duplicate source name {}",
                s.path.display()
            )));
        }
        source_names.push(name);
        for t in &s.tensors {
            if !tensor_names.insert(t.name.clone()) {
                return Err(Error::Layout(format!(
                    "tensor {} appears in more than one source",
                    t.name
                )));
            }
        }
    }

    let placeholder = format!("{HASH_PREFIX}{}", "0".repeat(64));
    let mut meta = Meta {
        version: VERSION.into(),
        crc32c: opts.crc,
        checksum: placeholder.clone(),
        block,
        sources: source_names
            .into_iter()
            .map(|name| SourceMeta {
                name,
                header: [u64::MAX; 2],
                checksum: placeholder.clone(),
                bytes: u64::MAX,
            })
            .collect(),
    };
    let mut entries: Vec<(String, Entry)> = sources
        .iter()
        .enumerate()
        .flat_map(|(i, s)| {
            s.tensors.iter().map(move |t| {
                let e = Entry {
                    dtype: t.dtype.name().into(),
                    shape: t.shape.clone(),
                    data_offsets: [u64::MAX; 2],
                    source: i,
                };
                (t.name.clone(), e)
            })
        })
        .collect();
    let reserved = (header_json(&meta, &entries)?.len() + 64).next_multiple_of(8);

    let mut out = File::create(output)?;
    out.seek(SeekFrom::Start(8 + reserved as u64))?;
    let mut at = 0u64;
    let mut stats = Stats::default();
    let mut total = Hasher::new();
    let mut bytes = Vec::new();
    let mut check = HugeBuf::new();
    let mut entry_at = 0;
    for (i, s) in sources.iter().enumerate() {
        let input = File::open(&s.path)?;
        out.write_all(&s.header)?;
        meta.sources[i].header = [at, at + s.header.len() as u64];
        at += s.header.len() as u64;
        stats.packed_bytes += s.header.len() as u64;
        let mut hasher = Hasher::new();
        let start = Instant::now();
        hasher.update(&s.header);
        if sources.len() > 1 {
            total.update(&s.header);
        }
        stats.hash_time += start.elapsed();
        for t in &s.tensors {
            bytes.resize(t.len(), 0);
            input.read_exact_at(&mut bytes, s.offset(t))?;
            let start = Instant::now();
            hasher.update(&bytes);
            if sources.len() > 1 {
                total.update(&bytes);
            }
            stats.hash_time += start.elapsed();
            let start = Instant::now();
            let packed = compress_bytes(t.dtype, &bytes, opts)?;
            stats.codec_time += start.elapsed();
            if verify {
                let back = check.resize(bytes.len())?;
                decompress_tensor(&packed, back)?;
                if back != &bytes[..] {
                    return Err(Error::Layout(format!("verification failed for {}", t.name)));
                }
            }
            out.write_all(&packed)?;
            entries[entry_at].1.data_offsets = [at, at + packed.len() as u64];
            entry_at += 1;
            at += packed.len() as u64;
            stats.tensors += 1;
            stats.elements += (t.len() / t.dtype.width()) as u64;
            stats.raw_bytes += t.len() as u64;
            stats.packed_bytes += packed.len() as u64;
        }
        meta.sources[i].checksum = hash_string(&hasher.finalize());
        meta.sources[i].bytes = (s.header.len() + s.data_len()) as u64;
        if sources.len() == 1 {
            total = hasher;
        }
    }
    meta.checksum = hash_string(&total.finalize());
    let mut json = header_json(&meta, &entries)?;
    if json.len() > reserved {
        return Err(Error::Layout(
            "container header outgrew its reservation".into(),
        ));
    }
    json.resize(reserved, b' ');
    out.write_all_at(&(reserved as u64).to_le_bytes(), 0)?;
    out.write_all_at(&json, 8)?;
    out.flush()?;
    stats.packed_bytes += 8 + reserved as u64;
    stats.checksum = meta.checksum;
    Ok(stats)
}

/// Read and validate a container header.
pub fn read_header(f: &mut File) -> Result<Header, Error> {
    let mut prefix = [0u8; 8];
    f.read_exact(&mut prefix)?;
    let len = u64::from_le_bytes(prefix);
    if len > 1 << 30 {
        return Err(Error::Header(format!(
            "container header length {len} too large"
        )));
    }
    let mut json = vec![0u8; len as usize];
    f.read_exact(&mut json)?;
    let mut map: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&json).map_err(|e| Error::Header(e.to_string()))?;
    let meta = map
        .remove(META_KEY)
        .ok_or_else(|| Error::Header(format!("missing {META_KEY}")))?;
    let meta: Meta = serde_json::from_value(meta).map_err(|e| Error::Header(e.to_string()))?;
    if meta.version != VERSION {
        return Err(Error::Header(format!(
            "unsupported container version {}",
            meta.version
        )));
    }
    let mut tensors = HashMap::with_capacity(map.len());
    for (name, v) in map {
        let e: Entry =
            serde_json::from_value(v).map_err(|e| Error::Header(format!("{name}: {e}")))?;
        if e.source >= meta.sources.len() || e.data_offsets[0] > e.data_offsets[1] {
            return Err(Error::Header(format!("{name}: bad entry")));
        }
        tensors.insert(name, e);
    }
    Ok(Header {
        meta,
        tensors,
        data_start: 8 + len,
    })
}

/// Where each source of a container is restored: an explicit file for a single-source
/// container, otherwise `dir/<source name>`.
pub fn output_paths(header: &Header, output: &Path) -> Vec<PathBuf> {
    match header.meta.sources.len() {
        1 if !output.is_dir() => vec![output.to_path_buf()],
        _ => header
            .meta
            .sources
            .iter()
            .map(|s| output.join(&s.name))
            .collect(),
    }
}

fn read_at(f: &File, offset: u64, len: u64) -> Result<Vec<u8>, Error> {
    let mut buf = vec![0u8; usize::try_from(len).map_err(|_| Error::Corrupt)?];
    f.read_exact_at(&mut buf, offset)?;
    Ok(buf)
}

/// Restore every source of the container `input` under `output` (see [`output_paths`]).
/// Existing files are only replaced when `force` is set. When `verify` is set each restored
/// file is hashed while it is written and must match its recorded BLAKE3.
pub fn decompress(input: &Path, output: &Path, verify: bool, force: bool) -> Result<Stats, Error> {
    let mut f = File::open(input)?;
    let header = read_header(&mut f)?;
    let paths = output_paths(&header, output);
    for p in &paths {
        if p.exists() && !force {
            return Err(Error::Layout(format!(
                "{} exists, pass --force to replace it",
                p.display()
            )));
        }
    }
    let mut stats = Stats {
        checksum: header.meta.checksum.clone(),
        ..Stats::default()
    };
    let mut buf = HugeBuf::new();
    for (i, (src, path)) in header.meta.sources.iter().zip(&paths).enumerate() {
        let [a, b] = src.header;
        let raw_header = read_at(&f, header.data_start + a, b.saturating_sub(a))?;
        let source = Source::parse(&raw_header)?;
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        let mut out = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        out.write_all(&raw_header)?;
        let mut hasher = Hasher::new();
        if verify {
            hasher.update(&raw_header);
        }
        for t in &source.tensors {
            let e = header
                .tensors
                .get(&t.name)
                .filter(|e| {
                    e.source == i
                        && e.shape == t.shape
                        && DType::from_name(&e.dtype) == Some(t.dtype)
                })
                .ok_or_else(|| Error::Header(format!("no matching entry for tensor {}", t.name)))?;
            let [a, b] = e.data_offsets;
            let packed = read_at(&f, header.data_start + a, b - a)?;
            let dst = buf.resize(t.len())?;
            let start = Instant::now();
            decompress_tensor(&packed, dst)?;
            stats.codec_time += start.elapsed();
            if verify {
                let start = Instant::now();
                hasher.update(buf.as_slice());
                stats.hash_time += start.elapsed();
            }
            out.write_all(buf.as_slice())?;
            stats.backing = buf.backing();
            stats.tensors += 1;
            stats.elements += (t.len() / t.dtype.width()) as u64;
            stats.raw_bytes += t.len() as u64;
            stats.packed_bytes += packed.len() as u64;
        }
        out.flush()?;
        if verify {
            let got = hash_string(&hasher.finalize());
            if got != src.checksum {
                return Err(Error::Digest {
                    expected: src.checksum.clone(),
                    got,
                });
            }
        }
    }
    Ok(stats)
}
