use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use cafetensor_lib::container::{self, compress, decompress, output_paths, read_header};
use cafetensor_lib::safetensors::{Source, TensorInfo, write_header};
use cafetensor_lib::tensor::{DType, Options};
use cafetensor_lib::{Error, SmMode, hash, shards};
use cafetensor_testkit::{Sampler, load_profiles, test_seed};

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> TempDir {
        let p = std::env::temp_dir().join(format!("cafetensor-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Write a `.safetensors` file of `(name, dtype, bytes)` tensors and return its bytes.
fn write_file(path: &Path, tensors: &[(String, DType, Vec<u8>)]) -> Vec<u8> {
    let infos: Vec<TensorInfo> = tensors
        .iter()
        .map(|(name, dtype, b)| TensorInfo {
            name: name.clone(),
            dtype: *dtype,
            shape: vec![b.len() / dtype.width()],
            start: 0,
            end: b.len(),
        })
        .collect();
    let mut meta = serde_json::Map::new();
    meta.insert("format".into(), "pt".into());
    let (mut file, _) = write_header(&infos, Some(&meta)).unwrap();
    for (_, _, b) in tensors {
        file.extend_from_slice(b);
    }
    File::create(path).unwrap().write_all(&file).unwrap();
    file
}

fn sample(dtype: &str, bytes: usize, salt: u64) -> Vec<u8> {
    let p = load_profiles();
    let d = p
        .iter()
        .flat_map(|p| &p.dtypes)
        .find(|d| d.dtype == dtype)
        .unwrap();
    Sampler::new(d, test_seed(d.seed) ^ salt).sample(bytes)
}

fn tensors(prefix: &str, salt: u64) -> Vec<(String, DType, Vec<u8>)> {
    let name = |s: &str| format!("{prefix}.{s}");
    vec![
        (name("embed"), DType::Bf16, sample("BF16", 3 << 20, salt)),
        (name("norm"), DType::F32, sample("F32", 1 << 12, salt)),
        (
            name("proj"),
            DType::F8E4M3,
            sample("F8_E4M3", 1 << 20, salt),
        ),
        (name("up"), DType::F8E5M2, sample("F8_E5M2", 300_001, salt)),
        (name("conv"), DType::F16, sample("F16", 1 << 19, salt)),
        (
            name("ids"),
            DType::I64,
            (0..8000u64).flat_map(u64::to_le_bytes).collect(),
        ),
        (name("empty"), DType::F16, Vec::new()),
        (name("tiny"), DType::Bf16, vec![0x80, 0x3F]),
    ]
}

#[test]
fn single_source_round_trip() {
    let dir = TempDir::new("single");
    let src = dir.0.join("model.safetensors");
    let original = write_file(&src, &tensors("a", 0));
    for mode in [SmMode::Raw, SmMode::Coded] {
        for crc in [false, true] {
            let out = dir.0.join("model.cafetensor");
            let opts = Options {
                crc,
                ..Options::new(1 << 21, mode)
            };
            let stats = compress(std::slice::from_ref(&src), &out, &opts, true).unwrap();
            assert_eq!(
                stats.checksum,
                hash::hash_string(blake3::hash(&original).as_bytes())
            );
            assert_eq!(stats.tensors, 8);
            let header = read_header(&mut File::open(&out).unwrap()).unwrap();
            assert_eq!(header.meta.block, 21);
            assert_eq!(header.meta.crc32c, crc);
            assert_eq!(header.meta.sources[0].bytes, original.len() as u64);
            let restored = dir.0.join("restored.safetensors");
            assert!(decompress(&out, &restored, false, false).is_ok());
            assert!(decompress(&out, &restored, true, false).is_err());
            decompress(&out, &restored, true, true).unwrap();
            assert!(
                fs::read(&restored).unwrap() == original,
                "{mode:?} crc={crc}"
            );
            fs::remove_file(&restored).unwrap();
        }
    }
}

#[test]
fn multi_source_round_trip_and_checksums() {
    let dir = TempDir::new("multi");
    let a = dir.0.join("model-00001-of-00002.safetensors");
    let b = dir.0.join("model-00002-of-00002.safetensors");
    let fa = write_file(&a, &tensors("a", 1));
    let fb = write_file(&b, &tensors("b", 2));
    let out = dir.0.join("both.cafetensor");
    let stats = compress(&[a.clone(), b.clone()], &out, &Options::default(), false).unwrap();
    let both: Vec<u8> = fa.iter().chain(&fb).copied().collect();
    assert_eq!(
        stats.checksum,
        hash::hash_string(blake3::hash(&both).as_bytes())
    );
    let header = read_header(&mut File::open(&out).unwrap()).unwrap();
    assert_eq!(header.meta.sources.len(), 2);
    assert_eq!(header.meta.sources[1].checksum, hash::b3sum(&b).unwrap());
    let restore = dir.0.join("restore");
    fs::create_dir_all(&restore).unwrap();
    let paths = output_paths(&header, &restore);
    decompress(&out, &restore, true, false).unwrap();
    assert!(fs::read(&paths[0]).unwrap() == fa);
    assert!(fs::read(&paths[1]).unwrap() == fb);
    let dup = compress(&[a.clone(), a.clone()], &out, &Options::default(), false);
    assert!(matches!(dup, Err(Error::Layout(_))));
}

#[test]
fn corrupt_container_is_rejected() {
    let dir = TempDir::new("corrupt");
    let src = dir.0.join("model.safetensors");
    write_file(&src, &tensors("c", 3));
    let out = dir.0.join("model.cafetensor");
    compress(std::slice::from_ref(&src), &out, &Options::default(), false).unwrap();
    let good = fs::read(&out).unwrap();
    let header = read_header(&mut File::open(&out).unwrap()).unwrap();
    let data = header.data_start as usize;
    let restored = dir.0.join("restored.safetensors");
    let mut s = 77u64;
    for _ in 0..40 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let mut bad = good.clone();
        let i = data + (s % (bad.len() - data) as u64) as usize;
        bad[i] ^= 1 << ((s >> 32) % 8);
        let path = dir.0.join("bad.cafetensor");
        fs::write(&path, &bad).unwrap();
        assert!(
            container::decompress(&path, &restored, true, true).is_err(),
            "flip at {i} went unnoticed"
        );
    }
}

#[test]
fn split_then_join_keeps_every_tensor() {
    let dir = TempDir::new("shards");
    let src = dir.0.join("model.safetensors");
    let original = tensors("s", 4);
    write_file(&src, &original);
    let shard_dir = dir.0.join("shards");
    let paths = shards::split(&src, &shard_dir, 1 << 20, "model").unwrap();
    assert!(paths.len() > 2);
    assert!(shard_dir.join("model.safetensors.index.json").exists());
    let joined = dir.0.join("joined.safetensors");
    assert_eq!(shards::join(&paths, &joined).unwrap(), original.len());
    let s = Source::open(&joined).unwrap();
    let file = fs::read(&joined).unwrap();
    for (name, dtype, bytes) in &original {
        let t = s.tensors.iter().find(|t| &t.name == name).unwrap();
        assert_eq!(t.dtype, *dtype);
        let at = s.offset(t) as usize;
        assert!(file[at..at + t.len()] == bytes[..], "{name}");
    }
    assert_eq!(
        s.metadata.unwrap().get("format").and_then(|v| v.as_str()),
        Some("pt")
    );
}
