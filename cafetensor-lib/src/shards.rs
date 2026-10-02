//! Joining sharded `.safetensors` files into one file and splitting one back into shards.

use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::Error;
use crate::safetensors::{Source, TensorInfo, copy_range, write_header};

/// Merge `inputs` into one `.safetensors` file at `output`, keeping tensor data in input order
/// and merging every `__metadata__` object. Returns the number of tensors written.
pub fn join(inputs: &[PathBuf], output: &Path) -> Result<usize, Error> {
    let sources = inputs
        .iter()
        .map(|p| Source::open(p))
        .collect::<Result<Vec<_>, _>>()?;
    let mut seen = HashSet::new();
    let mut tensors: Vec<TensorInfo> = Vec::new();
    let mut metadata = serde_json::Map::new();
    for s in &sources {
        for t in &s.tensors {
            if !seen.insert(t.name.clone()) {
                return Err(Error::Layout(format!(
                    "tensor {} appears in more than one input",
                    t.name
                )));
            }
            tensors.push(t.clone());
        }
        if let Some(m) = &s.metadata {
            metadata.extend(m.clone());
        }
    }
    let meta = (!metadata.is_empty()).then_some(&metadata);
    let (header, _) = write_header(&tensors, meta)?;
    let mut out = File::create(output)?;
    out.write_all(&header)?;
    for s in &sources {
        let input = File::open(&s.path)?;
        copy_range(&input, s.header.len() as u64, s.data_len() as u64, &mut out)?;
    }
    Ok(tensors.len())
}

/// Split `input` into shards of at most `max_bytes` of tensor data (a larger tensor gets a
/// shard of its own), written as `dir/<prefix>-NNNNN-of-NNNNN.safetensors` together with a
/// Hugging Face style `dir/<prefix>.safetensors.index.json`. Returns the shard paths.
pub fn split(
    input: &Path,
    dir: &Path,
    max_bytes: u64,
    prefix: &str,
) -> Result<Vec<PathBuf>, Error> {
    let src = Source::open(input)?;
    let mut groups: Vec<Vec<&TensorInfo>> = vec![Vec::new()];
    let mut size = 0u64;
    for t in &src.tensors {
        if size > 0 && size + t.len() as u64 > max_bytes {
            groups.push(Vec::new());
            size = 0;
        }
        if let Some(g) = groups.last_mut() {
            g.push(t);
        }
        size += t.len() as u64;
    }
    fs::create_dir_all(dir)?;
    let input_file = File::open(input)?;
    let n = groups.len();
    let mut paths = Vec::with_capacity(n);
    let mut weight_map = BTreeMap::new();
    for (i, g) in groups.iter().enumerate() {
        let name = format!("{prefix}-{:05}-of-{n:05}.safetensors", i + 1);
        let tensors: Vec<TensorInfo> = g.iter().map(|t| (*t).clone()).collect();
        let (header, _) = write_header(&tensors, src.metadata.as_ref())?;
        let path = dir.join(&name);
        let mut out = File::create(&path)?;
        out.write_all(&header)?;
        for t in g {
            copy_range(&input_file, src.offset(t), t.len() as u64, &mut out)?;
            weight_map.insert(t.name.clone(), name.clone());
        }
        paths.push(path);
    }
    let index = serde_json::json!({
        "metadata": { "total_size": src.data_len() },
        "weight_map": weight_map,
    });
    let index_path = dir.join(format!("{prefix}.safetensors.index.json"));
    let json = serde_json::to_vec_pretty(&index).map_err(|e| Error::Header(e.to_string()))?;
    fs::write(index_path, json)?;
    Ok(paths)
}
