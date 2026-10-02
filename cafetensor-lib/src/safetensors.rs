//! Reading and writing plain `.safetensors` files.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anamnesis::{Dtype, parse_safetensors_header_from_reader};

use crate::Error;
use crate::tensor::DType;

/// One tensor of a `.safetensors` file. Offsets are relative to the start of the data section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub start: usize,
    pub end: usize,
}

impl TensorInfo {
    /// Byte length of the tensor data.
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Whether the tensor holds no bytes.
    pub fn is_empty(&self) -> bool {
        self.end == self.start
    }
}

/// A parsed `.safetensors` header together with its verbatim bytes.
#[derive(Debug, Clone)]
pub struct Source {
    pub path: PathBuf,
    /// Length prefix and JSON header exactly as stored, so the file can be rebuilt bit for bit.
    pub header: Vec<u8>,
    /// Tensors in data-offset order, covering the data section without gaps.
    pub tensors: Vec<TensorInfo>,
    /// The `__metadata__` object, if any.
    pub metadata: Option<serde_json::Map<String, serde_json::Value>>,
}

fn dtype_of(name: &str, d: Dtype) -> Result<DType, Error> {
    let t = match d {
        Dtype::BF16 => DType::Bf16,
        Dtype::F32 => DType::F32,
        Dtype::F16 => DType::F16,
        Dtype::F8E4M3 => DType::F8E4M3,
        Dtype::F8E5M2 => DType::F8E5M2,
        Dtype::F64 => DType::F64,
        Dtype::Bool => DType::Bool,
        Dtype::U8 => DType::U8,
        Dtype::I8 => DType::I8,
        Dtype::U16 => DType::U16,
        Dtype::I16 => DType::I16,
        Dtype::U32 => DType::U32,
        Dtype::I32 => DType::I32,
        Dtype::U64 => DType::U64,
        Dtype::I64 => DType::I64,
        other => {
            return Err(Error::UnsupportedDtype {
                name: name.to_string(),
                dtype: format!("{other:?}"),
            });
        }
    };
    Ok(t)
}

impl Source {
    /// Parse the header of a `.safetensors` file without reading tensor data.
    pub fn open(path: &Path) -> Result<Source, Error> {
        let mut f = File::open(path)?;
        let mut prefix = [0u8; 8];
        f.read_exact(&mut prefix)?;
        let len = u64::from_le_bytes(prefix);
        if len > 100 << 20 {
            return Err(Error::Header(format!(
                "{}: header length {len} too large",
                path.display()
            )));
        }
        let mut header = prefix.to_vec();
        header.resize(8 + len as usize, 0);
        f.read_exact(&mut header[8..])?;
        let mut src = Source::parse(&header)?;
        src.path = path.to_path_buf();
        Ok(src)
    }

    /// Parse verbatim header bytes (length prefix included).
    pub fn parse(header: &[u8]) -> Result<Source, Error> {
        let parsed = parse_safetensors_header_from_reader(header)?;
        let json: serde_json::Value = serde_json::from_slice(header.get(8..).unwrap_or_default())
            .map_err(|e| Error::Header(e.to_string()))?;
        let metadata = json
            .get("__metadata__")
            .and_then(|m| m.as_object())
            .cloned();
        let mut tensors = Vec::with_capacity(parsed.tensors.len());
        for t in parsed.tensors {
            tensors.push(TensorInfo {
                dtype: dtype_of(&t.name, t.dtype)?,
                shape: t.shape,
                start: t.data_offsets.0,
                end: t.data_offsets.1,
                name: t.name,
            });
        }
        tensors.sort_by_key(|t| (t.start, t.end));
        let mut end = 0;
        for t in &tensors {
            let elems = t.shape.iter().try_fold(1usize, |a, &d| a.checked_mul(d));
            let want = elems.and_then(|e| e.checked_mul(t.dtype.width()));
            if t.start != end || want != Some(t.len()) {
                return Err(Error::Layout(format!(
                    "tensor {} at {}..{}",
                    t.name, t.start, t.end
                )));
            }
            end = t.end;
        }
        Ok(Source {
            path: PathBuf::new(),
            header: header.to_vec(),
            tensors,
            metadata,
        })
    }

    /// Size of the data section.
    pub fn data_len(&self) -> usize {
        self.tensors.last().map_or(0, |t| t.end)
    }

    /// Absolute file offset of a tensor's data.
    pub fn offset(&self, t: &TensorInfo) -> u64 {
        (self.header.len() + t.start) as u64
    }
}

/// Serialized header bytes (length prefix included) and the data offsets of every tensor.
pub type WrittenHeader = (Vec<u8>, Vec<(usize, usize)>);

/// Serialize a safetensors header for `tensors` laid out contiguously in the given order, with
/// the JSON padded by spaces to a multiple of 8 bytes. Returns the bytes and the new offsets.
pub fn write_header(
    tensors: &[TensorInfo],
    metadata: Option<&serde_json::Map<String, serde_json::Value>>,
) -> Result<WrittenHeader, Error> {
    let mut map = serde_json::Map::new();
    if let Some(m) = metadata {
        map.insert("__metadata__".into(), serde_json::Value::Object(m.clone()));
    }
    let mut at = 0;
    let mut offsets = Vec::with_capacity(tensors.len());
    for t in tensors {
        offsets.push((at, at + t.len()));
        map.insert(
            t.name.clone(),
            serde_json::json!({ "dtype": t.dtype.name(), "shape": t.shape, "data_offsets": [at, at + t.len()] }),
        );
        at += t.len();
    }
    let mut json = serde_json::to_vec(&map).map_err(|e| Error::Header(e.to_string()))?;
    json.resize(json.len().next_multiple_of(8), b' ');
    let mut out = (json.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&json);
    Ok((out, offsets))
}

/// Copy `len` bytes at `offset` of `from` to the current end of `to`.
pub fn copy_range(from: &File, offset: u64, len: u64, to: &mut File) -> Result<(), Error> {
    use std::io::{Seek, SeekFrom};
    let mut from = from.try_clone()?;
    from.seek(SeekFrom::Start(offset))?;
    let copied = std::io::copy(&mut from.take(len), to)?;
    if copied != len {
        return Err(Error::Truncated);
    }
    to.flush()?;
    Ok(())
}
