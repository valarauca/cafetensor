//! Lossless compression of safetensors checkpoints, on the CPU tier `general_backend` selects
//! once per process.
//!
//! Every floating point value (BF16, F32, F16, F8_E4M3, F8_E5M2) is split into an exponent
//! symbol and a raw residual. The exponent is coded with static 16-lane interleaved rANS, one
//! model per tensor, shared by independently decodable chunks. Other dtypes are passed through.
//! The formats are byte-identical to the `tensor-compressor` project's.
//!
//! Layering:
//! * [`tensor`] is the per-tensor blob format,
//! * [`container`] gathers one or more `.safetensors` files into a `.cafetensor` and back,
//! * [`safetensors`] and [`shards`] read, write, join and split plain `.safetensors` files,
//! * [`hash`] is BLAKE3 over the selected tier, in the container's `blake3-<hex>` notation.

mod bytes;
pub mod container;
pub mod hash;
pub mod safetensors;
pub mod shards;
pub mod tensor;

use std::fmt;
use std::sync::LazyLock;

use general_backend::Operations;

pub use general_backend::OpError;
pub use general_backend::codebook::SmMode;
pub use os_common::{Backing, HugeBuf};

static TIER: LazyLock<&'static dyn Operations> = LazyLock::new(general_backend::operations);

pub(crate) fn ops() -> &'static dyn Operations {
    *TIER
}

/// Name of the CPU tier this process selected.
pub fn tier_name() -> &'static str {
    ops().tier_name()
}

/// Round-trip a small generated BF16 tensor through the selected tier, raw and coded.
pub fn self_test() -> Result<(), Error> {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let bytes: Vec<u8> = (0..100_003)
        .flat_map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let exp = 0x76 + (s % 7).min((s >> 32) & 7);
            (((s >> 40) as u16 & 0x807F) | ((exp as u16) << 7)).to_le_bytes()
        })
        .collect();
    for mode in [SmMode::Raw, SmMode::Coded] {
        let opts = tensor::Options::new(1 << 16, mode);
        let packed = tensor::compress_bytes(tensor::DType::Bf16, &bytes, &opts)?;
        let mut back = vec![0u8; bytes.len()];
        tensor::decompress_tensor(&packed, &mut back)?;
        if back != bytes || tensor::is_raw(&packed)? {
            return Err(Error::Corrupt);
        }
    }
    Ok(())
}

/// Errors produced while compressing or decompressing.
#[derive(Debug)]
pub enum Error {
    BadMagic,
    Truncated,
    Corrupt,
    Checksum,
    Digest { expected: String, got: String },
    LengthMismatch { expected: u64, got: u64 },
    UnsupportedDtype { name: String, dtype: String },
    Layout(String),
    Header(String),
    Io(std::io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::BadMagic => write!(f, "bad magic"),
            Error::Truncated => write!(f, "truncated input"),
            Error::Corrupt => write!(f, "corrupt stream"),
            Error::Checksum => write!(f, "CRC32C mismatch"),
            Error::Digest { expected, got } => {
                write!(f, "BLAKE3 mismatch: expected {expected}, got {got}")
            }
            Error::LengthMismatch { expected, got } => {
                write!(
                    f,
                    "length mismatch: header says {expected}, output has {got}"
                )
            }
            Error::UnsupportedDtype { name, dtype } => {
                write!(f, "tensor {name} has unsupported dtype {dtype}")
            }
            Error::Layout(s) => write!(f, "unsupported layout: {s}"),
            Error::Header(s) => write!(f, "safetensors header: {s}"),
            Error::Io(e) => write!(f, "io: {e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<anamnesis::AnamnesisError> for Error {
    fn from(e: anamnesis::AnamnesisError) -> Self {
        Error::Header(e.to_string())
    }
}

impl From<OpError> for Error {
    fn from(e: OpError) -> Self {
        match e {
            OpError::OutputTooSmall => Error::Truncated,
            _ => Error::Corrupt,
        }
    }
}
