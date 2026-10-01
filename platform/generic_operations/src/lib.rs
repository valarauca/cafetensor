//! Every algorithm, written once over a [`Kernels`] type and monomorphized inside each tier
//! crate with that crate's target flags.
#![cfg_attr(not(test), no_std)]
#![feature(portable_simd)]
#![feature(core_intrinsics)]
#![allow(
    internal_features,
    reason = "core::intrinsics::log matches the old std f64::ln bit for bit"
)]

use core::marker::PhantomData;
use core::simd::prelude::*;

mod algorithms;
pub mod codebook;
pub mod crc;
pub mod decode;
pub mod format;
mod planes;
pub mod rans;

pub use format::{Format, plane_len};

/// Fine-grained static hooks. Not object-safe by design.
/// Every method has a portable `core::simd` default. A tier overrides a method only where an
/// intrinsic beats the default's codegen on that tier.
pub trait Kernels: 'static {
    /// Encode renormalization without a data-dependent branch. Output is identical either way;
    /// which is faster depends on the tier (see docs/codegen-parity.md).
    const ENCODE_BRANCHLESS: bool = false;

    /// Decode as many leading elements of a chunk as this tier can with vector code, in whole
    /// groups, and return how many; the scalar path decodes the rest. The default decodes
    /// nothing.
    #[inline(always)]
    fn decode_groups(t: &decode::DecTables, st: &mut decode::ChunkState, out: &mut [u8]) -> usize {
        let _ = (t, st, out);
        0
    }

    /// `out = a ^ b` over one 64-byte block.
    #[inline(always)]
    fn xor64(a: &[u8; 64], b: &[u8; 64], out: &mut [u8; 64]) {
        *out = (u8x64::from_array(*a) ^ u8x64::from_array(*b)).to_array();
    }

    /// Advance a raw CRC32C state over one little-endian 8-byte word.
    #[inline(always)]
    fn crc32c_u64(state: u32, word: u64) -> u32 {
        word.to_le_bytes()
            .iter()
            .fold(state, |c, &b| crc::update_byte(c, b))
    }

    /// Advance a raw CRC32C state over whole 256-byte blocks.
    #[inline(always)]
    fn crc32c_blocks(state: u32, blocks: &[[u8; 256]]) -> u32 {
        blocks
            .iter()
            .flat_map(|b| b.as_chunks::<8>().0)
            .fold(state, |c, w| Self::crc32c_u64(c, u64::from_le_bytes(*w)))
    }
}

/// Coarse, object-safe contract: the only thing that crosses a tier boundary.
/// Signatures use slices, scalars and plain enums, never SIMD types.
pub trait Operations: Sync + 'static {
    /// Name of the tier that implements this handle.
    fn tier_name(&self) -> &'static str;

    /// Plumbing check: XOR the first half of `input` with its second half into `output`.
    /// Returns the number of bytes written.
    fn op_a(&self, input: &[u8], output: &mut [u8]) -> Result<usize, OpError>;

    /// Continue a raw CRC32C state over `data`, without pre or post inversion. The standard
    /// CRC32C of `data` is `!crc32c_update(!0, data)`.
    fn crc32c_update(&self, state: u32, data: &[u8]) -> u32;

    /// Split little-endian elements of `fmt` into the exponent plane, the high residual plane
    /// (one byte per element each) and `fmt.low_bytes()` low mantissa bytes per element.
    fn split_planes(
        &self,
        fmt: Format,
        bytes: &[u8],
        exps: &mut [u8],
        his: &mut [u8],
        los: &mut [u8],
    ) -> Result<(), OpError>;

    /// Pack the low `hi_bits` bits of every high residual into bitplanes, see [`plane_len`].
    fn pack_bitplanes(&self, his: &[u8], hi_bits: u32, out: &mut [u8]) -> Result<(), OpError>;

    /// Add the joint (exponent, high residual) counts of one chunk to `joint`, indexed
    /// `exp << 8 | hi`. The caller keeps a chunk below 2^32 elements and sums chunks in `u64`.
    fn histogram(&self, exps: &[u8], his: &[u8], joint: &mut [u32; 65536]) -> Result<(), OpError>;

    /// rANS-encode one chunk's exponent symbols (and coded sign|mantissa symbols), writing per
    /// way the exponent words, sign|mantissa words and escape bytes. `scratch` needs
    /// [`rans::encode_scratch_len`] words and `out` at most [`rans::encode_bound`] bytes.
    fn encode_chunk(
        &self,
        tables: &rans::EncTables,
        exps: &[u8],
        his: &[u8],
        scratch: &mut [u16],
        out: &mut [u8],
    ) -> Result<rans::ChunkInfo, OpError>;

    /// Decode one chunk into `out` (`chunk.elems * fmt.width` bytes). Every stream must be
    /// consumed exactly and every lane must return to its initial state, otherwise the chunk
    /// is rejected as corrupt.
    fn decode_chunk(
        &self,
        tables: &decode::DecTables,
        fmt: Format,
        chunk: decode::ChunkRef,
        out: &mut [u8],
    ) -> Result<(), OpError>;
}

/// Errors returned by [`Operations`] methods.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum OpError {
    OutputTooSmall,
    Corrupt,
}

/// Binds a kernel set to the contract. Instantiated inside tier crates, plus once in
/// general_backend for the portable fallback.
pub struct Engine<K: Kernels> {
    name: &'static str,
    _k: PhantomData<fn() -> K>,
}

impl<K: Kernels> Engine<K> {
    /// An engine named `name`.
    pub const fn new(name: &'static str) -> Self {
        Self {
            name,
            _k: PhantomData,
        }
    }
}

impl<K: Kernels> Operations for Engine<K> {
    fn tier_name(&self) -> &'static str {
        self.name
    }

    #[inline(never)]
    fn op_a(&self, input: &[u8], output: &mut [u8]) -> Result<usize, OpError> {
        algorithms::op_a::<K>(input, output)
    }

    #[inline(never)]
    fn crc32c_update(&self, state: u32, data: &[u8]) -> u32 {
        algorithms::crc32c_update::<K>(state, data)
    }

    #[inline(never)]
    fn split_planes(
        &self,
        fmt: Format,
        bytes: &[u8],
        exps: &mut [u8],
        his: &mut [u8],
        los: &mut [u8],
    ) -> Result<(), OpError> {
        planes::split_planes::<K>(fmt, bytes, exps, his, los)
    }

    #[inline(never)]
    fn encode_chunk(
        &self,
        tables: &rans::EncTables,
        exps: &[u8],
        his: &[u8],
        scratch: &mut [u16],
        out: &mut [u8],
    ) -> Result<rans::ChunkInfo, OpError> {
        rans::encode_chunk::<K>(tables, exps, his, scratch, out)
    }

    #[inline(never)]
    fn decode_chunk(
        &self,
        tables: &decode::DecTables,
        fmt: Format,
        chunk: decode::ChunkRef,
        out: &mut [u8],
    ) -> Result<(), OpError> {
        decode::decode_chunk::<K>(tables, fmt, chunk, out)
    }

    #[inline(never)]
    fn histogram(&self, exps: &[u8], his: &[u8], joint: &mut [u32; 65536]) -> Result<(), OpError> {
        algorithms::histogram::<K>(exps, his, joint)
    }

    #[inline(never)]
    fn pack_bitplanes(&self, his: &[u8], hi_bits: u32, out: &mut [u8]) -> Result<(), OpError> {
        planes::pack_bitplanes::<K>(his, hi_bits, out)
    }
}
