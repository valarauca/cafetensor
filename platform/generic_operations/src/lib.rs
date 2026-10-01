//! Every algorithm, written once over a [`Kernels`] type and monomorphized inside each tier
//! crate with that crate's target flags.
#![no_std]
#![feature(portable_simd)]

use core::marker::PhantomData;
use core::simd::prelude::*;

mod algorithms;

/// Fine-grained static hooks. Not object-safe by design.
/// Every method has a portable `core::simd` default. A tier overrides a method only where an
/// intrinsic beats the default's codegen on that tier.
pub trait Kernels: 'static {
    /// `out = a ^ b` over one 64-byte block.
    #[inline(always)]
    fn xor64(a: &[u8; 64], b: &[u8; 64], out: &mut [u8; 64]) {
        *out = (u8x64::from_array(*a) ^ u8x64::from_array(*b)).to_array();
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
}
