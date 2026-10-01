//! Every algorithm, written once over a [`Kernels`] type and monomorphized inside each tier
//! crate with that crate's target flags.
#![no_std]
#![feature(portable_simd)]

use core::marker::PhantomData;

mod algorithms;

/// Fine-grained static hooks. Not object-safe by design.
/// Every method has a portable `core::simd` default. A tier overrides a method only where an
/// intrinsic beats the default's codegen on that tier.
pub trait Kernels: 'static {}

/// Coarse, object-safe contract: the only thing that crosses a tier boundary.
/// Signatures use slices, scalars and plain enums, never SIMD types.
pub trait Operations: Sync + 'static {
    /// Name of the tier that implements this handle.
    fn tier_name(&self) -> &'static str;
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
}
