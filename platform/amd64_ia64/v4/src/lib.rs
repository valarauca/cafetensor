//! `amd64_v4` tier: built with `-Ctarget-cpu=x86-64-v4`.
#![no_std]
#![cfg(target_arch = "x86_64")]

// Flag guard, the one permitted use of cfg(target_feature) in a tier crate. It fails the build
// when this crate is compiled without its tier flags. The list is scripts/tier-features/amd64_v4.txt.
#[cfg(not(all(
    target_feature = "avx",
    target_feature = "avx2",
    target_feature = "avx512bw",
    target_feature = "avx512cd",
    target_feature = "avx512dq",
    target_feature = "avx512f",
    target_feature = "avx512vl",
    target_feature = "bmi1",
    target_feature = "bmi2",
    target_feature = "cmpxchg16b",
    target_feature = "f16c",
    target_feature = "fma",
    target_feature = "lahfsahf",
    target_feature = "lzcnt",
    target_feature = "movbe",
    target_feature = "popcnt",
    target_feature = "sse3",
    target_feature = "sse4.1",
    target_feature = "sse4.2",
    target_feature = "ssse3",
    target_feature = "xsave"
)))]
compile_error!("amd64_v4 was built without its tier flags; see cafetensor-lib/README.md");

use generic_operations::{Engine, Kernels, Operations};

/// Private on purpose. A crate that can name this type can instantiate generics against it at
/// its own (baseline) flags.
struct V4;

impl Kernels for V4 {}

/// The vtable is built here, so every `Engine<V4>` method is codegen'd here.
static ENGINE: Engine<V4> = Engine::new("amd64_v4");

/// The crate's only public item. Non-generic and out of line, so it is codegen'd in this crate
/// and inlined nowhere else.
#[inline(never)]
pub fn operations() -> &'static dyn Operations {
    &ENGINE
}
