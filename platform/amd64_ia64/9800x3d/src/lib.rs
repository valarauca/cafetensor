//! `amd64_9800x3d` tier: built with `-Ctarget-cpu=znver5 -Ctarget-feature=-rdseed`.
#![no_std]
#![cfg(target_arch = "x86_64")]

// Flag guard, the one permitted use of cfg(target_feature) in a tier crate. It fails the build
// when this crate is compiled without its tier flags. The list is scripts/tier-features/amd64_9800x3d.txt.
#[cfg(not(all(
    target_feature = "adx",
    target_feature = "aes",
    target_feature = "avx",
    target_feature = "avx2",
    target_feature = "avx512bf16",
    target_feature = "avx512bitalg",
    target_feature = "avx512bw",
    target_feature = "avx512cd",
    target_feature = "avx512dq",
    target_feature = "avx512f",
    target_feature = "avx512ifma",
    target_feature = "avx512vbmi",
    target_feature = "avx512vbmi2",
    target_feature = "avx512vl",
    target_feature = "avx512vnni",
    target_feature = "avx512vp2intersect",
    target_feature = "avx512vpopcntdq",
    target_feature = "avxvnni",
    target_feature = "bmi1",
    target_feature = "bmi2",
    target_feature = "clflushopt",
    target_feature = "cmpxchg16b",
    target_feature = "f16c",
    target_feature = "fma",
    target_feature = "gfni",
    target_feature = "lahfsahf",
    target_feature = "lzcnt",
    target_feature = "movbe",
    target_feature = "pclmulqdq",
    target_feature = "popcnt",
    target_feature = "prfchw",
    target_feature = "rdrand",
    target_feature = "sha",
    target_feature = "sse3",
    target_feature = "sse4.1",
    target_feature = "sse4.2",
    target_feature = "sse4a",
    target_feature = "ssse3",
    target_feature = "vaes",
    target_feature = "vpclmulqdq",
    target_feature = "xsave",
    target_feature = "xsavec",
    target_feature = "xsaveopt",
    target_feature = "xsaves"
)))]
compile_error!("amd64_9800x3d was built without its tier flags; see cafetensor-lib/README.md");

use generic_operations::{Engine, Kernels, Operations};

/// Private on purpose. A crate that can name this type can instantiate generics against it at
/// its own (baseline) flags.
struct Zen5;

impl Kernels for Zen5 {}

/// The vtable is built here, so every `Engine<Zen5>` method is codegen'd here.
static ENGINE: Engine<Zen5> = Engine::new("amd64_9800x3d");

/// The crate's only public item. Non-generic and out of line, so it is codegen'd in this crate
/// and inlined nowhere else.
#[inline(never)]
pub fn operations() -> &'static dyn Operations {
    &ENGINE
}
