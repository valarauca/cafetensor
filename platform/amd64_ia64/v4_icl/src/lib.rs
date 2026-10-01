//! `amd64_v4_icl` tier: built with `-Ctarget-cpu=x86-64-v4 -Ctarget-feature=+avx512vbmi,+avx512vbmi2,+pclmulqdq,+vpclmulqdq`.
#![no_std]
#![cfg(target_arch = "x86_64")]

// Flag guard, the one permitted use of cfg(target_feature) in a tier crate. It fails the build
// when this crate is compiled without its tier flags. The list is scripts/tier-features/amd64_v4_icl.txt.
#[cfg(not(all(
    target_feature = "avx",
    target_feature = "avx2",
    target_feature = "avx512bw",
    target_feature = "avx512cd",
    target_feature = "avx512dq",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "avx512vbmi2",
    target_feature = "avx512vl",
    target_feature = "bmi1",
    target_feature = "bmi2",
    target_feature = "cmpxchg16b",
    target_feature = "f16c",
    target_feature = "fma",
    target_feature = "lahfsahf",
    target_feature = "lzcnt",
    target_feature = "movbe",
    target_feature = "pclmulqdq",
    target_feature = "popcnt",
    target_feature = "sse3",
    target_feature = "sse4.1",
    target_feature = "sse4.2",
    target_feature = "ssse3",
    target_feature = "vpclmulqdq",
    target_feature = "xsave"
)))]
compile_error!("amd64_v4_icl was built without its tier flags; see cafetensor-lib/README.md");

use generic_operations::{Engine, Kernels, Operations};

/// Private on purpose. A crate that can name this type can instantiate generics against it at
/// its own (baseline) flags.
struct V4Icl;

impl Kernels for V4Icl {
    #[inline(always)]
    fn xor64(a: &[u8; 64], b: &[u8; 64], out: &mut [u8; 64]) {
        use core::arch::x86_64::*;
        unsafe {
            let r = _mm512_xor_si512(
                _mm512_loadu_si512(a.as_ptr().cast()),
                _mm512_loadu_si512(b.as_ptr().cast()),
            );
            _mm512_storeu_si512(out.as_mut_ptr().cast(), r);
        }
    }
}

/// The vtable is built here, so every `Engine<V4Icl>` method is codegen'd here.
static ENGINE: Engine<V4Icl> = Engine::new("amd64_v4_icl");

/// The crate's only public item. Non-generic and out of line, so it is codegen'd in this crate
/// and inlined nowhere else.
#[inline(never)]
pub fn operations() -> &'static dyn Operations {
    &ENGINE
}
