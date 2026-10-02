//! `amd64_v3` tier: built with `-Ctarget-cpu=x86-64-v3`.
//!
//! # Safety
//!
//! Every `unsafe` block calls a `core::arch` intrinsic whose CPU feature the flag guard below
//! proves is enabled for this whole crate, and loads or stores only within the arrays passed
//! to the hook.
#![no_std]
#![feature(portable_simd)]
#![cfg(target_arch = "x86_64")]

// Flag guard, the one permitted use of cfg(target_feature) in a tier crate. It fails the build
// when this crate is compiled without its tier flags. The list is scripts/tier-features/amd64_v3.txt.
#[cfg(not(all(
    target_feature = "avx",
    target_feature = "avx2",
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
compile_error!("amd64_v3 was built without its tier flags; see cafetensor-lib/README.md");

use core::arch::x86_64::*;
use core::simd::u8x16;

use generic_operations::{Engine, Kernels, Operations};

/// Private on purpose. A crate that can name this type can instantiate generics against it at
/// its own (baseline) flags.
struct V3;

impl Kernels for V3 {
    const BLAKE3_LANES: usize = 16;

    #[inline(always)]
    fn xor64(a: &[u8; 64], b: &[u8; 64], out: &mut [u8; 64]) {
        for i in 0..2 {
            let at = 32 * i;
            unsafe {
                let r = _mm256_xor_si256(
                    _mm256_loadu_si256(a[at..].as_ptr().cast()),
                    _mm256_loadu_si256(b[at..].as_ptr().cast()),
                );
                _mm256_storeu_si256(out[at..].as_mut_ptr().cast(), r);
            }
        }
    }

    #[inline(always)]
    fn shuffle_bytes(v: u8x16, idx: u8x16) -> u8x16 {
        unsafe { _mm_shuffle_epi8(v.into(), idx.into()).into() }
    }

    #[inline(always)]
    fn crc32c_u64(state: u32, word: u64) -> u32 {
        unsafe { _mm_crc32_u64(state as u64, word) as u32 }
    }
}

/// The vtable is built here, so every `Engine<V3>` method is codegen'd here.
static ENGINE: Engine<V3> = Engine::new("amd64_v3");

/// The crate's only public item. Non-generic and out of line, so it is codegen'd in this crate
/// and inlined nowhere else.
#[inline(never)]
pub fn operations() -> &'static dyn Operations {
    &ENGINE
}
