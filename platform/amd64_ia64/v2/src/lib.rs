//! `amd64_v2` tier: built with `-Ctarget-cpu=x86-64-v2`.
#![no_std]
#![cfg(target_arch = "x86_64")]

// Flag guard, the one permitted use of cfg(target_feature) in a tier crate. It fails the build
// when this crate is compiled without its tier flags. The list is scripts/tier-features/amd64_v2.txt.
#[cfg(not(all(
    target_feature = "cmpxchg16b",
    target_feature = "lahfsahf",
    target_feature = "popcnt",
    target_feature = "sse3",
    target_feature = "sse4.1",
    target_feature = "sse4.2",
    target_feature = "ssse3"
)))]
compile_error!("amd64_v2 was built without its tier flags; see cafetensor-lib/README.md");

use generic_operations::{Engine, Kernels, Operations};

/// Private on purpose. A crate that can name this type can instantiate generics against it at
/// its own (baseline) flags.
struct V2;

impl Kernels for V2 {
    #[inline(always)]
    fn xor64(a: &[u8; 64], b: &[u8; 64], out: &mut [u8; 64]) {
        use core::arch::x86_64::*;
        for i in 0..4 {
            let at = 16 * i;
            unsafe {
                let r = _mm_xor_si128(
                    _mm_loadu_si128(a[at..].as_ptr().cast()),
                    _mm_loadu_si128(b[at..].as_ptr().cast()),
                );
                _mm_storeu_si128(out[at..].as_mut_ptr().cast(), r);
            }
        }
    }
}

/// The vtable is built here, so every `Engine<V2>` method is codegen'd here.
static ENGINE: Engine<V2> = Engine::new("amd64_v2");

/// The crate's only public item. Non-generic and out of line, so it is codegen'd in this crate
/// and inlined nowhere else.
#[inline(never)]
pub fn operations() -> &'static dyn Operations {
    &ENGINE
}
