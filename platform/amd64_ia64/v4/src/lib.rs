//! `amd64_v4` tier: built with `-Ctarget-cpu=x86-64-v4`.
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

use core::arch::x86_64::*;
use core::simd::u32x16;

use generic_operations::codebook::{EXP_BITS, MAX_CTX, SM_BITS};
use generic_operations::decode::DecTables;
use generic_operations::rans::RANS_L;
use generic_operations::{Engine, Kernels, Operations};

/// Private on purpose. A crate that can name this type can instantiate generics against it at
/// its own (baseline) flags.
struct V4;

impl Kernels for V4 {
    const ENCODE_BRANCHLESS: bool = true;
    const BLAKE3_LANES: usize = 16;

    #[inline(always)]
    fn xor64(a: &[u8; 64], b: &[u8; 64], out: &mut [u8; 64]) {
        unsafe {
            let r = _mm512_xor_si512(
                _mm512_loadu_si512(a.as_ptr().cast()),
                _mm512_loadu_si512(b.as_ptr().cast()),
            );
            _mm512_storeu_si512(out.as_mut_ptr().cast(), r);
        }
    }

    #[inline(always)]
    fn crc32c_u64(state: u32, word: u64) -> u32 {
        unsafe { _mm_crc32_u64(state as u64, word) as u32 }
    }

    #[inline(always)]
    fn exp_entries<const N: usize>(t: &DecTables, slot: u32x16) -> u32x16 {
        let shift = EXP_BITS - N.trailing_zeros();
        let b: __m512i = (slot >> shift).into();
        let j = slot & u32x16::splat((1 << shift) - 1);
        let div = t.alias.div.as_chunks::<16>().0;
        let div = core::array::from_fn(|i| unsafe {
            _mm512_cvtepu8_epi32(_mm_loadu_si128(div[i].as_ptr().cast()))
        });
        let e: u32x16 = unsafe {
            let side = _mm512_cmpge_epu32_mask(j.into(), lookup::<N>(&div, b));
            _mm512_mask_blend_epi32(
                side,
                lookup::<N>(&regs(&t.alias.lo), b),
                lookup::<N>(&regs(&t.alias.hi), b),
            )
        }
        .into();
        let k = (j + (e >> 20)) & u32x16::splat((1 << EXP_BITS) - 1);
        (e & u32x16::splat(0xF_FFFF)) | (k << 20)
    }

    #[inline(always)]
    fn refill(x: u32x16, words: &[u8; 32]) -> (u32x16, usize) {
        let x: __m512i = x.into();
        unsafe {
            let k = _mm512_cmplt_epu32_mask(x, _mm512_set1_epi32(RANS_L as i32));
            let w = _mm512_cvtepu16_epi32(_mm256_loadu_si256(words.as_ptr().cast()));
            let w = _mm512_maskz_expand_epi32(k, w);
            let x = _mm512_mask_or_epi32(x, k, _mm512_slli_epi32::<16>(x), w);
            (x.into(), k.count_ones() as usize)
        }
    }

    #[inline(always)]
    fn sm_bases(t: &DecTables, exp: u32x16) -> u32x16 {
        let e = exp & u32x16::splat(0xFF);
        let c = t.ctx_nibbles.as_chunks::<16>().0;
        let words: u32x16 = unsafe {
            _mm512_permutex2var_epi32(
                _mm512_loadu_si512(c[0].as_ptr().cast()),
                (e >> 3).into(),
                _mm512_loadu_si512(c[1].as_ptr().cast()),
            )
        }
        .into();
        let ctx = (words >> ((e & u32x16::splat(7)) << 2)) & u32x16::splat(MAX_CTX as u32 - 1);
        ctx << SM_BITS
    }

    #[inline(always)]
    fn sm_entries(t: &DecTables, slot: u32x16) -> u32x16 {
        let i: __m512i = (slot & u32x16::splat(t.sm.len() as u32 - 1)).into();
        unsafe { _mm512_i32gather_epi32::<4>(i, t.sm.as_ptr().cast()).into() }
    }
}

/// Four registers of a 64-entry table.
#[inline(always)]
fn regs(v: &[u32; 64]) -> [__m512i; 4] {
    let c = v.as_chunks::<16>().0;
    core::array::from_fn(|i| unsafe { _mm512_loadu_si512(c[i].as_ptr().cast()) })
}

/// Look up an `N`-entry `u32` table held in registers, `N` in {16, 32, 64}.
#[inline(always)]
fn lookup<const N: usize>(t: &[__m512i; 4], b: __m512i) -> __m512i {
    unsafe {
        match N {
            16 => _mm512_permutexvar_epi32(b, t[0]),
            32 => _mm512_permutex2var_epi32(t[0], b, t[1]),
            _ => {
                let upper = _mm512_test_epi32_mask(b, _mm512_set1_epi32(32));
                let lo = _mm512_permutex2var_epi32(t[0], b, t[1]);
                let hi = _mm512_permutex2var_epi32(t[2], b, t[3]);
                _mm512_mask_blend_epi32(upper, lo, hi)
            }
        }
    }
}

/// The vtable is built here, so every `Engine<V4>` method is codegen'd here.
static ENGINE: Engine<V4> = Engine::new("amd64_v4");

/// The crate's only public item. Non-generic and out of line, so it is codegen'd in this crate
/// and inlined nowhere else.
#[inline(never)]
pub fn operations() -> &'static dyn Operations {
    &ENGINE
}
