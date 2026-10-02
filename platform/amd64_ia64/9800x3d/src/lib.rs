//! `amd64_9800x3d` tier: built with `-Ctarget-cpu=znver5 -Ctarget-feature=-rdseed`.
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

use core::arch::x86_64::*;
use core::simd::u32x16;

use generic_operations::codebook::{EXP_BITS, MAX_CTX, SM_BITS};
use generic_operations::decode::DecTables;
use generic_operations::rans::RANS_L;
use generic_operations::{Engine, Kernels, Operations};

/// Private on purpose. A crate that can name this type can instantiate generics against it at
/// its own (baseline) flags.
struct Zen5;

impl Kernels for Zen5 {
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
    fn crc32c_blocks(state: u32, blocks: &[[u8; 256]]) -> u32 {
        use generic_operations::crc::{K128, K512, K2048};
        let Some((first, rest)) = blocks.split_first() else {
            return state;
        };
        unsafe {
            let pair = |k: (u64, u64)| _mm_set_epi64x(k.1 as i64, k.0 as i64);
            let k2048 = _mm512_broadcast_i32x4(pair(K2048));
            let k512 = _mm512_broadcast_i32x4(pair(K512));
            let k128 = pair(K128);
            let fold512 = |x: __m512i, k: __m512i| {
                _mm512_xor_si512(
                    _mm512_clmulepi64_epi128::<0x00>(x, k),
                    _mm512_clmulepi64_epi128::<0x11>(x, k),
                )
            };
            let fold128 = |x: __m128i, k: __m128i| {
                _mm_xor_si128(
                    _mm_clmulepi64_si128::<0x00>(x, k),
                    _mm_clmulepi64_si128::<0x11>(x, k),
                )
            };
            let load = |b: &[u8; 256], j: usize| {
                _mm512_loadu_si512(b[64 * j..64 * j + 64].as_ptr().cast())
            };
            let mut acc = [
                load(first, 0),
                load(first, 1),
                load(first, 2),
                load(first, 3),
            ];
            acc[0] = _mm512_xor_si512(
                acc[0],
                _mm512_zextsi128_si512(_mm_cvtsi32_si128(state as i32)),
            );
            for b in rest {
                for (j, a) in acc.iter_mut().enumerate() {
                    *a = _mm512_xor_si512(fold512(*a, k2048), load(b, j));
                }
            }
            let mut x = acc[0];
            for &a in &acc[1..] {
                x = _mm512_xor_si512(fold512(x, k512), a);
            }
            let lanes = [
                _mm512_extracti32x4_epi32::<0>(x),
                _mm512_extracti32x4_epi32::<1>(x),
                _mm512_extracti32x4_epi32::<2>(x),
                _mm512_extracti32x4_epi32::<3>(x),
            ];
            let mut r = lanes[0];
            for &l in &lanes[1..] {
                r = _mm_xor_si128(fold128(r, k128), l);
            }
            let c = _mm_crc32_u64(0, _mm_cvtsi128_si64(r) as u64);
            _mm_crc32_u64(c, _mm_extract_epi64::<1>(r) as u64) as u32
        }
    }

    #[inline(always)]
    fn exp_entries<const N: usize>(t: &DecTables, slot: u32x16) -> u32x16 {
        let shift = EXP_BITS - N.trailing_zeros();
        let b: __m512i = (slot >> shift).into();
        let j = slot & u32x16::splat((1 << shift) - 1);
        let e: u32x16 = unsafe {
            let div = _mm512_loadu_si512(t.alias.div.as_ptr().cast());
            let div = _mm512_and_si512(_mm512_permutexvar_epi8(b, div), _mm512_set1_epi32(0xFF));
            let side = _mm512_cmpge_epu32_mask(j.into(), div);
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
        let c = t.ctx_of.as_chunks::<64>().0;
        let e: __m512i = exp.into();
        let ctx: u32x16 = unsafe {
            let r = |i: usize| _mm512_loadu_si512(c[i].as_ptr().cast());
            let lo = _mm512_permutex2var_epi8(r(0), e, r(1));
            let hi = _mm512_permutex2var_epi8(r(2), e, r(3));
            let upper = _mm512_test_epi32_mask(e, _mm512_set1_epi32(0x80));
            _mm512_mask_blend_epi32(upper, lo, hi)
        }
        .into();
        (ctx & u32x16::splat(MAX_CTX as u32 - 1)) << SM_BITS
    }

    #[inline(always)]
    fn sm_entries(t: &DecTables, slot: u32x16) -> u32x16 {
        let i: __m512i = (slot & u32x16::splat(t.sm.len() as u32 - 1)).into();
        unsafe { _mm512_i32gather_epi32::<4>(i, t.sm.as_ptr().cast()).into() }
    }

    #[inline(always)]
    fn escapes(exp: u32x16, m: u16, esc: &[u8]) -> u32x16 {
        if esc.len() < m.count_ones() as usize {
            return exp;
        }
        unsafe {
            let real = _mm512_cvtepu8_epi32(_mm_maskz_expandloadu_epi8(m, esc.as_ptr().cast()));
            _mm512_mask_mov_epi32(exp.into(), m, real).into()
        }
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

/// The vtable is built here, so every `Engine<Zen5>` method is codegen'd here.
static ENGINE: Engine<Zen5> = Engine::new("amd64_9800x3d");

/// The crate's only public item. Non-generic and out of line, so it is codegen'd in this crate
/// and inlined nowhere else.
#[inline(never)]
pub fn operations() -> &'static dyn Operations {
    &ENGINE
}
