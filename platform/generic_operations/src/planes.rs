//! Element split into exponent / high residual / low byte planes, and bitplane packing.

use core::simd::prelude::*;

use crate::format::{Format, plane_len};
use crate::{Kernels, OpError};

/// Split little-endian elements of `fmt` into planes. `exps` and `his` take one byte per
/// element, `los` takes `fmt.low_bytes()` bytes per element.
#[inline]
#[allow(
    clippy::extra_unused_type_parameters,
    reason = "algorithms are generic over K so each tier instantiates them with its own flags"
)]
pub(crate) fn split_planes<K: Kernels>(
    fmt: Format,
    bytes: &[u8],
    exps: &mut [u8],
    his: &mut [u8],
    los: &mut [u8],
) -> Result<(), OpError> {
    if !bytes.len().is_multiple_of(fmt.width) {
        return Err(OpError::Corrupt);
    }
    let n = bytes.len() / fmt.width;
    let exps = exps.get_mut(..n).ok_or(OpError::OutputTooSmall)?;
    let his = his.get_mut(..n).ok_or(OpError::OutputTooSmall)?;
    let los = los
        .get_mut(..n * fmt.low_bytes())
        .ok_or(OpError::OutputTooSmall)?;
    match fmt {
        Format::BF16 => split_w2::<7>(bytes, exps, his, los),
        Format::F16 => split_w2::<10>(bytes, exps, his, los),
        Format::F32 => split_w4::<23>(bytes, exps, his, los),
        Format::E4M3 => split_w1::<3>(bytes, exps, his),
        Format::E5M2 => split_w1::<2>(bytes, exps, his),
        _ => return Err(OpError::Corrupt),
    }
    Ok(())
}

#[inline]
fn split_scalar(fmt: Format, bytes: &[u8], exps: &mut [u8], his: &mut [u8], los: &mut [u8]) {
    let low = fmt.low_bytes();
    for (i, el) in bytes.chunks_exact(fmt.width).enumerate() {
        let v = el.iter().rev().fold(0u32, |a, &b| (a << 8) | b as u32);
        let (e, h, l) = fmt.split(v);
        exps[i] = e;
        his[i] = h;
        los[i * low..(i + 1) * low].copy_from_slice(&l.to_le_bytes()[..low]);
    }
}

#[inline]
fn split_w1<const M: u32>(bytes: &[u8], exps: &mut [u8], his: &mut [u8]) {
    let fmt = Format {
        width: 1,
        mant_bits: M,
    };
    let emask = u8x64::splat(((1u32 << fmt.exp_bits()) - 1) as u8);
    let mmask = u8x64::splat(((1u32 << M) - 1) as u8);
    let (blocks, tail) = bytes.as_chunks::<64>();
    let (eb, et) = exps.as_chunks_mut::<64>();
    let (hb, ht) = his.as_chunks_mut::<64>();
    for ((b, e), h) in blocks.iter().zip(eb.iter_mut()).zip(hb.iter_mut()) {
        let v = u8x64::from_array(*b);
        *e = ((v >> M as u8) & emask).to_array();
        *h = (((v >> 7) << M as u8) | (v & mmask)).to_array();
    }
    split_scalar(fmt, tail, et, ht, &mut []);
}

#[inline]
fn split_w2<const M: u32>(bytes: &[u8], exps: &mut [u8], his: &mut [u8], los: &mut [u8]) {
    let fmt = Format {
        width: 2,
        mant_bits: M,
    };
    let low = 8 * fmt.low_bytes() as u16;
    let h = fmt.hi_bits() as u16;
    let emask = u16x64::splat(((1u32 << fmt.exp_bits()) - 1) as u16);
    let mmask = u16x64::splat(((1u32 << M) - 1) as u16);
    let (blocks, tail) = bytes.as_chunks::<128>();
    let n = blocks.len() * 64;
    let (eb, et) = exps.split_at_mut(n);
    let (hb, ht) = his.split_at_mut(n);
    let (lb, lt) = los.split_at_mut(n * fmt.low_bytes());
    for (i, b) in blocks.iter().enumerate() {
        let v = u16x64::from_array(core::array::from_fn(|j| {
            u16::from_le_bytes([b[2 * j], b[2 * j + 1]])
        }));
        let e: u8x64 = ((v >> M as u16) & emask).cast();
        let hi: u8x64 = (((v >> 15) << (h - 1)) | ((v & mmask) >> low)).cast();
        eb[64 * i..64 * i + 64].copy_from_slice(e.as_array());
        hb[64 * i..64 * i + 64].copy_from_slice(hi.as_array());
        if low > 0 {
            let l: u8x64 = v.cast();
            lb[64 * i..64 * i + 64].copy_from_slice(l.as_array());
        }
    }
    split_scalar(fmt, tail, et, ht, lt);
}

#[inline]
fn split_w4<const M: u32>(bytes: &[u8], exps: &mut [u8], his: &mut [u8], los: &mut [u8]) {
    let fmt = Format {
        width: 4,
        mant_bits: M,
    };
    let low = 8 * fmt.low_bytes() as u32;
    let h = fmt.hi_bits();
    let emask = u32x16::splat((1u32 << fmt.exp_bits()) - 1);
    let mmask = u32x16::splat((1u32 << M) - 1);
    let (blocks, tail) = bytes.as_chunks::<64>();
    let n = blocks.len() * 16;
    let (eb, et) = exps.split_at_mut(n);
    let (hb, ht) = his.split_at_mut(n);
    let (lb, lt) = los.split_at_mut(n * fmt.low_bytes());
    for (i, b) in blocks.iter().enumerate() {
        let v = u32x16::from_array(core::array::from_fn(|j| {
            u32::from_le_bytes([b[4 * j], b[4 * j + 1], b[4 * j + 2], b[4 * j + 3]])
        }));
        let e: u8x16 = ((v >> M) & emask).cast();
        let hi: u8x16 = (((v >> 31) << (h - 1)) | ((v & mmask) >> low)).cast();
        let l: u16x16 = v.cast();
        eb[16 * i..16 * i + 16].copy_from_slice(e.as_array());
        hb[16 * i..16 * i + 16].copy_from_slice(hi.as_array());
        for (j, w) in l.to_array().iter().enumerate() {
            lb[32 * i + 2 * j..32 * i + 2 * j + 2].copy_from_slice(&w.to_le_bytes());
        }
    }
    split_scalar(fmt, tail, et, ht, lt);
}

/// Pack bit `j` of every high residual into plane `j`, for `j < hi_bits`. Plane `j` starts at
/// `j * plane_len(n)` and holds one little-endian `u16` per group of 16 elements.
#[inline]
#[allow(
    clippy::extra_unused_type_parameters,
    reason = "algorithms are generic over K so each tier instantiates them with its own flags"
)]
pub(crate) fn pack_bitplanes<K: Kernels>(
    his: &[u8],
    hi_bits: u32,
    out: &mut [u8],
) -> Result<(), OpError> {
    if !(1..=8).contains(&hi_bits) {
        return Err(OpError::Corrupt);
    }
    let pl = plane_len(his.len());
    let out = out
        .get_mut(..hi_bits as usize * pl)
        .ok_or(OpError::OutputTooSmall)?;
    if pl == 0 {
        return Ok(());
    }
    let (blocks, tail) = his.as_chunks::<64>();
    for (j, plane) in out.chunks_exact_mut(pl).enumerate() {
        let (pb, pt) = plane.split_at_mut(8 * blocks.len());
        for (b, p) in blocks.iter().zip(pb.as_chunks_mut::<8>().0) {
            let bits =
                ((u8x64::from_array(*b) >> j as u8) & u8x64::splat(1)).simd_ne(u8x64::splat(0));
            *p = bits.to_bitmask().to_le_bytes();
        }
        for (g, w) in tail.chunks(16).zip(pt.as_chunks_mut::<2>().0) {
            let word = g
                .iter()
                .enumerate()
                .fold(0u16, |a, (l, &v)| a | ((((v >> j) & 1) as u16) << l));
            *w = word.to_le_bytes();
        }
    }
    Ok(())
}
