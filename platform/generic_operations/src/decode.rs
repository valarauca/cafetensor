//! Chunk decoder. A scalar reference path decodes any suffix of a chunk; tiers add vector
//! paths through [`Kernels`] hooks that decode whole groups first. Ported from the old
//! project's `codec.rs` (scalar) and `avx512.rs` (vector).

use core::simd::prelude::*;

use crate::codebook::{Codebook, EXP_BITS, ExpAlias, MAX_CTX, SM_BITS, SmMode, Tier};
use crate::format::{Format, plane_len};
use crate::rans::{LANES, RANS_L, WAYS};
use crate::{Kernels, OpError};

fn dec_table_into(freqs: &[u16; 256], bits: u32, dst: &mut [u32]) {
    let mut start = 0u32;
    for (s, &f) in freqs.iter().enumerate() {
        let f = f as u32;
        for k in 0..f {
            dst[(start + k) as usize] = s as u32 | ((f - 1) << 8) | (k << (8 + bits));
        }
        start += f;
    }
}

/// Decoder view of a [`Codebook`], built once per tensor. Entries pack
/// `sym | (freq - 1) << 8 | rank << (8 + bits)`. The sign|mantissa table always spans
/// [`MAX_CTX`] contexts so any 4-bit context index stays in bounds.
#[derive(Debug, Clone)]
pub struct DecTables {
    pub exp: [u32; 1 << EXP_BITS],
    pub esc: Option<u8>,
    pub alias: ExpAlias,
    pub sm: [u32; MAX_CTX << SM_BITS],
    pub ctx_of: [u8; 256],
    /// `ctx_of[e] << SM_BITS`, the offset of each exponent's context in `sm`.
    pub sm_base: [u32; 256],
    /// `ctx_of` packed eight contexts per word: exponent `e` has context
    /// `(ctx_nibbles[e / 8] >> (4 * (e % 8))) & 15`.
    pub ctx_nibbles: [u32; 32],
    pub mode: SmMode,
}

impl DecTables {
    /// Build decoder tables from a codebook. `None` if its exponent table has no alias layout.
    pub fn new(cb: &Codebook) -> Option<Self> {
        let alias = ExpAlias::new(&cb.exp_freqs, cb.tier)?;
        let mut t = DecTables {
            exp: [0; 1 << EXP_BITS],
            esc: cb.esc,
            alias,
            sm: [0; MAX_CTX << SM_BITS],
            ctx_of: [0; 256],
            sm_base: [0; 256],
            ctx_nibbles: [0; 32],
            mode: cb.sm_mode(),
        };
        for (slot, e) in t.exp.iter_mut().enumerate() {
            *e = t.alias.flat(slot as u32);
        }
        if let Some(m) = &cb.sm {
            for (dst, f) in
                t.sm.as_chunks_mut::<{ 1 << SM_BITS }>()
                    .0
                    .iter_mut()
                    .zip(&m.freqs)
                    .take(m.n_ctx)
            {
                dec_table_into(f, SM_BITS, dst);
            }
            t.ctx_of = m.ctx_of;
            for (e, &c) in m.ctx_of.iter().enumerate() {
                let c = c as u32 % MAX_CTX as u32;
                t.sm_base[e] = c << SM_BITS;
                t.ctx_nibbles[e / 8] |= c << (4 * (e % 8));
            }
        }
        Some(t)
    }
}

/// The streams and raw planes of one chunk, located by the caller from the chunk index.
#[derive(Debug, Clone, Copy)]
pub struct ChunkRef<'a> {
    pub elems: usize,
    pub exp: [&'a [u8]; WAYS],
    pub sm: [&'a [u8]; WAYS],
    pub esc: [&'a [u8]; WAYS],
    /// High residual bytes, one per element, for raw full-byte residuals.
    pub hi: &'a [u8],
    /// High residual bitplanes for narrower raw residuals.
    pub bits: &'a [u8],
    /// Low mantissa bytes.
    pub lo: &'a [u8],
}

#[inline(always)]
fn step(x: u32, e: u32, bits: u32) -> u32 {
    let mask = (1u32 << bits) - 1;
    (((e >> 8) & mask) + 1)
        .wrapping_mul(x >> bits)
        .wrapping_add(e >> (8 + bits))
}

/// Forward reader over a little-endian `u16` word stream.
#[derive(Debug, Clone, Copy)]
pub struct Reader<'a> {
    pub bytes: &'a [u8],
    pub pos: usize,
}

impl Reader<'_> {
    /// Stream length in words.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.bytes.len() / 2
    }

    /// Whether the stream holds no words.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline(always)]
    fn peek(&self) -> u32 {
        match self.bytes.get(2 * self.pos..2 * self.pos + 2) {
            Some(b) => u16::from_le_bytes([b[0], b[1]]) as u32,
            None => 0,
        }
    }

    fn next(&mut self) -> Result<u32, OpError> {
        if self.pos >= self.len() {
            return Err(OpError::Corrupt);
        }
        let v = self.peek();
        self.pos += 1;
        Ok(v)
    }

    /// Branch-free renormalization: always peek, advance only if needed. An out-of-range peek
    /// yields 0 and overruns `pos`, which the end-of-chunk check rejects.
    #[inline(always)]
    fn refill(&mut self, x: u32) -> u32 {
        let w = self.peek();
        let need = x < RANS_L;
        self.pos += need as usize;
        if need { (x << 16) | w } else { x }
    }
}

/// Decoder state of one chunk: lane states and stream cursors of every way.
pub struct ChunkState<'a> {
    pub fmt: Format,
    pub e: [[u32; LANES]; WAYS],
    pub s: [[u32; LANES]; WAYS],
    pub re: [Reader<'a>; WAYS],
    pub rs: [Reader<'a>; WAYS],
    pub esc: [&'a [u8]; WAYS],
    pub esc_pos: [usize; WAYS],
    pub chunk: ChunkRef<'a>,
    pub plane_len: usize,
    pub bad: bool,
}

fn init_lanes(r: &mut Reader) -> Result<[u32; LANES], OpError> {
    let mut lanes = [0u32; LANES];
    for x in lanes.iter_mut() {
        let hi = r.next()?;
        *x = (hi << 16) | r.next()?;
    }
    Ok(lanes)
}

impl<'a> ChunkState<'a> {
    /// Validate the chunk's stream and plane sizes against `fmt` and `t`, then read the initial
    /// lane states of every way.
    pub fn new(t: &DecTables, fmt: Format, chunk: ChunkRef<'a>) -> Result<Self, OpError> {
        let n = chunk.elems;
        let coded = t.mode == SmMode::Coded;
        if coded && fmt.hi_bits() != 8 {
            return Err(OpError::Corrupt);
        }
        let hi_ok = match (coded, fmt.hi_bits()) {
            (true, _) => chunk.hi.is_empty() && chunk.bits.is_empty(),
            (false, 8) => chunk.hi.len() == n && chunk.bits.is_empty(),
            (false, h) => chunk.hi.is_empty() && chunk.bits.len() == h as usize * plane_len(n),
        };
        let streams_ok = (0..WAYS).all(|w| {
            chunk.exp[w].len().is_multiple_of(2)
                && chunk.sm[w].len().is_multiple_of(2)
                && (coded || chunk.sm[w].is_empty())
                && (t.esc.is_some() || chunk.esc[w].is_empty())
        });
        if !hi_ok || !streams_ok || chunk.lo.len() != n * fmt.low_bytes() {
            return Err(OpError::Corrupt);
        }
        let mut st = ChunkState {
            fmt,
            e: [[0; LANES]; WAYS],
            s: [[RANS_L; LANES]; WAYS],
            re: core::array::from_fn(|w| Reader {
                bytes: chunk.exp[w],
                pos: 0,
            }),
            rs: core::array::from_fn(|w| Reader {
                bytes: chunk.sm[w],
                pos: 0,
            }),
            esc: chunk.esc,
            esc_pos: [0; WAYS],
            chunk,
            plane_len: plane_len(n),
            bad: false,
        };
        for w in 0..WAYS {
            st.e[w] = init_lanes(&mut st.re[w])?;
            if coded {
                st.s[w] = init_lanes(&mut st.rs[w])?;
            }
        }
        Ok(st)
    }

    /// Decode elements `first..` into `out` (`fmt.width` bytes per element, the whole chunk).
    /// `first` must be a multiple of [`LANES`].
    #[inline]
    pub fn decode_scalar(&mut self, t: &DecTables, out: &mut [u8], first: usize) {
        let emask = (1u32 << EXP_BITS) - 1;
        let smask = (1u32 << SM_BITS) - 1;
        let fmt = self.fmt;
        let width = fmt.width;
        let low = fmt.low_bytes();
        for (i, o) in out.chunks_exact_mut(width).enumerate().skip(first) {
            let l = i % LANES;
            let w = (i / LANES) % WAYS;
            let x = self.e[w][l];
            let ee = t.exp[(x & emask) as usize];
            self.e[w][l] = self.re[w].refill(step(x, ee, EXP_BITS));
            let mut exp = ee as u8;
            if Some(exp) == t.esc {
                match self.esc[w].get(self.esc_pos[w]) {
                    Some(&b) => exp = b,
                    None => self.bad = true,
                }
                self.esc_pos[w] += 1;
            }
            let hi = match (t.mode, fmt.hi_bits()) {
                (SmMode::Coded, _) => {
                    let x = self.s[w][l];
                    let slot = t.sm_base[exp as usize] as usize | (x & smask) as usize;
                    let es = t.sm[slot % t.sm.len()];
                    self.s[w][l] = self.rs[w].refill(step(x, es, SM_BITS));
                    es as u8
                }
                (SmMode::Raw, 8) => self.chunk.hi[i],
                (SmMode::Raw, h) => (0..h as usize).fold(0u8, |acc, j| {
                    let at = j * self.plane_len + 2 * (i / LANES);
                    let word = u16::from_le_bytes([self.chunk.bits[at], self.chunk.bits[at + 1]]);
                    acc | ((((word >> (i % LANES)) & 1) as u8) << j)
                }),
            };
            let lo = self.chunk.lo[i * low..(i + 1) * low]
                .iter()
                .rev()
                .fold(0u32, |acc, &b| (acc << 8) | b as u32);
            let v = fmt.join(exp, hi, lo);
            o.copy_from_slice(&v.to_le_bytes()[..width]);
        }
    }

    /// Every lane must land on the encoder's initial state with every stream consumed.
    pub fn finish(&self) -> Result<(), OpError> {
        let done = self.re.iter().chain(&self.rs).all(|r| r.pos == r.len())
            && (0..WAYS).all(|w| self.esc_pos[w] == self.esc[w].len())
            && !self.bad;
        let home = self.e.iter().chain(&self.s).flatten().all(|&x| x == RANS_L);
        if done && home {
            Ok(())
        } else {
            Err(OpError::Corrupt)
        }
    }
}

/// Byte shuffles that widen up to four consecutive `u16` words into the `u32` lanes of a
/// quarter whose renormalization mask is the index. Index 0xFF selects zero.
static EXPAND: [[u8; 16]; 16] = {
    let mut t = [[0xFFu8; 16]; 16];
    let mut k = 0;
    while k < 16 {
        let mut r = 0u8;
        let mut l = 0;
        while l < 4 {
            if (k >> l) & 1 == 1 {
                t[k][4 * l] = 2 * r;
                t[k][4 * l + 1] = 2 * r + 1;
                r += 1;
            }
            l += 1;
        }
        k += 1;
    }
    t
};

/// Renormalize the lanes of `x` below [`RANS_L`] from the next words of `words`, in lane
/// order, one table-driven [`Kernels::shuffle_bytes`] per four lanes. Returns the new states and
/// the number of words taken.
#[inline(always)]
pub(crate) fn refill_shuffle<K: Kernels + ?Sized>(x: u32x16, words: &[u8; 32]) -> (u32x16, usize) {
    let k = x.simd_lt(u32x16::splat(RANS_L)).to_bitmask();
    let mut bytes = [0u8; 64];
    let mut base = 0usize;
    for (q, dst) in bytes.as_chunks_mut::<16>().0.iter_mut().enumerate() {
        let kq = (k >> (4 * q)) as usize & 15;
        let mut src = [0u8; 16];
        src[..8].copy_from_slice(&words[2 * base..2 * base + 8]);
        *dst = K::shuffle_bytes(u8x16::from_array(src), u8x16::from_array(EXPAND[kq])).to_array();
        base += kq.count_ones() as usize;
    }
    let w = u32x16::from_le_bytes(u8x64::from_array(bytes));
    let m = Mask::<i32, LANES>::from_bitmask(k);
    (m.select((x << 16) | w, x), k.count_ones() as usize)
}

#[inline(always)]
fn step_v(x: u32x16, e: u32x16, bits: u32) -> u32x16 {
    let mask = u32x16::splat((1 << bits) - 1);
    (((e >> 8) & mask) + u32x16::splat(1)) * (x >> bits) + (e >> (8 + bits))
}

#[inline(always)]
fn refill_stream<K: Kernels>(x: u32x16, r: &mut Reader) -> u32x16 {
    let words = r
        .bytes
        .get(2 * r.pos..)
        .and_then(<[u8]>::first_chunk::<32>)
        .unwrap_or(&[0; 32]);
    let (x, n) = K::refill(x, words);
    r.pos += n;
    x
}

#[inline(always)]
fn join_v<const W: usize, const M: u32>(exp: u32x16, hi: u32x16, lo: u32x16) -> u32x16 {
    let low = 8 * (M / 8);
    let h = 1 + M - low;
    let sign = ((hi >> (h - 1)) & u32x16::splat(1)) << (8 * W as u32 - 1);
    let mant_hi = (hi & u32x16::splat((1 << (h - 1)) - 1)) << low;
    sign | (exp << M) | mant_hi | lo
}

impl ChunkState<'_> {
    #[inline(always)]
    fn room(&self, coded: bool) -> bool {
        let fits = |r: &Reader| r.pos + LANES <= r.len();
        !self.bad && self.re.iter().all(fits) && (!coded || self.rs.iter().all(fits))
    }

    /// Raw high residual of the group at element `at`.
    #[inline(always)]
    fn raw_hi(&self, at: usize, h: u32) -> u32x16 {
        if h == 8 {
            return u8x16::from_slice(&self.chunk.hi[at..at + LANES]).cast();
        }
        let mut v = u32x16::splat(0);
        for j in 0..h as usize {
            let w = j * self.plane_len + 2 * (at / LANES);
            let k = u16::from_le_bytes([self.chunk.bits[w], self.chunk.bits[w + 1]]);
            v |= Mask::<i32, LANES>::from_bitmask(k as u64)
                .select(u32x16::splat(1 << j), u32x16::splat(0));
        }
        v
    }

    /// Raw low mantissa bytes of the group at element `at`.
    #[inline(always)]
    fn raw_lo(&self, at: usize, low: usize) -> u32x16 {
        match low {
            0 => u32x16::splat(0),
            1 => u8x16::from_slice(&self.chunk.lo[at..at + LANES]).cast(),
            _ => {
                u16x16::from_le_bytes(u8x32::from_slice(&self.chunk.lo[2 * at..2 * at + 32])).cast()
            }
        }
    }
}

/// Decode whole lines of `WAYS * LANES` elements while every stream holds a full refill, and
/// return the number of elements decoded.
#[inline]
fn decode_lines<
    K: Kernels,
    const CODED: bool,
    const N: usize,
    const ESC: bool,
    const W: usize,
    const M: u32,
>(
    t: &DecTables,
    st: &mut ChunkState,
    out: &mut [u8],
) -> usize {
    let low = (M / 8) as usize;
    let h = 1 + M - 8 * (M / 8);
    let emask = u32x16::splat((1 << EXP_BITS) - 1);
    let smask = u32x16::splat((1 << SM_BITS) - 1);
    let byte = u32x16::splat(0xFF);
    let esc = u32x16::splat(t.esc.map_or(256, u32::from));
    let mut xe = st.e.map(u32x16::from_array);
    let mut xs = st.s.map(u32x16::from_array);
    let mut done = 0;
    for line in out.chunks_exact_mut(W * LANES * WAYS) {
        if !st.room(CODED) {
            break;
        }
        for (w, dst) in line.chunks_exact_mut(W * LANES).enumerate() {
            let at = (done * WAYS + w) * LANES;
            let e = K::exp_entries::<N>(t, xe[w] & emask);
            xe[w] = refill_stream::<K>(step_v(xe[w], e, EXP_BITS), &mut st.re[w]);
            let mut exp = e & byte;
            if ESC {
                let m = exp.simd_eq(esc).to_bitmask() as u16;
                if m != 0 {
                    let n = m.count_ones() as usize;
                    match st.esc[w].get(st.esc_pos[w]..st.esc_pos[w] + n) {
                        Some(b) => exp = K::escapes(exp, m, b),
                        None => st.bad = true,
                    }
                    st.esc_pos[w] += n;
                }
            }
            let hi = if CODED {
                let es = K::sm_entries(t, K::sm_bases(t, exp) | (xs[w] & smask));
                xs[w] = refill_stream::<K>(step_v(xs[w], es, SM_BITS), &mut st.rs[w]);
                es & byte
            } else {
                st.raw_hi(at, h)
            };
            let v = join_v::<W, M>(exp, hi, st.raw_lo(at, low));
            match W {
                1 => dst.copy_from_slice(v.cast::<u8>().as_array()),
                2 => dst.copy_from_slice(v.cast::<u16>().to_le_bytes().as_array()),
                _ => dst.copy_from_slice(v.to_le_bytes().as_array()),
            }
        }
        done += 1;
    }
    st.e = xe.map(u32x16::to_array);
    st.s = xs.map(u32x16::to_array);
    done * LANES * WAYS
}

/// Decode the leading whole lines of a chunk with the vector path specialized for its format,
/// sign|mantissa mode, alias tier and escape use. Returns the number of elements decoded.
#[inline]
fn decode_groups<K: Kernels>(t: &DecTables, st: &mut ChunkState, out: &mut [u8]) -> usize {
    macro_rules! specialize {
        ($coded:literal, $w:literal, $m:literal) => {
            match (t.alias.tier, t.esc.is_some()) {
                (Tier::T16, false) => decode_lines::<K, $coded, 16, false, $w, $m>(t, st, out),
                (Tier::T32, false) => decode_lines::<K, $coded, 32, false, $w, $m>(t, st, out),
                (Tier::T64, false) => decode_lines::<K, $coded, 64, false, $w, $m>(t, st, out),
                (Tier::T64, true) => decode_lines::<K, $coded, 64, true, $w, $m>(t, st, out),
                _ => 0,
            }
        };
    }
    match (st.fmt, t.mode) {
        (Format::BF16, SmMode::Raw) => specialize!(false, 2, 7),
        (Format::BF16, SmMode::Coded) => specialize!(true, 2, 7),
        (Format::F32, SmMode::Raw) => specialize!(false, 4, 23),
        (Format::F32, SmMode::Coded) => specialize!(true, 4, 23),
        (Format::F16, SmMode::Raw) => specialize!(false, 2, 10),
        (Format::E4M3, SmMode::Raw) => specialize!(false, 1, 3),
        (Format::E5M2, SmMode::Raw) => specialize!(false, 1, 2),
        _ => 0,
    }
}

/// Decode one chunk into `out`, exactly `chunk.elems * fmt.width` bytes.
#[inline]
pub(crate) fn decode_chunk<K: Kernels>(
    t: &DecTables,
    fmt: Format,
    chunk: ChunkRef,
    out: &mut [u8],
) -> Result<(), OpError> {
    if out.len() != chunk.elems * fmt.width {
        return Err(OpError::OutputTooSmall);
    }
    if !matches!(
        fmt,
        Format::BF16 | Format::F32 | Format::F16 | Format::E4M3 | Format::E5M2
    ) {
        return Err(OpError::Corrupt);
    }
    let mut st = ChunkState::new(t, fmt, chunk)?;
    let first = decode_groups::<K>(t, &mut st, out);
    st.decode_scalar(t, out, first);
    st.finish()
}
