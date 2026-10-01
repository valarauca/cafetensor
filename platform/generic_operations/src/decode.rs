//! Chunk decoder. A scalar reference path decodes any suffix of a chunk; tiers add vector
//! paths through [`Kernels`] hooks that decode whole groups first. Ported from the old
//! project's `codec.rs` (scalar) and `avx512.rs` (vector).

use crate::codebook::{Codebook, EXP_BITS, ExpAlias, MAX_CTX, SM_BITS, SmMode};
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
                    let base = (t.ctx_of[exp as usize] as usize) << SM_BITS;
                    let es = t.sm[base + (x & smask) as usize];
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
    let first = K::decode_groups(t, &mut st, out);
    st.decode_scalar(t, out, first);
    st.finish()
}
