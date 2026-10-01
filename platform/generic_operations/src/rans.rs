//! Static interleaved rANS: 32-bit states, 16-bit renormalization, [`LANES`] states per stream
//! and [`WAYS`] independent stream sets per chunk. Element `i` of a chunk belongs to lane
//! `i % LANES` of way `(i / LANES) % WAYS`. Ported from the old project's `codec.rs`; the word
//! streams are byte-identical.

use crate::codebook::{Codebook, EXP_BITS, ExpAlias, MAX_CTX, SM_BITS};
use crate::{Kernels, OpError};

/// Interleaved rANS states per stream.
pub const LANES: usize = 16;
/// Independent stream sets per chunk.
pub const WAYS: usize = 4;
/// Lower bound of the normalized rANS state interval.
pub const RANS_L: u32 = 1 << 16;

/// Encoder entry for one symbol.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EncSym {
    pub start: u32,
    pub freq: u32,
    pub x_max: u64,
    /// `ceil(2^64 / freq)` for `freq >= 2`, so `x / freq == mulhi(rcp, x)` exactly for every
    /// 32-bit `x` (Lemire, Kaser and Kurz, "Faster remainder by direct computation").
    pub rcp: u64,
}

/// `(x / freq, x % freq)` without a divide instruction. A zero frequency yields `(0, x)`; the
/// caller rejects it.
#[inline(always)]
fn divmod(x: u32, e: &EncSym) -> (u32, u32) {
    let q = if e.freq == 1 {
        x
    } else {
        ((e.rcp as u128 * x as u128) >> 64) as u32
    };
    (q, x.wrapping_sub(q.wrapping_mul(e.freq)))
}

fn enc_table(freqs: &[u16; 256], bits: u32) -> [EncSym; 256] {
    let mut t = [EncSym::default(); 256];
    let mut start = 0u32;
    for (s, &f) in freqs.iter().enumerate() {
        let freq = f as u32;
        let rcp = if freq >= 2 {
            u64::MAX / freq as u64 + 1
        } else {
            0
        };
        t[s] = EncSym {
            start,
            freq,
            x_max: (freq as u64) << (32 - bits),
            rcp,
        };
        start += freq;
    }
    t
}

/// Encoder view of a [`Codebook`], built once per tensor.
#[derive(Debug, Clone)]
pub struct EncTables {
    pub exp: [EncSym; 256],
    /// Coded symbol of each exponent byte: itself, or the escape symbol.
    pub exp_code: [u8; 256],
    /// Alias slot of rank `k` of symbol `s`, at `exp[s].start + k`.
    pub exp_slot: [u16; 1 << EXP_BITS],
    /// Coded sign|mantissa tables per context, when `coded`.
    pub sm: [[EncSym; 256]; MAX_CTX],
    pub ctx_of: [u8; 256],
    pub coded: bool,
}

impl EncTables {
    /// Build encoder tables from a codebook. `None` if its exponent table has no alias layout.
    pub fn new(cb: &Codebook) -> Option<Self> {
        let exp = enc_table(&cb.exp_freqs, EXP_BITS);
        let alias = ExpAlias::new(&cb.exp_freqs, cb.tier)?;
        let mut exp_slot = [0u16; 1 << EXP_BITS];
        for slot in 0..1u32 << EXP_BITS {
            let e = alias.flat(slot);
            exp_slot[(exp[(e & 0xFF) as usize].start + (e >> 20)) as usize] = slot as u16;
        }
        let mut exp_code = [0u8; 256];
        for (e, c) in exp_code.iter_mut().enumerate() {
            *c = if cb.exp_freqs[e] > 0 {
                e as u8
            } else {
                cb.esc.unwrap_or(0)
            };
        }
        let mut t = EncTables {
            exp,
            exp_code,
            exp_slot,
            sm: [[EncSym::default(); 256]; MAX_CTX],
            ctx_of: [0; 256],
            coded: cb.sm.is_some(),
        };
        if let Some(m) = &cb.sm {
            for (dst, f) in t.sm.iter_mut().zip(&m.freqs).take(m.n_ctx) {
                *dst = enc_table(f, SM_BITS);
            }
            t.ctx_of = m.ctx_of;
        }
        Some(t)
    }
}

/// Stream sizes of one encoded chunk.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ChunkInfo {
    /// Per way: exponent words, sign|mantissa words, escape bytes.
    pub index: [[u32; 3]; WAYS],
    /// Bytes written to the output.
    pub len: usize,
}

/// Elements of way `w` in a chunk of `elems` elements.
pub const fn way_len(elems: usize, w: usize) -> usize {
    let groups = elems / LANES;
    let full = groups / WAYS + if groups % WAYS > w { 1 } else { 0 };
    let partial = if groups % WAYS == w { elems % LANES } else { 0 };
    full * LANES + partial
}

/// `u16` scratch an [`encode_chunk`] call needs for a chunk of `elems` elements.
pub const fn encode_scratch_len(elems: usize) -> usize {
    way_len(elems, 0) + 2 * LANES
}

/// Upper bound on the bytes [`encode_chunk`] writes for `elems` elements.
pub const fn encode_bound(elems: usize, coded: bool) -> usize {
    let streams = if coded { 2 } else { 1 };
    streams * 2 * (elems + WAYS * 2 * LANES) + elems
}

/// Encode one stream of way `w` backward into the tail of `scratch`; returns the start of the
/// written words, which read forward are the stream in decode order. Symbols are visited one
/// contiguous group of [`LANES`] at a time, last group first, lanes in reverse. With
/// [`Kernels::ENCODE_BRANCHLESS`] the candidate renormalization word is always written below
/// `pos`, and `pos` moves only when the word is kept; both forms write identical streams.
#[inline]
#[allow(
    clippy::too_many_arguments,
    reason = "both symbol planes and both table closures are needed per call"
)]
fn encode_stream<K: Kernels>(
    len: usize,
    w: usize,
    bits: u32,
    scratch: &mut [u16],
    exps: &[u8],
    his: &[u8],
    entry: impl Fn(u8, u8) -> EncSym,
    place: impl Fn(u32) -> u32,
) -> Result<usize, OpError> {
    let mut pos = scratch.len();
    let mut st = [RANS_L; LANES];
    let mut bad = false;
    for g in (0..len.div_ceil(LANES)).rev() {
        let base = (g * WAYS + w) * LANES;
        let cnt = (len - g * LANES).min(LANES);
        let es = exps.get(base..base + cnt).ok_or(OpError::Corrupt)?;
        let hs = his.get(base..base + cnt).ok_or(OpError::Corrupt)?;
        for (x, (&e, &h)) in st.iter_mut().zip(es.iter().zip(hs)).rev() {
            let sym = entry(e, h);
            bad |= sym.freq == 0;
            let mut v = *x;
            let need = v as u64 >= sym.x_max;
            if K::ENCODE_BRANCHLESS {
                let slot = pos.checked_sub(1).ok_or(OpError::OutputTooSmall)?;
                scratch[slot] = v as u16;
                pos -= need as usize;
                v = if need { v >> 16 } else { v };
            } else if need {
                pos = pos.checked_sub(1).ok_or(OpError::OutputTooSmall)?;
                scratch[pos] = v as u16;
                v >>= 16;
            }
            let (q, r) = divmod(v, &sym);
            *x = (q << bits).wrapping_add(place(sym.start.wrapping_add(r) & ((1 << bits) - 1)));
        }
    }
    if bad {
        return Err(OpError::Corrupt);
    }
    for lane in (0..LANES).rev() {
        for word in [st[lane] as u16, (st[lane] >> 16) as u16] {
            pos = pos.checked_sub(1).ok_or(OpError::OutputTooSmall)?;
            scratch[pos] = word;
        }
    }
    Ok(pos)
}

/// Append `words` to `out` at `at` as little-endian `u16`.
#[inline(always)]
fn emit(out: &mut [u8], words: &[u16], at: &mut usize) -> Result<(), OpError> {
    let dst = out
        .get_mut(*at..*at + 2 * words.len())
        .ok_or(OpError::OutputTooSmall)?;
    for (d, &w) in dst.as_chunks_mut::<2>().0.iter_mut().zip(words) {
        *d = w.to_le_bytes();
    }
    *at += 2 * words.len();
    Ok(())
}

/// Encode the exponent stream (and the coded sign|mantissa stream) of every way of one chunk,
/// writing per way: exponent words, sign|mantissa words, escape bytes. Raw planes are not
/// written here.
#[inline]
#[allow(
    clippy::extra_unused_type_parameters,
    reason = "algorithms are generic over K so each tier instantiates them with its own flags"
)]
pub(crate) fn encode_chunk<K: Kernels>(
    t: &EncTables,
    exps: &[u8],
    his: &[u8],
    scratch: &mut [u16],
    out: &mut [u8],
) -> Result<ChunkInfo, OpError> {
    if exps.len() != his.len() {
        return Err(OpError::Corrupt);
    }
    let n = exps.len();
    let scratch = scratch
        .get_mut(..encode_scratch_len(n))
        .ok_or(OpError::OutputTooSmall)?;
    let mut info = ChunkInfo::default();
    let mut at = 0usize;
    for (w, idx) in info.index.iter_mut().enumerate() {
        let len = way_len(n, w);
        let start = encode_stream::<K>(
            len,
            w,
            EXP_BITS,
            scratch,
            exps,
            his,
            |e, _| t.exp[t.exp_code[e as usize] as usize],
            |c| t.exp_slot[c as usize] as u32,
        )?;
        idx[0] = (scratch.len() - start) as u32;
        emit(out, &scratch[start..], &mut at)?;
        if t.coded {
            let start = encode_stream::<K>(
                len,
                w,
                SM_BITS,
                scratch,
                exps,
                his,
                |e, h| t.sm[t.ctx_of[e as usize] as usize][h as usize],
                |c| c,
            )?;
            idx[1] = (scratch.len() - start) as u32;
            emit(out, &scratch[start..], &mut at)?;
        }
        let mut escapes = 0u32;
        for g in 0..len.div_ceil(LANES) {
            let base = (g * WAYS + w) * LANES;
            let cnt = (len - g * LANES).min(LANES);
            for &e in &exps[base..base + cnt] {
                if t.exp_code[e as usize] != e {
                    *out.get_mut(at).ok_or(OpError::OutputTooSmall)? = e;
                    at += 1;
                    escapes += 1;
                }
            }
        }
        idx[2] = escapes;
    }
    info.len = at;
    Ok(info)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reciprocal_division_is_exact() {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        for freq in 0..=4096u32 {
            let rcp = if freq >= 2 {
                u64::MAX / freq as u64 + 1
            } else {
                0
            };
            let e = EncSym {
                start: 0,
                freq,
                x_max: 0,
                rcp,
            };
            let edge = [
                0u32,
                1,
                freq.saturating_sub(1),
                freq,
                freq + 1,
                u32::MAX,
                u32::MAX - 1,
                u32::MAX / 2,
            ];
            for i in 0..2000 {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let x = if i < edge.len() {
                    edge[i]
                } else {
                    (s >> 32) as u32 >> (i % 32)
                };
                let want = x
                    .checked_div(freq)
                    .zip(x.checked_rem(freq))
                    .unwrap_or((0, x));
                assert_eq!(divmod(x, &e), want, "x={x} freq={freq}");
            }
        }
    }
}
