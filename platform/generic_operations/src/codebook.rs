//! Per-tensor entropy model: frequency normalization, the exponent alias table, the
//! exponent-conditioned sign|mantissa contexts, and the codebook's on-disk form. Built once per
//! tensor, so this is plain code compiled at baseline, not a tier hot path. Ported from the old
//! project's `codec.rs` with identical output.

use crate::OpError;

/// Probability precision of the exponent model.
pub const EXP_BITS: u32 = 12;
/// Probability precision of every sign|mantissa model.
pub const SM_BITS: u32 = 11;
/// Upper bound on sign|mantissa contexts.
pub const MAX_CTX: usize = 16;
/// An exponent needs at least this many occurrences in a tensor to earn a dedicated context.
pub const MIN_CTX_COUNT: u64 = 4096;
/// Largest alias table, also the largest exponent alphabet coded without the escape symbol.
pub const ALIAS_BUCKETS: usize = 64;
/// Escape flag in the codebook tag byte.
const TAG_ESC: u8 = 0x80;
/// Upper bound on [`Codebook::write`] output.
pub const CODEBOOK_MAX_BYTES: usize = (32 + 512) + 2 + 1 + 256 + MAX_CTX * (32 + 512);

/// Natural logarithm through the same libm `log` the old project's `f64::ln` called.
#[inline]
fn ln(x: f64) -> f64 {
    core::intrinsics::log(x)
}

/// Scale `counts` to frequencies summing to exactly `1 << bits`, keeping every present symbol
/// at frequency >= 1. Rounding is repaired greedily by coding cost.
pub fn normalize(counts: &[u64; 256], bits: u32) -> [u16; 256] {
    let m = 1u64 << bits;
    let total: u64 = counts.iter().sum();
    let mut f = [0u64; 256];
    if total == 0 {
        return [0; 256];
    }
    for s in 0..256 {
        if counts[s] > 0 {
            f[s] = ((counts[s] as u128 * m as u128 / total as u128) as u64).max(1);
        }
    }
    let mut sum: u64 = f.iter().sum();
    while sum < m {
        let mut best = 0;
        let mut best_v = f64::MIN;
        for s in 0..256 {
            if counts[s] > 0 {
                let v = counts[s] as f64 * ln((f[s] + 1) as f64 / f[s] as f64);
                if v > best_v {
                    best_v = v;
                    best = s;
                }
            }
        }
        f[best] += 1;
        sum += 1;
    }
    while sum > m {
        let mut best = 0;
        let mut best_v = f64::MAX;
        for s in 0..256 {
            if f[s] > 1 {
                let v = counts[s] as f64 * ln(f[s] as f64 / (f[s] - 1) as f64);
                if v < best_v {
                    best_v = v;
                    best = s;
                }
            }
        }
        f[best] -= 1;
        sum -= 1;
    }
    let mut out = [0u16; 256];
    for s in 0..256 {
        out[s] = f[s] as u16;
    }
    out
}

/// How the high residual (sign|mantissa byte) of every element is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmMode {
    /// Raw, one byte per element (or bitplanes). Decodes fastest.
    Raw,
    /// rANS coded with exponent-selected contexts. Smallest output.
    Coded,
}

/// Alias table size of a tensor; each tier matches a register permute width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// 16 buckets.
    T16,
    /// 32 buckets.
    T32,
    /// 64 buckets.
    T64,
}

impl Tier {
    /// Smallest tier holding `alphabet` symbols, `None` above [`ALIAS_BUCKETS`].
    pub fn fitting(alphabet: usize) -> Option<Tier> {
        match alphabet {
            0..=16 => Some(Tier::T16),
            17..=32 => Some(Tier::T32),
            33..=64 => Some(Tier::T64),
            _ => None,
        }
    }

    /// Number of buckets.
    pub fn buckets(self) -> usize {
        match self {
            Tier::T16 => 16,
            Tier::T32 => 32,
            Tier::T64 => 64,
        }
    }

    /// Slots per bucket is `1 << shift()`.
    pub fn shift(self) -> u32 {
        EXP_BITS - self.buckets().trailing_zeros()
    }

    fn code(self) -> u8 {
        self as u8
    }

    fn from_code(c: u8) -> Option<Tier> {
        [Tier::T16, Tier::T32, Tier::T64].get(c as usize).copied()
    }
}

/// Fixed-capacity LIFO with the same push/pop order as the old `Vec` stacks.
struct Stack<const N: usize> {
    items: [(usize, u32); N],
    len: usize,
}

impl<const N: usize> Stack<N> {
    fn new() -> Self {
        Stack {
            items: [(0, 0); N],
            len: 0,
        }
    }

    fn push(&mut self, v: (usize, u32)) {
        self.items[self.len] = v;
        self.len += 1;
    }

    fn pop(&mut self) -> Option<(usize, u32)> {
        self.len = self.len.checked_sub(1)?;
        Some(self.items[self.len])
    }
}

/// Alias-method layout of the exponent model (Vose construction, as in Giesen's alias rANS).
///
/// Slot `x & (M - 1)` falls in bucket `b = slot >> shift` at offset `j = slot & (bucket - 1)`.
/// Offsets below `div[b]` belong to the primary entry `lo[b]`, the rest to the alias entry
/// `hi[b]`. Entries pack `sym | (freq - 1) << 8 | base << 20` and the rank of the slot within its
/// symbol is `(j + base) mod M`. A full bucket stores `hi[b] == lo[b]` so `div` fits a byte.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpAlias {
    pub tier: Tier,
    pub div: [u8; ALIAS_BUCKETS],
    pub lo: [u32; ALIAS_BUCKETS],
    pub hi: [u32; ALIAS_BUCKETS],
}

impl ExpAlias {
    /// Build the alias layout for `tier`. Returns `None` when the alphabet does not fit.
    pub fn new(freqs: &[u16; 256], tier: Tier) -> Option<ExpAlias> {
        let n = tier.buckets();
        let bucket = 1u32 << tier.shift();
        let mut syms = [0usize; ALIAS_BUCKETS];
        let mut n_syms = 0;
        for (s, &f) in freqs.iter().enumerate() {
            if f > 0 {
                if n_syms == n {
                    return None;
                }
                syms[n_syms] = s;
                n_syms += 1;
            }
        }
        let mut small = Stack::<ALIAS_BUCKETS>::new();
        let mut large = Stack::<ALIAS_BUCKETS>::new();
        for i in 0..n {
            let (s, r) = if i < n_syms {
                (syms[i], freqs[syms[i]] as u32)
            } else {
                (0, 0)
            };
            if r < bucket {
                small.push((s, r))
            } else {
                large.push((s, r))
            }
        }
        let mut buckets = [(0usize, 0u32, 0usize); ALIAS_BUCKETS];
        let mut nb = 0;
        while let Some((s, r)) = small.pop() {
            let (l, lr) = large.pop()?;
            buckets[nb] = (s, r, l);
            nb += 1;
            let lr = lr - (bucket - r);
            if lr < bucket {
                small.push((l, lr))
            } else {
                large.push((l, lr))
            }
        }
        while let Some((l, lr)) = large.pop() {
            if lr != bucket {
                return None;
            }
            buckets[nb] = (l, bucket, l);
            nb += 1;
        }
        let mask = (1u32 << EXP_BITS) - 1;
        let entry = |s: usize, base: u32| {
            s as u32 | ((freqs[s].max(1) as u32 - 1) << 8) | ((base & mask) << 20)
        };
        let mut rank = [0u32; 256];
        let mut t = ExpAlias {
            tier,
            div: [0; ALIAS_BUCKETS],
            lo: [0; ALIAS_BUCKETS],
            hi: [0; ALIAS_BUCKETS],
        };
        for (b, &(p, div, a)) in buckets[..nb].iter().enumerate() {
            t.lo[b] = entry(p, rank[p]);
            if div == bucket {
                t.div[b] = (bucket - 1).min(255) as u8;
                t.hi[b] = t.lo[b];
            } else {
                t.div[b] = div as u8;
                t.hi[b] = entry(a, rank[a].wrapping_sub(div));
            }
            rank[p] += div;
            rank[a] += bucket - div;
        }
        (0..256).all(|s| rank[s] == freqs[s] as u32).then_some(t)
    }

    /// Decode entry of `slot`, in the flat format `sym | (freq - 1) << 8 | rank << 20`.
    pub fn flat(&self, slot: u32) -> u32 {
        let shift = self.tier.shift();
        let b = (slot >> shift) as usize;
        let j = slot & ((1 << shift) - 1);
        let e = if j < self.div[b] as u32 {
            self.lo[b]
        } else {
            self.hi[b]
        };
        let k = (j + (e >> 20)) & ((1 << EXP_BITS) - 1);
        (e & 0xF_FFFF) | (k << 20)
    }
}

/// Exponent-conditioned sign|mantissa model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmModel {
    pub ctx_of: [u8; 256],
    pub n_ctx: usize,
    pub freqs: [[u16; 256]; MAX_CTX],
}

/// Frequency tables shared by every chunk of one tensor. Exponents with zero frequency in
/// `exp_freqs` are coded as the escape symbol `esc`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Codebook {
    pub exp_freqs: [u16; 256],
    pub tier: Tier,
    pub esc: Option<u8>,
    pub sm: Option<SmModel>,
}

/// Present symbols of `counts`, most frequent first, ties by value.
fn order(counts: &[u64; 256]) -> ([usize; 256], usize) {
    let mut order = [0usize; 256];
    let mut n = 0;
    for (e, &c) in counts.iter().enumerate() {
        if c > 0 {
            order[n] = e;
            n += 1;
        }
    }
    order[..n].sort_unstable_by(|&a, &b| counts[b].cmp(&counts[a]).then(a.cmp(&b)));
    (order, n)
}

impl Codebook {
    /// Build from the joint histogram of (exponent, high residual), indexed `exp << 8 | hi`.
    /// Returns `None` when the histogram is empty, or when the tensor needs an escape symbol
    /// but uses all 256 exponent bytes.
    pub fn build(joint: &[u64; 65536], mode: SmMode) -> Option<Codebook> {
        let mut exp_counts = [0u64; 256];
        for (e, c) in exp_counts.iter_mut().enumerate() {
            *c = joint[e << 8..(e + 1) << 8].iter().sum();
        }
        let (ord, n) = order(&exp_counts);
        if n == 0 {
            return None;
        }
        let mut coded = exp_counts;
        let mut esc = None;
        if n > ALIAS_BUCKETS {
            let code = (0..256).find(|&e| exp_counts[e] == 0)?;
            for &e in &ord[ALIAS_BUCKETS - 1..n] {
                coded[code] += coded[e];
                coded[e] = 0;
            }
            esc = Some(code as u8);
        }
        let exp_freqs = normalize(&coded, EXP_BITS);
        let tier = Tier::fitting(n.min(ALIAS_BUCKETS))?;
        ExpAlias::new(&exp_freqs, tier)?;
        let sm = match mode {
            SmMode::Raw => None,
            SmMode::Coded => Some(sm_model(joint, &exp_counts)),
        };
        Some(Codebook {
            exp_freqs,
            tier,
            esc,
            sm,
        })
    }

    /// How the high residual is stored under this codebook.
    pub fn sm_mode(&self) -> SmMode {
        if self.sm.is_some() {
            SmMode::Coded
        } else {
            SmMode::Raw
        }
    }

    /// Serialize sparsely: exponent table, the tag byte (tier in bits 0..2, escape flag in
    /// bit 7) and escape symbol, then for coded sign|mantissa the context of every present
    /// exponent and each context table. Returns the bytes written.
    pub fn write(&self, out: &mut [u8]) -> Result<usize, OpError> {
        let mut w = Writer { out, at: 0 };
        write_freqs(&mut w, &self.exp_freqs)?;
        let flag = if self.esc.is_some() { TAG_ESC } else { 0 };
        w.put(&[self.tier.code() | flag])?;
        if let Some(c) = self.esc {
            w.put(&[c])?;
        }
        let Some(sm) = &self.sm else {
            w.put(&[0])?;
            return Ok(w.at);
        };
        w.put(&[sm.n_ctx as u8])?;
        for e in 0..256 {
            if self.exp_freqs[e] > 0 {
                w.put(&[sm.ctx_of[e]])?;
            }
        }
        for t in &sm.freqs[..sm.n_ctx] {
            write_freqs(&mut w, t)?;
        }
        Ok(w.at)
    }

    /// Parse a codebook, returning it and the bytes consumed. Every table is validated.
    pub fn read(data: &[u8]) -> Result<(Codebook, usize), OpError> {
        let mut r = Reader { data, at: 0 };
        let exp_freqs = read_freqs(&mut r, EXP_BITS)?;
        let tag = r.u8()?;
        let tier = Tier::from_code(tag & !TAG_ESC).ok_or(OpError::Corrupt)?;
        let esc = match tag & TAG_ESC != 0 {
            false => None,
            true => match r.u8()? {
                e if exp_freqs[e as usize] > 0 => Some(e),
                _ => return Err(OpError::Corrupt),
            },
        };
        ExpAlias::new(&exp_freqs, tier).ok_or(OpError::Corrupt)?;
        let n_ctx = r.u8()? as usize;
        if n_ctx == 0 {
            return Ok((
                Codebook {
                    exp_freqs,
                    tier,
                    esc,
                    sm: None,
                },
                r.at,
            ));
        }
        if n_ctx > MAX_CTX {
            return Err(OpError::Corrupt);
        }
        let mut ctx_of = [n_ctx as u8 - 1; 256];
        for e in 0..256 {
            if exp_freqs[e] > 0 {
                let k = r.u8()?;
                if k as usize >= n_ctx {
                    return Err(OpError::Corrupt);
                }
                ctx_of[e] = k;
            }
        }
        let mut freqs = [[0u16; 256]; MAX_CTX];
        for f in freqs.iter_mut().take(n_ctx) {
            *f = read_freqs(&mut r, SM_BITS)?;
        }
        Ok((
            Codebook {
                exp_freqs,
                tier,
                esc,
                sm: Some(SmModel {
                    ctx_of,
                    n_ctx,
                    freqs,
                }),
            },
            r.at,
        ))
    }
}

fn sm_model(joint: &[u64; 65536], exp_counts: &[u64; 256]) -> SmModel {
    let (ord, n) = order(exp_counts);
    let dedicated = ord[..n]
        .iter()
        .take(MAX_CTX - 1)
        .take_while(|&&e| exp_counts[e] >= MIN_CTX_COUNT)
        .count();
    let n_ctx = dedicated + usize::from(n > dedicated);
    let mut ctx_of = [n_ctx as u8 - 1; 256];
    for (rank, &e) in ord[..n].iter().enumerate() {
        ctx_of[e] = rank.min(dedicated) as u8;
    }
    let mut counts = [[0u64; 256]; MAX_CTX];
    for &e in &ord[..n] {
        let c = &mut counts[ctx_of[e] as usize];
        for s in 0..256 {
            c[s] += joint[e << 8 | s];
        }
    }
    let mut freqs = [[0u16; 256]; MAX_CTX];
    for (f, c) in freqs.iter_mut().zip(&counts).take(n_ctx) {
        *f = normalize(c, SM_BITS);
    }
    SmModel {
        ctx_of,
        n_ctx,
        freqs,
    }
}

struct Writer<'a> {
    out: &'a mut [u8],
    at: usize,
}

impl Writer<'_> {
    fn put(&mut self, b: &[u8]) -> Result<(), OpError> {
        let dst = self
            .out
            .get_mut(self.at..self.at + b.len())
            .ok_or(OpError::OutputTooSmall)?;
        dst.copy_from_slice(b);
        self.at += b.len();
        Ok(())
    }
}

struct Reader<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], OpError> {
        let s = self
            .data
            .get(self.at..self.at.checked_add(n).ok_or(OpError::Corrupt)?)
            .ok_or(OpError::Corrupt)?;
        self.at += n;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8, OpError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, OpError> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
}

fn write_freqs(w: &mut Writer, f: &[u16; 256]) -> Result<(), OpError> {
    let mut bitmap = [0u8; 32];
    for s in 0..256 {
        if f[s] > 0 {
            bitmap[s / 8] |= 1 << (s % 8);
        }
    }
    w.put(&bitmap)?;
    for &v in f.iter().filter(|&&v| v > 0) {
        w.put(&v.to_le_bytes())?;
    }
    Ok(())
}

fn read_freqs(r: &mut Reader, bits: u32) -> Result<[u16; 256], OpError> {
    let bitmap = r.take(32)?;
    let mut f = [0u16; 256];
    let mut sum = 0u32;
    for s in 0..256 {
        if bitmap[s / 8] & (1 << (s % 8)) != 0 {
            let v = r.u16()?;
            if v == 0 {
                return Err(OpError::Corrupt);
            }
            f[s] = v;
            sum += v as u32;
        }
    }
    if sum != 1 << bits {
        return Err(OpError::Corrupt);
    }
    Ok(f)
}
