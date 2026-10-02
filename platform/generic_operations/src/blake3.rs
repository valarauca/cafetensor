//! BLAKE3 hashing: the compression function over one block or over 16 lanes of independent
//! chunks or parents, and the left-balanced chunk tree. The output equals the reference
//! implementation (`b3sum`) for every input.

use core::simd::prelude::*;

use crate::Kernels;

/// Bytes per chunk, the leaves of the tree.
pub const CHUNK_LEN: usize = 1024;
/// Bytes per compression block.
pub const BLOCK_LEN: usize = 64;
/// Bytes of a hash or chaining value.
pub const OUT_LEN: usize = 32;

/// Chunks hashed into one subtree in a stack buffer.
const BATCH: usize = 256;

const CHUNK_START: u32 = 1;
const CHUNK_END: u32 = 2;
const PARENT: u32 = 4;
const ROOT: u32 = 8;

const IV: [u32; 8] = [
    0x6A09_E667,
    0xBB67_AE85,
    0x3C6E_F372,
    0xA54F_F53A,
    0x510E_527F,
    0x9B05_688C,
    0x1F83_D9AB,
    0x5BE0_CD19,
];

const PERMUTATION: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

/// Message word order of each of the seven rounds.
const SCHEDULE: [[usize; 16]; 7] = {
    let mut s = [[0usize; 16]; 7];
    let mut i = 0;
    while i < 16 {
        s[0][i] = i;
        i += 1;
    }
    let mut r = 1;
    while r < 7 {
        let mut i = 0;
        while i < 16 {
            s[r][i] = s[r - 1][PERMUTATION[i]];
            i += 1;
        }
        r += 1;
    }
    s
};

/// A state word: one `u32`, or one `u32` per lane.
trait Word: Copy + core::ops::BitXor<Output = Self> {
    fn add(self, other: Self) -> Self;
    fn rotr(self, r: u32) -> Self;
}

impl Word for u32 {
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        self.wrapping_add(other)
    }

    #[inline(always)]
    fn rotr(self, r: u32) -> Self {
        self.rotate_right(r)
    }
}

impl<const N: usize> Word for Simd<u32, N> {
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        self + other
    }

    #[inline(always)]
    fn rotr(self, r: u32) -> Self {
        (self >> r) | (self << (32 - r))
    }
}

#[inline(always)]
fn g<W: Word>(v: &mut [W; 16], [a, b, c, d]: [usize; 4], x: W, y: W) {
    v[a] = v[a].add(v[b]).add(x);
    v[d] = (v[d] ^ v[a]).rotr(16);
    v[c] = v[c].add(v[d]);
    v[b] = (v[b] ^ v[c]).rotr(12);
    v[a] = v[a].add(v[b]).add(y);
    v[d] = (v[d] ^ v[a]).rotr(8);
    v[c] = v[c].add(v[d]);
    v[b] = (v[b] ^ v[c]).rotr(7);
}

#[inline(always)]
fn round<W: Word, const R: usize>(v: &mut [W; 16], m: &[W; 16]) {
    let s = &SCHEDULE[R];
    g(v, [0, 4, 8, 12], m[s[0]], m[s[1]]);
    g(v, [1, 5, 9, 13], m[s[2]], m[s[3]]);
    g(v, [2, 6, 10, 14], m[s[4]], m[s[5]]);
    g(v, [3, 7, 11, 15], m[s[6]], m[s[7]]);
    g(v, [0, 5, 10, 15], m[s[8]], m[s[9]]);
    g(v, [1, 6, 11, 12], m[s[10]], m[s[11]]);
    g(v, [2, 7, 8, 13], m[s[12]], m[s[13]]);
    g(v, [3, 4, 9, 14], m[s[14]], m[s[15]]);
}

/// The seven rounds of the compression function.
#[inline(always)]
fn rounds<W: Word>(v: &mut [W; 16], m: &[W; 16]) {
    round::<W, 0>(v, m);
    round::<W, 1>(v, m);
    round::<W, 2>(v, m);
    round::<W, 3>(v, m);
    round::<W, 4>(v, m);
    round::<W, 5>(v, m);
    round::<W, 6>(v, m);
}

/// Compress one block given as words; returns the new chaining value (the first eight output
/// words).
#[inline(always)]
fn compress(cv: &[u32; 8], m: &[u32; 16], counter: u64, len: u32, flags: u32) -> [u32; 8] {
    let mut v = [
        cv[0],
        cv[1],
        cv[2],
        cv[3],
        cv[4],
        cv[5],
        cv[6],
        cv[7],
        IV[0],
        IV[1],
        IV[2],
        IV[3],
        counter as u32,
        (counter >> 32) as u32,
        len,
        flags,
    ];
    rounds(&mut v, m);
    core::array::from_fn(|i| v[i] ^ v[i + 8])
}

#[inline(always)]
fn words(block: &[u8; BLOCK_LEN]) -> [u32; 16] {
    let w = block.as_chunks::<4>().0;
    core::array::from_fn(|i| u32::from_le_bytes(w[i]))
}

/// Chaining value of one chunk of at most [`CHUNK_LEN`] bytes, or the root output when `root`.
#[allow(
    clippy::extra_unused_type_parameters,
    reason = "generic over K so each tier instantiates it with its own flags"
)]
fn chunk_cv<K: Kernels>(data: &[u8], counter: u64, root: bool) -> [u32; 8] {
    let mut cv = IV;
    let blocks = data.len().div_ceil(BLOCK_LEN).max(1);
    for b in 0..blocks {
        let part = &data[(b * BLOCK_LEN).min(data.len())..((b + 1) * BLOCK_LEN).min(data.len())];
        let mut block = [0u8; BLOCK_LEN];
        block[..part.len()].copy_from_slice(part);
        let mut flags = if b == 0 { CHUNK_START } else { 0 };
        if b == blocks - 1 {
            flags |= CHUNK_END | if root { ROOT } else { 0 };
        }
        cv = compress(&cv, &words(&block), counter, part.len() as u32, flags);
    }
    cv
}

/// Parent node of two chaining values.
#[allow(
    clippy::extra_unused_type_parameters,
    reason = "generic over K so each tier instantiates it with its own flags"
)]
fn parent<K: Kernels>(left: &[u32; 8], right: &[u32; 8], flags: u32) -> [u32; 8] {
    let m = core::array::from_fn(|i| if i < 8 { left[i] } else { right[i - 8] });
    compress(&IV, &m, 0, BLOCK_LEN as u32, PARENT | flags)
}

/// Lane groups to message words. `rows[g * N + l]` holds words `g * N..(g + 1) * N` of lane
/// `l`; the result holds word `w` of every lane in entry `w`. Each group of `N` vectors is an
/// `N × N` transpose by rounds of pairwise interleaving.
#[inline(always)]
fn transpose<const N: usize>(rows: [Simd<u32, N>; 16]) -> [Simd<u32, N>; 16] {
    let mut r = rows;
    for base in (0..16).step_by(N) {
        for _ in 0..N.ilog2() {
            let mut next = r;
            for i in 0..N / 2 {
                let (a, b) = r[base + i].interleave(r[base + i + N / 2]);
                next[base + 2 * i] = a;
                next[base + 2 * i + 1] = b;
            }
            r = next;
        }
    }
    r
}

/// Compress one block in each of `N` lanes. `h` holds the lanes' chaining values by word.
#[inline(always)]
fn compress_lanes<const N: usize>(
    h: &mut [Simd<u32, N>; 8],
    m: &[Simd<u32, N>; 16],
    counter: [Simd<u32, N>; 2],
    flags: u32,
) {
    let mut v = [
        h[0],
        h[1],
        h[2],
        h[3],
        h[4],
        h[5],
        h[6],
        h[7],
        Simd::splat(IV[0]),
        Simd::splat(IV[1]),
        Simd::splat(IV[2]),
        Simd::splat(IV[3]),
        counter[0],
        counter[1],
        Simd::splat(BLOCK_LEN as u32),
        Simd::splat(flags),
    ];
    rounds(&mut v, m);
    for (i, x) in h.iter_mut().enumerate() {
        *x = v[i] ^ v[i + 8];
    }
}

/// The lanes of `h` as one chaining value each.
#[inline(always)]
fn lanes<const N: usize>(h: &[Simd<u32, N>; 8], out: &mut [[u32; 8]]) {
    let h = h.map(Simd::to_array);
    for (l, cv) in out.iter_mut().enumerate().take(N) {
        *cv = core::array::from_fn(|i| h[i][l]);
    }
}

/// Chaining values of `N` whole consecutive chunks, the first being chunk `counter`.
#[inline(always)]
fn chunks_lanes<const N: usize>(data: &[[u8; CHUNK_LEN]], counter: u64, out: &mut [[u32; 8]]) {
    let counter = [
        Simd::from_array(core::array::from_fn(|l| (counter + l as u64) as u32)),
        Simd::from_array(core::array::from_fn(|l| {
            ((counter + l as u64) >> 32) as u32
        })),
    ];
    let mut h = IV.map(Simd::splat);
    for b in 0..CHUNK_LEN / BLOCK_LEN {
        let rows: [Simd<u32, N>; 16] = core::array::from_fn(|k| {
            let block = data[k % N].as_chunks::<BLOCK_LEN>().0[b];
            let mut v = u32x16::from_le_bytes(u8x64::from_array(block));
            for _ in 0..k / N * N / 4 {
                v = v.rotate_elements_left::<4>();
            }
            v.resize::<N>(0)
        });
        let mut flags = 0;
        if b == 0 {
            flags |= CHUNK_START;
        }
        if b == CHUNK_LEN / BLOCK_LEN - 1 {
            flags |= CHUNK_END;
        }
        compress_lanes(&mut h, &transpose(rows), counter, flags);
    }
    lanes(&h, out);
}

/// Parents of the `N` pairs `cvs[2l], cvs[2l + 1]`, written to `out`.
#[inline(always)]
fn parents_lanes<const N: usize>(cvs: &[[u32; 8]], out: &mut [[u32; 8]]) {
    let rows: [Simd<u32, N>; 16] = core::array::from_fn(|k| {
        let (l, g) = (k % N, k / N * N);
        Simd::from_array(core::array::from_fn(|j| {
            let w = g + j;
            if w < 8 {
                cvs[2 * l][w]
            } else {
                cvs[2 * l + 1][w - 8]
            }
        }))
    });
    let mut h: [Simd<u32, N>; 8] = IV.map(Simd::splat);
    let zero = [Simd::splat(0); 2];
    compress_lanes(&mut h, &transpose(rows), zero, PARENT);
    lanes(&h, out);
}

/// Chaining value of the subtree over at most [`BATCH`] chunks: every chunk, then each level
/// of parents pairwise with an odd last node carried up, which is the left-balanced tree.
#[inline]
fn batch_cv<K: Kernels, const N: usize>(data: &[u8], counter: u64) -> [u32; 8] {
    let mut cvs = [[0u32; 8]; BATCH];
    let (whole, tail) = data.as_chunks::<CHUNK_LEN>();
    let mut n = 0;
    for group in whole.as_chunks::<N>().0 {
        chunks_lanes::<N>(group.as_slice(), counter + n as u64, &mut cvs[n..n + N]);
        n += N;
    }
    for c in &whole[n..] {
        cvs[n] = chunk_cv::<K>(c, counter + n as u64, false);
        n += 1;
    }
    if !tail.is_empty() || n == 0 {
        cvs[n] = chunk_cv::<K>(tail, counter + n as u64, false);
        n += 1;
    }
    while n > 1 {
        let pairs = n / 2;
        for g in 0..pairs.div_ceil(N) {
            let mut out = [[0u32; 8]; 16];
            parents_lanes::<N>(&cvs[2 * N * g..2 * N * (g + 1)], &mut out);
            let k = (pairs - N * g).min(N);
            cvs[N * g..N * g + k].copy_from_slice(&out[..k]);
        }
        if n % 2 == 1 {
            cvs[pairs] = cvs[n - 1];
        }
        n = pairs + n % 2;
    }
    cvs[0]
}

/// Bytes in the left subtree of an input of `len > CHUNK_LEN` bytes: the largest power of two
/// number of chunks that leaves at least one byte on the right.
const fn left_len(len: usize) -> usize {
    CHUNK_LEN << ((len - 1) / CHUNK_LEN).ilog2()
}

/// Non-root chaining value of the subtree over `data`, whose first chunk is chunk `counter` of
/// the input, hashed `N` chunks or parents at a time.
#[inline]
fn subtree_lanes<K: Kernels, const N: usize>(data: &[u8], counter: u64) -> [u32; 8] {
    if data.len() <= BATCH * CHUNK_LEN {
        return batch_cv::<K, N>(data, counter);
    }
    let left = left_len(data.len());
    let l = subtree_lanes::<K, N>(&data[..left], counter);
    let r = subtree_lanes::<K, N>(&data[left..], counter + (left / CHUNK_LEN) as u64);
    parent::<K>(&l, &r, 0)
}

/// Non-root chaining value of the subtree over `data`, whose first chunk is chunk `counter` of
/// the input, at the tier's [`Kernels::BLAKE3_LANES`].
#[inline]
pub(crate) fn subtree<K: Kernels>(data: &[u8], counter: u64) -> [u32; 8] {
    match K::BLAKE3_LANES {
        4 => subtree_lanes::<K, 4>(data, counter),
        8 => subtree_lanes::<K, 8>(data, counter),
        _ => subtree_lanes::<K, 16>(data, counter),
    }
}

/// Serialize a chaining value or hash.
pub fn to_bytes(cv: &[u32; 8]) -> [u8; OUT_LEN] {
    let mut out = [0u8; OUT_LEN];
    for (o, w) in out.as_chunks_mut::<4>().0.iter_mut().zip(cv) {
        *o = w.to_le_bytes();
    }
    out
}

/// Parse a chaining value.
pub fn from_bytes(b: &[u8; OUT_LEN]) -> [u32; 8] {
    let w = b.as_chunks::<4>().0;
    core::array::from_fn(|i| u32::from_le_bytes(w[i]))
}

/// The BLAKE3 hash of `data`.
#[inline]
pub(crate) fn hash<K: Kernels>(data: &[u8]) -> [u8; OUT_LEN] {
    if data.len() <= CHUNK_LEN {
        return to_bytes(&chunk_cv::<K>(data, 0, true));
    }
    let left = left_len(data.len());
    let l = subtree::<K>(&data[..left], 0);
    let r = subtree::<K>(&data[left..], (left / CHUNK_LEN) as u64);
    to_bytes(&parent::<K>(&l, &r, ROOT))
}

/// Parent of two chaining values, or the hash of the whole input when `root`.
pub(crate) fn merge<K: Kernels>(
    left: &[u8; OUT_LEN],
    right: &[u8; OUT_LEN],
    root: bool,
) -> [u8; OUT_LEN] {
    to_bytes(&parent::<K>(
        &from_bytes(left),
        &from_bytes(right),
        if root { ROOT } else { 0 },
    ))
}
