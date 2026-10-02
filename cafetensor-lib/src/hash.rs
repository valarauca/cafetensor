//! BLAKE3 over the selected tier, in the container's `blake3-<hex>` notation.
//!
//! [`Hasher`] keeps the reference implementation's stack of subtree chaining values. Large
//! inputs are cut into the biggest aligned power-of-two subtrees that leave at least one byte
//! for later. Every subtree of one update is hashed at once on the rayon pool, in 256 KiB
//! leaves through `Operations::blake3_subtree`, and its parent levels are merged through
//! `Operations::blake3_parent`. The last chunk stays buffered until more input arrives,
//! because it may turn out to be the root.

use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::mpsc::sync_channel;

use general_backend::blake3::{CHUNK_LEN, OUT_LEN};
use rayon::prelude::*;

use crate::{Error, HugeBuf, ops};

/// Prefix of a hash in the container's notation.
pub const HASH_PREFIX: &str = "blake3-";
/// Subtree hashed by one task: big enough that a call runs near full speed, small enough that
/// a few-MiB tensor still spreads over the pool.
const LEAF: usize = 1 << 18;
/// Bytes [`b3sum`] reads per buffer.
const READ_LEN: usize = 1 << 24;

/// Incremental BLAKE3 hasher.
#[derive(Clone)]
pub struct Hasher {
    buf: [u8; CHUNK_LEN],
    buf_len: usize,
    chunks: u64,
    stack: Vec<[u8; OUT_LEN]>,
}

impl Default for Hasher {
    fn default() -> Self {
        Hasher {
            buf: [0; CHUNK_LEN],
            buf_len: 0,
            chunks: 0,
            stack: Vec::new(),
        }
    }
}

/// Chaining value of a complete power-of-two subtree whose first chunk is chunk `counter`:
/// leaves on the rayon pool, then the parent levels in order.
fn subtree(data: &[u8], counter: u64) -> [u8; OUT_LEN] {
    if data.len() <= LEAF {
        return ops().blake3_subtree(data, counter);
    }
    let mut cvs: Vec<[u8; OUT_LEN]> = data
        .par_chunks(LEAF)
        .enumerate()
        .map(|(i, leaf)| ops().blake3_subtree(leaf, counter + (i * LEAF / CHUNK_LEN) as u64))
        .collect();
    while cvs.len() > 1 {
        cvs = cvs
            .as_chunks::<2>()
            .0
            .iter()
            .map(|[l, r]| ops().blake3_parent(l, r, false))
            .collect();
    }
    cvs[0]
}

impl Hasher {
    /// A hasher over no input.
    pub fn new() -> Self {
        Self::default()
    }

    /// Push the chaining value of a complete subtree of `chunks` chunks (a power of two) that
    /// ends where more input follows, merging every subtree it completes.
    fn push(&mut self, mut cv: [u8; OUT_LEN], chunks: u64) {
        self.chunks += chunks;
        let mut total = self.chunks / chunks;
        while total & 1 == 0 {
            let left = self
                .stack
                .pop()
                .unwrap_or_else(|| unreachable!("a sibling subtree"));
            cv = ops().blake3_parent(&left, &cv, false);
            total >>= 1;
        }
        self.stack.push(cv);
    }

    /// Add `input` to the hashed stream.
    pub fn update(&mut self, mut input: &[u8]) -> &mut Self {
        if self.buf_len > 0 {
            let take = (CHUNK_LEN - self.buf_len).min(input.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&input[..take]);
            self.buf_len += take;
            input = &input[take..];
            if input.is_empty() {
                return self;
            }
            let cv = ops().blake3_subtree(&self.buf, self.chunks);
            self.buf_len = 0;
            self.push(cv, 1);
        }
        let mut pieces = Vec::new();
        let (mut at, mut counter) = (0, self.chunks);
        while input.len() - at > CHUNK_LEN {
            let mut chunks = 1u64 << ((input.len() - at - 1) / CHUNK_LEN).ilog2();
            while !counter.is_multiple_of(chunks) {
                chunks /= 2;
            }
            let len = chunks as usize * CHUNK_LEN;
            pieces.push((at, len, counter));
            at += len;
            counter += chunks;
        }
        let cvs: Vec<[u8; OUT_LEN]> = pieces
            .par_iter()
            .map(|&(at, len, counter)| subtree(&input[at..at + len], counter))
            .collect();
        for (&(_, len, _), cv) in pieces.iter().zip(cvs) {
            self.push(cv, (len / CHUNK_LEN) as u64);
        }
        input = &input[at..];
        self.buf[..input.len()].copy_from_slice(input);
        self.buf_len = input.len();
        self
    }

    /// The hash of everything added so far.
    pub fn finalize(&self) -> [u8; OUT_LEN] {
        let last = &self.buf[..self.buf_len];
        let Some((root, rest)) = self.stack.split_first() else {
            return ops().blake3_hash(last);
        };
        let mut cv = ops().blake3_subtree(last, self.chunks);
        for left in rest.iter().rev() {
            cv = ops().blake3_parent(left, &cv, false);
        }
        ops().blake3_parent(root, &cv, true)
    }
}

/// A hash in the container's `blake3-<hex>` notation.
pub fn hash_string(hash: &[u8; OUT_LEN]) -> String {
    let hex: String = hash.iter().map(|b| format!("{b:02x}")).collect();
    format!("{HASH_PREFIX}{hex}")
}

/// The BLAKE3 hash of `data`.
pub fn hash(data: &[u8]) -> [u8; OUT_LEN] {
    Hasher::new().update(data).finalize()
}

/// BLAKE3 of a file in the container's notation. One thread reads the file sequentially into
/// two alternating huge-page buffers while the rayon pool hashes the other.
pub fn b3sum(path: &Path) -> Result<String, Error> {
    let mut f = File::open(path)?;
    let (full_tx, full_rx) = sync_channel::<std::io::Result<(HugeBuf, usize)>>(1);
    let (free_tx, free_rx) = sync_channel::<HugeBuf>(2);
    for _ in 0..2 {
        let mut buf = HugeBuf::new();
        buf.resize(READ_LEN)?;
        let _ = free_tx.send(buf);
    }
    std::thread::scope(|s| {
        s.spawn(move || {
            for mut buf in free_rx {
                let mut filled = 0;
                let result = buf.resize(READ_LEN).and_then(|dst| {
                    while filled < dst.len() {
                        match f.read(&mut dst[filled..])? {
                            0 => break,
                            k => filled += k,
                        }
                    }
                    Ok(())
                });
                if let Err(e) = result {
                    let _ = full_tx.send(Err(e));
                    return;
                }
                let last = filled < READ_LEN;
                if full_tx.send(Ok((buf, filled))).is_err() || last {
                    return;
                }
            }
        });
        let mut h = Hasher::new();
        for msg in full_rx {
            let (buf, filled) = msg?;
            h.update(&buf.as_slice()[..filled]);
            if filled < READ_LEN {
                break;
            }
            let _ = free_tx.send(buf);
        }
        drop(free_tx);
        Ok(hash_string(&h.finalize()))
    })
}
