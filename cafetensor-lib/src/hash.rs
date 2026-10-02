//! BLAKE3 over the selected tier, in the container's `blake3-<hex>` notation.
//!
//! [`Hasher`] keeps the reference implementation's stack of subtree chaining values. Large
//! inputs are cut into the biggest aligned power-of-two subtrees that leave at least one byte
//! for later, and each subtree is hashed in parallel halves on the rayon pool through
//! `Operations::blake3_subtree` and `Operations::blake3_parent`. The last chunk stays buffered
//! until more input arrives, because it may turn out to be the root.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use general_backend::blake3::{CHUNK_LEN, OUT_LEN};

use crate::{Error, ops};

/// Prefix of a hash in the container's notation.
pub const HASH_PREFIX: &str = "blake3-";
/// Subtrees at most this long are hashed by one task.
const TASK_LEN: usize = 1 << 17;
/// Bytes read per `read` call by [`b3sum`].
const READ_LEN: usize = 1 << 26;

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

/// Chaining value of a complete power-of-two subtree whose first chunk is chunk `counter`.
fn subtree(data: &[u8], counter: u64) -> [u8; OUT_LEN] {
    if data.len() <= TASK_LEN {
        return ops().blake3_subtree(data, counter);
    }
    let half = data.len() / 2;
    let (l, r) = rayon::join(
        || subtree(&data[..half], counter),
        || subtree(&data[half..], counter + (half / CHUNK_LEN) as u64),
    );
    ops().blake3_parent(&l, &r, false)
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
        while input.len() > CHUNK_LEN {
            let mut chunks = 1u64 << ((input.len() - 1) / CHUNK_LEN).ilog2();
            while !self.chunks.is_multiple_of(chunks) {
                chunks /= 2;
            }
            let len = chunks as usize * CHUNK_LEN;
            let cv = subtree(&input[..len], self.chunks);
            self.push(cv, chunks);
            input = &input[len..];
        }
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

/// BLAKE3 of a file in the container's notation, read sequentially and hashed in parallel.
pub fn b3sum(path: &Path) -> Result<String, Error> {
    let mut f = File::open(path)?;
    let mut h = Hasher::new();
    let mut buf = vec![0u8; READ_LEN];
    loop {
        let mut filled = 0;
        while filled < buf.len() {
            match f.read(&mut buf[filled..])? {
                0 => break,
                k => filled += k,
            }
        }
        h.update(&buf[..filled]);
        if filled < buf.len() {
            break;
        }
    }
    Ok(hash_string(&h.finalize()))
}
