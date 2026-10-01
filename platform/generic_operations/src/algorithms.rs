//! Algorithms, generic over `K: Kernels`.
use crate::{Kernels, OpError, crc};

/// XOR the two halves of `input` into `output`, 64 bytes at a time through `K::xor64`.
#[inline]
pub(crate) fn op_a<K: Kernels>(input: &[u8], output: &mut [u8]) -> Result<usize, OpError> {
    if !input.len().is_multiple_of(2) {
        return Err(OpError::Corrupt);
    }
    let half = input.len() / 2;
    let output = output.get_mut(..half).ok_or(OpError::OutputTooSmall)?;
    let (a, b) = input.split_at(half);
    let (a_blocks, a_tail) = a.as_chunks::<64>();
    let (b_blocks, b_tail) = b.as_chunks::<64>();
    let (out_blocks, out_tail) = output.as_chunks_mut::<64>();
    for ((x, y), o) in a_blocks.iter().zip(b_blocks).zip(out_blocks.iter_mut()) {
        K::xor64(x, y, o);
    }
    for ((x, y), o) in a_tail.iter().zip(b_tail).zip(out_tail.iter_mut()) {
        *o = x ^ y;
    }
    Ok(half)
}

/// CRC32C over `data`: whole 256-byte blocks through `K::crc32c_blocks`, then 8-byte words
/// through `K::crc32c_u64`, then single bytes through the table.
#[inline]
pub(crate) fn crc32c_update<K: Kernels>(state: u32, data: &[u8]) -> u32 {
    let (blocks, rest) = data.as_chunks::<256>();
    let state = if blocks.is_empty() {
        state
    } else {
        K::crc32c_blocks(state, blocks)
    };
    let (words, bytes) = rest.as_chunks::<8>();
    let state = words
        .iter()
        .fold(state, |c, w| K::crc32c_u64(c, u64::from_le_bytes(*w)));
    bytes.iter().fold(state, |c, &b| crc::update_byte(c, b))
}
