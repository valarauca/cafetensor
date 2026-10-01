//! Algorithms, generic over `K: Kernels`.
use crate::{Kernels, OpError};

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
