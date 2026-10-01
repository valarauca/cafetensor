//! CRC32C (Castagnoli, reflected polynomial `0x82F63B78`) tables and folding constants.
//!
//! Folding by `D` bits multiplies the low and high qwords of each 128-bit lane by
//! `x^(D+32) mod P` and `x^(D-32) mod P`, bit reflected and shifted left by one. The constants are
//! computed at compile time by [`fold_pair`].

const POLY_NORMAL: u64 = 0x1_1EDC_6F41;
/// The reflected CRC32C polynomial.
pub const POLY_REFLECTED: u32 = 0x82F6_3B78;

const fn build_table() -> [u32; 256] {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                (c >> 1) ^ POLY_REFLECTED
            } else {
                c >> 1
            };
            k += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
}

/// Byte-at-a-time lookup table.
pub static TABLE: [u32; 256] = build_table();

const fn xpow_mod(n: u32) -> u32 {
    let mut r: u64 = 1;
    let mut i = 0;
    while i < n {
        r <<= 1;
        if r & (1 << 32) != 0 {
            r ^= POLY_NORMAL;
        }
        i += 1;
    }
    r as u32
}

/// Folding constants for a distance of `bits`: (low qword multiplier, high qword multiplier).
pub const fn fold_pair(bits: u32) -> (u64, u64) {
    (
        (xpow_mod(bits + 32).reverse_bits() as u64) << 1,
        (xpow_mod(bits - 32).reverse_bits() as u64) << 1,
    )
}

/// Fold four 512-bit accumulators forward by 256 bytes.
pub const K2048: (u64, u64) = fold_pair(2048);
/// Fold one 512-bit accumulator forward by 64 bytes.
pub const K512: (u64, u64) = fold_pair(512);
/// Fold one 128-bit lane forward by 16 bytes.
pub const K128: (u64, u64) = fold_pair(128);

/// One byte through the table.
#[inline(always)]
pub fn update_byte(state: u32, b: u8) -> u32 {
    TABLE[((state ^ b as u32) & 0xFF) as usize] ^ (state >> 8)
}
