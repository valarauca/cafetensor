//! Bit layout of the floating point element formats, `sign | exponent | mantissa`.
//!
//! The exponent is the entropy-coded symbol. The `1 + mant_bits` residual bits split into
//! [`Format::low_bytes`] whole low mantissa bytes and a high part of [`Format::hi_bits`] bits
//! with the sign on top. Identical to the old project's `codec::Format`.

/// Layout of one floating point element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Format {
    pub width: usize,
    pub mant_bits: u32,
}

impl Format {
    pub const BF16: Format = Format {
        width: 2,
        mant_bits: 7,
    };
    pub const F32: Format = Format {
        width: 4,
        mant_bits: 23,
    };
    pub const F16: Format = Format {
        width: 2,
        mant_bits: 10,
    };
    pub const E4M3: Format = Format {
        width: 1,
        mant_bits: 3,
    };
    pub const E5M2: Format = Format {
        width: 1,
        mant_bits: 2,
    };

    /// Exponent field width.
    #[inline]
    pub const fn exp_bits(self) -> u32 {
        8 * self.width as u32 - 1 - self.mant_bits
    }

    /// Whole low mantissa bytes stored as a raw plane.
    #[inline]
    pub const fn low_bytes(self) -> usize {
        (self.mant_bits / 8) as usize
    }

    /// Bits in the high residual, sign included.
    #[inline]
    pub const fn hi_bits(self) -> u32 {
        1 + self.mant_bits - 8 * self.low_bytes() as u32
    }

    /// Split a value into (exponent, high residual, low mantissa bytes).
    #[inline]
    pub const fn split(self, v: u32) -> (u8, u8, u32) {
        let low = 8 * self.low_bytes() as u32;
        let lo = if low == 0 { 0 } else { v & ((1 << low) - 1) };
        let mant = v & ((1 << self.mant_bits) - 1);
        let exp = (v >> self.mant_bits) & ((1 << self.exp_bits()) - 1);
        let sign = (v >> (8 * self.width as u32 - 1)) & 1;
        let hi = (sign << (self.hi_bits() - 1)) | (mant >> low);
        (exp as u8, hi as u8, lo)
    }

    /// Inverse of [`Format::split`].
    #[inline]
    pub const fn join(self, exp: u8, hi: u8, lo: u32) -> u32 {
        let low = 8 * self.low_bytes() as u32;
        let h = self.hi_bits();
        let sign = (hi as u32 >> (h - 1)) & 1;
        let mant_hi = hi as u32 & ((1 << (h - 1)) - 1);
        (sign << (8 * self.width as u32 - 1))
            | ((exp as u32) << self.mant_bits)
            | (mant_hi << low)
            | lo
    }
}

/// Bytes of one bitplane for `elems` elements: one little-endian `u16` per group of 16.
#[inline]
pub const fn plane_len(elems: usize) -> usize {
    2 * elems.div_ceil(16)
}
