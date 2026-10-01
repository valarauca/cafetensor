//! Bit layout of the floating point dtypes the codec models, `sign | exponent | mantissa`.
//!
//! Mirrors the old project's `Format`: the exponent is the coded symbol, the residual is split
//! into whole low mantissa bytes and a high part of `hi_bits` bits with the sign on top.

/// Layout of one floating point element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    pub width: usize,
    pub mant_bits: u32,
    /// FP8 E4M3 (`fn` variant): no infinities, NaN only at exponent and mantissa all ones.
    pub e4m3: bool,
}

impl Layout {
    /// Layout of a safetensors float dtype, `None` for every other dtype.
    pub fn of(dtype: &str) -> Option<Layout> {
        let (width, mant_bits, e4m3) = match dtype {
            "BF16" => (2, 7, false),
            "F32" => (4, 23, false),
            "F16" => (2, 10, false),
            "F8_E4M3" => (1, 3, true),
            "F8_E5M2" => (1, 2, false),
            _ => return None,
        };
        Some(Layout {
            width,
            mant_bits,
            e4m3,
        })
    }

    /// Exponent field width.
    pub fn exp_bits(self) -> u32 {
        8 * self.width as u32 - 1 - self.mant_bits
    }

    /// Whole low mantissa bytes stored raw.
    pub fn low_bytes(self) -> usize {
        (self.mant_bits / 8) as usize
    }

    /// Bits in the high residual, sign included.
    pub fn hi_bits(self) -> u32 {
        1 + self.mant_bits - 8 * self.low_bytes() as u32
    }

    /// Split into (exponent, high residual, low mantissa bytes).
    pub fn split(self, v: u32) -> (u8, u8, u32) {
        let low = 8 * self.low_bytes() as u32;
        let lo = if low == 0 { 0 } else { v & ((1 << low) - 1) };
        let mant = v & ((1 << self.mant_bits) - 1);
        let exp = (v >> self.mant_bits) & ((1 << self.exp_bits()) - 1);
        let sign = (v >> (8 * self.width as u32 - 1)) & 1;
        let hi = (sign << (self.hi_bits() - 1)) | (mant >> low);
        (exp as u8, hi as u8, lo)
    }

    /// Inverse of [`Layout::split`].
    pub fn join(self, exp: u8, hi: u8, lo: u32) -> u32 {
        let low = 8 * self.low_bytes() as u32;
        let h = self.hi_bits();
        let sign = (hi as u32 >> (h - 1)) & 1;
        let mant_hi = hi as u32 & ((1 << (h - 1)) - 1);
        (sign << (8 * self.width as u32 - 1))
            | ((exp as u32) << self.mant_bits)
            | (mant_hi << low)
            | lo
    }

    /// Classify a value as zero, subnormal, infinity, NaN or normal.
    pub fn class(self, v: u32) -> Class {
        let mant = v & ((1 << self.mant_bits) - 1);
        let exp = (v >> self.mant_bits) & ((1 << self.exp_bits()) - 1);
        let max = (1 << self.exp_bits()) - 1;
        match (exp, mant) {
            (0, 0) => Class::Zero,
            (0, _) => Class::Subnormal,
            (e, m) if self.e4m3 && e == max && m == (1 << self.mant_bits) - 1 => Class::Nan,
            (e, _) if self.e4m3 || e != max => Class::Normal,
            (_, 0) => Class::Inf,
            _ => Class::Nan,
        }
    }

    /// Read element `i` of little-endian bytes.
    pub fn load(self, bytes: &[u8], i: usize) -> u32 {
        bytes[i * self.width..(i + 1) * self.width]
            .iter()
            .rev()
            .fold(0, |a, &b| (a << 8) | b as u32)
    }
}

/// Value class of a float element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    Zero,
    Subnormal,
    Normal,
    Inf,
    Nan,
}

/// Bytes per element of any safetensors dtype.
pub fn width(dtype: &str) -> Option<usize> {
    let w = match dtype {
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" => 1,
        "BF16" | "F16" | "U16" | "I16" => 2,
        "F32" | "U32" | "I32" => 4,
        "F64" | "U64" | "I64" => 8,
        _ => return None,
    };
    Some(w)
}
