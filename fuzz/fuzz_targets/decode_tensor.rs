//! The tensor decoder on hostile input must return an error, never panic. Even first bytes
//! decode the rest as a blob; odd first bytes build a valid blob from the input and then flip
//! the bits the input's tail names, so mutations reach past the header checks.
#![no_main]

use cafetensor_lib::SmMode;
use cafetensor_lib::tensor::{
    DType, Options, compress_bytes, decompress_tensor, decompressed_len, dtype,
};
use libfuzzer_sys::fuzz_target;

fn decode(blob: &[u8]) {
    let (Ok(n), Ok(dt)) = (decompressed_len(blob), dtype(blob)) else {
        return;
    };
    let Some(len) = n.checked_mul(dt.width()).filter(|&l| l <= 1 << 24) else {
        return;
    };
    let mut out = vec![0u8; len];
    let _ = decompress_tensor(blob, &mut out);
}

fuzz_target!(|data: &[u8]| {
    let Some((&sel, rest)) = data.split_first() else {
        return;
    };
    if sel & 1 == 0 {
        return decode(rest);
    }
    let dt = DType::from_code(1 + u32::from(sel >> 1) % 5).unwrap_or(DType::Bf16);
    let split = rest.len() * 3 / 4;
    let (values, flips) = rest.split_at(split);
    let values = &values[..values.len() / dt.width() * dt.width()];
    let mode = if sel & 0x80 != 0 {
        SmMode::Coded
    } else {
        SmMode::Raw
    };
    let opts = Options {
        crc: sel & 0x40 != 0,
        ..Options::new(64 << (sel >> 4 & 3), mode)
    };
    let Ok(mut blob) = compress_bytes(dt, values, &opts) else {
        return;
    };
    for [lo, hi, x] in flips.as_chunks::<3>().0 {
        let at = usize::from(u16::from_le_bytes([*lo, *hi])) % blob.len();
        blob[at] ^= x;
    }
    decode(&blob);
});
