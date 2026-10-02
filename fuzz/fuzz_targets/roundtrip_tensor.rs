//! Compressing any tensor and decompressing it returns the same bytes, for every dtype, both
//! sign|mantissa modes, with and without checksums, and small to default block sizes.
#![no_main]

use cafetensor_lib::SmMode;
use cafetensor_lib::tensor::{DType, Options, compress_bytes, decompress_tensor, decompressed_len};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let [sel, block, rest @ ..] = data else {
        return;
    };
    let dt = DType::from_code(1 + u32::from(*sel) % 15).unwrap_or(DType::Bf16);
    let mode = if sel & 0x80 != 0 {
        SmMode::Coded
    } else {
        SmMode::Raw
    };
    let opts = Options {
        crc: sel & 0x40 != 0,
        ..Options::new(64 << (block % 16), mode)
    };
    let bytes = &rest[..rest.len() / dt.width() * dt.width()];
    let packed = compress_bytes(dt, bytes, &opts).expect("whole elements always compress");
    assert_eq!(decompressed_len(&packed).unwrap() * dt.width(), bytes.len());
    let mut out = vec![0u8; bytes.len()];
    decompress_tensor(&packed, &mut out).expect("a fresh blob decodes");
    assert!(out == bytes, "{dt:?} {opts:?} n={}", bytes.len());
});
