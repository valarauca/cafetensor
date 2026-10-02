//! Every tier this host can run produces the same bytes as `portable` for every operation on
//! the same input: CRC32C, plane split, bitplanes, histogram, BLAKE3, and rANS encode and
//! decode of one chunk.
#![no_main]

use general_backend::codebook::{Codebook, SmMode};
use general_backend::decode::{ChunkRef, DecTables};
use general_backend::rans::{EncTables, WAYS, encode_bound, encode_scratch_len};
use general_backend::{Format, Operations, available, plane_len};
use libfuzzer_sys::fuzz_target;

const FORMATS: [Format; 5] = [
    Format::BF16,
    Format::F32,
    Format::F16,
    Format::E4M3,
    Format::E5M2,
];

fn run(t: &dyn Operations, fmt: Format, mode: SmMode, bytes: &[u8]) -> Vec<u8> {
    let n = bytes.len() / fmt.width;
    let mut out = t.crc32c_update(!0, bytes).to_le_bytes().to_vec();
    out.extend_from_slice(&t.blake3_hash(bytes));
    let (mut e, mut h, mut l) = (vec![0; n], vec![0; n], vec![0; n * fmt.low_bytes()]);
    t.split_planes(fmt, bytes, &mut e, &mut h, &mut l).unwrap();
    out.extend(e.iter().chain(&h).chain(&l));
    let mut bits = vec![0u8; fmt.hi_bits() as usize * plane_len(n)];
    t.pack_bitplanes(&h, fmt.hi_bits(), &mut bits).unwrap();
    out.extend_from_slice(&bits);
    let mut small = vec![0u32; 65536];
    t.histogram(&e, &h, small.as_mut_slice().try_into().unwrap())
        .unwrap();
    let joint: Vec<u64> = small.iter().map(|&c| c.into()).collect();
    let mode = if fmt.hi_bits() == 8 {
        mode
    } else {
        SmMode::Raw
    };
    let Some(cb) = Codebook::build(joint.as_slice().try_into().unwrap(), mode) else {
        return out;
    };
    let (enc, dec) = (
        Box::new(EncTables::new(&cb).unwrap()),
        Box::new(DecTables::new(&cb).unwrap()),
    );
    let mut scratch = vec![0u16; encode_scratch_len(n)];
    let mut streams = vec![0u8; encode_bound(n, enc.coded)];
    let info = t
        .encode_chunk(&enc, &e, &h, &mut scratch, &mut streams)
        .unwrap();
    out.extend_from_slice(&streams[..info.len]);
    let mut parts = [[&[][..]; 3]; WAYS];
    let mut at = 0;
    for (w, p) in parts.iter_mut().enumerate() {
        for (k, q) in p.iter_mut().enumerate() {
            let len = info.index[w][k] as usize * if k < 2 { 2 } else { 1 };
            *q = &streams[at..at + len];
            at += len;
        }
    }
    let (hi, bits): (&[u8], &[u8]) = match (enc.coded, fmt.hi_bits()) {
        (true, _) => (&[], &[]),
        (false, 8) => (&h, &[]),
        (false, _) => (&[], &bits),
    };
    let chunk = ChunkRef {
        elems: n,
        exp: parts.map(|p| p[0]),
        sm: parts.map(|p| p[1]),
        esc: parts.map(|p| p[2]),
        hi,
        bits,
        lo: &l,
    };
    let mut back = vec![0u8; bytes.len()];
    t.decode_chunk(&dec, fmt, chunk, &mut back).unwrap();
    assert!(back == bytes, "{} did not round-trip", t.tier_name());
    out
}

fuzz_target!(|data: &[u8]| {
    let Some((&sel, rest)) = data.split_first() else {
        return;
    };
    let fmt = FORMATS[usize::from(sel) % FORMATS.len()];
    let mode = if sel & 0x80 != 0 {
        SmMode::Coded
    } else {
        SmMode::Raw
    };
    let bytes = &rest[..rest.len() / fmt.width * fmt.width];
    let tiers = available();
    let Some((portable, others)) = tiers.split_last() else {
        return;
    };
    let want = run(*portable, fmt, mode, bytes);
    for t in others {
        assert!(
            run(*t, fmt, mode, bytes) == want,
            "{} differs from portable",
            t.tier_name()
        );
    }
});
