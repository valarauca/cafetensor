//! Per-tensor blob format, byte-identical to the `tensor-compressor` project's.
//!
//! ```text
//!   [u8;4] TENSOR_MAGIC
//!   u32    tag                    dtype code in bits 0..30, bit 30 = big endian, bit 31 = passthrough
//!   u8     mode                   0 = raw, 1 = rANS
//!   u8     flags                  bit 0 = CRC32C present
//!   u64    n                      elements
//!   -- mode 0 --
//!   values[n]                     little-endian, 1, 2 or 4 bytes each
//!   u32    crc                    CRC32C of values, if flagged
//!   -- mode 1 --
//!   u32    chunk_elems            multiple of 32
//!   codebook                      shared by every chunk, n_ctx = 0 means raw high residual
//!   u32    n_chunks
//!   (u32 exp_words, u32 sm_words, u32 esc_bytes)[n_chunks][WAYS]
//!   u32    head_crc               CRC32C of every byte above, if flagged
//!   u32    chunk_crc[n_chunks]    CRC32C of each chunk's payload, if flagged
//!   per chunk:
//!     per way: u16 exp stream[exp_words], u16 sm stream[sm_words], u8 esc[esc_bytes]
//!     full-byte raw high residual: u8 hi[chunk length]
//!     narrower high residual: hi_bits bitplanes of u16[ceil(chunk length / 16)]
//!     u8     low[chunk length * low_bytes]
//! ```
//!
//! Dtype codes are listed on [`DType`]. Only the floating point codes 1..=5 are entropy coded,
//! every other dtype is a passthrough tensor stored in mode 0 with bit 31 of the tag set.
//!
//! Chunks are independent rANS substreams, so any chunk decodes without touching the others,
//! and both directions run chunks in parallel on the rayon pool. Checksums cover the
//! compressed bytes, so they are verified while the input is still in cache.

use rayon::prelude::*;

use general_backend::codebook::{CODEBOOK_MAX_BYTES, Codebook, SmMode};
use general_backend::decode::{ChunkRef, DecTables};
use general_backend::rans::{EncTables, WAYS, encode_bound, encode_scratch_len};
use general_backend::{Format, plane_len};

use crate::bytes::{Cursor, put_u8, put_u32, put_u64};
use crate::{Error, ops};

const TENSOR_MAGIC: [u8; 4] = *b"CAF1";
const MODE_RAW: u8 = 0;
const MODE_RANS: u8 = 1;
const INDEX_ENTRY: usize = 12;
const FLAG_CRC: u8 = 1;
/// Elements per plane split task.
const SPLIT: usize = 1 << 16;
/// Elements per histogram task.
const HIST: usize = 1 << 20;
/// Tag bit marking a tensor stored without entropy coding because its dtype is not supported.
pub const TAG_PASSTHROUGH: u32 = 1 << 31;
/// Tag bit marking big-endian element bytes. Never written, rejected on read.
pub const TAG_BIG_ENDIAN: u32 = 1 << 30;
/// Smallest block size, as a power of two in bytes (2 MiB).
pub const MIN_BLOCK_LOG2: u32 = 21;
/// Largest block size, as a power of two in bytes (256 MiB).
pub const MAX_BLOCK_LOG2: u32 = 28;
/// Default uncompressed bytes per independently decodable chunk (2 MiB).
pub const DEFAULT_BLOCK: usize = 1 << MIN_BLOCK_LOG2;

/// Encoder settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Options {
    /// Uncompressed bytes per chunk. The element count is rounded up to a multiple of 32.
    pub block_bytes: usize,
    /// How sign|mantissa is stored.
    pub mode: SmMode,
    /// Store CRC32C checksums of the header and every chunk.
    pub crc: bool,
}

impl Options {
    /// Options with checksums enabled.
    pub fn new(block_bytes: usize, mode: SmMode) -> Self {
        Options {
            block_bytes,
            mode,
            crc: true,
        }
    }

    fn chunk_elems(&self, width: usize) -> usize {
        (self.block_bytes / width).max(1).next_multiple_of(32)
    }
}

impl Default for Options {
    fn default() -> Self {
        Options::new(DEFAULT_BLOCK, SmMode::Raw)
    }
}

/// Element type of a tensor, with its stable tag code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Bf16 = 1,
    F32 = 2,
    F16 = 3,
    F8E4M3 = 4,
    F8E5M2 = 5,
    F64 = 6,
    Bool = 7,
    U8 = 8,
    I8 = 9,
    U16 = 10,
    I16 = 11,
    U32 = 12,
    I32 = 13,
    U64 = 14,
    I64 = 15,
}

const ALL_DTYPES: [DType; 15] = [
    DType::Bf16,
    DType::F32,
    DType::F16,
    DType::F8E4M3,
    DType::F8E5M2,
    DType::F64,
    DType::Bool,
    DType::U8,
    DType::I8,
    DType::U16,
    DType::I16,
    DType::U32,
    DType::I32,
    DType::U64,
    DType::I64,
];

impl DType {
    /// Bit layout of the element, `None` for dtypes that are only passed through.
    pub fn format(self) -> Option<Format> {
        match self {
            DType::Bf16 => Some(Format::BF16),
            DType::F32 => Some(Format::F32),
            DType::F16 => Some(Format::F16),
            DType::F8E4M3 => Some(Format::E4M3),
            DType::F8E5M2 => Some(Format::E5M2),
            _ => None,
        }
    }

    /// Bytes per element.
    pub fn width(self) -> usize {
        match self {
            DType::Bool | DType::U8 | DType::I8 | DType::F8E4M3 | DType::F8E5M2 => 1,
            DType::Bf16 | DType::F16 | DType::U16 | DType::I16 => 2,
            DType::F32 | DType::U32 | DType::I32 => 4,
            DType::F64 | DType::U64 | DType::I64 => 8,
        }
    }

    /// The safetensors dtype string.
    pub fn name(self) -> &'static str {
        match self {
            DType::Bf16 => "BF16",
            DType::F32 => "F32",
            DType::F16 => "F16",
            DType::F8E4M3 => "F8_E4M3",
            DType::F8E5M2 => "F8_E5M2",
            DType::F64 => "F64",
            DType::Bool => "BOOL",
            DType::U8 => "U8",
            DType::I8 => "I8",
            DType::U16 => "U16",
            DType::I16 => "I16",
            DType::U32 => "U32",
            DType::I32 => "I32",
            DType::U64 => "U64",
            DType::I64 => "I64",
        }
    }

    /// Parse a safetensors dtype string.
    pub fn from_name(s: &str) -> Option<DType> {
        ALL_DTYPES.into_iter().find(|d| d.name() == s)
    }

    /// Dtype from the low 30 bits of a tag.
    pub fn from_code(code: u32) -> Option<DType> {
        ALL_DTYPES.into_iter().find(|&d| d as u32 == code)
    }
}

/// Standard CRC32C of `data` on the selected tier.
pub fn crc32c(data: &[u8]) -> u32 {
    !ops().crc32c_update(!0, data)
}

fn put_raw(out: &mut Vec<u8>, n: usize, bytes: &[u8], crc: bool) {
    put_u8(out, MODE_RAW);
    put_u8(out, if crc { FLAG_CRC } else { 0 });
    put_u64(out, n as u64);
    out.extend_from_slice(bytes);
    if crc {
        put_u32(out, crc32c(bytes));
    }
}

fn zeroed<T: Copy + Default, const N: usize>() -> Box<[T; N]> {
    vec![T::default(); N]
        .into_boxed_slice()
        .try_into()
        .unwrap_or_else(|_| unreachable!("the vector has exactly N elements"))
}

/// Joint (exponent, high residual) counts, kept in `u32` per task and folded into `u64` before
/// a bin could overflow.
struct Joint {
    small: Box<[u32; 65536]>,
    big: Box<[u64; 65536]>,
    pending: usize,
}

impl Joint {
    fn new() -> Self {
        Joint {
            small: zeroed(),
            big: zeroed(),
            pending: 0,
        }
    }

    fn flush(&mut self) {
        for (b, s) in self.big.iter_mut().zip(self.small.iter_mut()) {
            *b += *s as u64;
            *s = 0;
        }
        self.pending = 0;
    }

    fn add(mut self, exps: &[u8], his: &[u8]) -> Result<Self, Error> {
        if self.pending + exps.len() > u32::MAX as usize {
            self.flush();
        }
        ops().histogram(exps, his, &mut self.small)?;
        self.pending += exps.len();
        Ok(self)
    }

    fn total(mut self) -> Box<[u64; 65536]> {
        self.flush();
        self.big
    }

    fn merge(mut self, mut other: Joint) -> Joint {
        self.flush();
        other.flush();
        for (a, b) in self.big.iter_mut().zip(other.big.iter()) {
            *a += b;
        }
        self
    }
}

/// Exponent, high residual and low byte planes of a tensor.
type Planes = (Vec<u8>, Vec<u8>, Vec<u8>);

/// Split elements into exponent, high residual and low byte planes, in parallel.
fn split(fmt: Format, bytes: &[u8]) -> Result<Planes, Error> {
    let n = bytes.len() / fmt.width;
    let low = fmt.low_bytes();
    let (mut exps, mut his, mut los) = (vec![0u8; n], vec![0u8; n], vec![0u8; n * low]);
    let src = bytes.par_chunks(SPLIT * fmt.width);
    let (e, h) = (exps.par_chunks_mut(SPLIT), his.par_chunks_mut(SPLIT));
    if low == 0 {
        (src, e, h)
            .into_par_iter()
            .try_for_each(|(b, e, h)| ops().split_planes(fmt, b, e, h, &mut []))?;
    } else {
        (src, e, h, los.par_chunks_mut(SPLIT * low))
            .into_par_iter()
            .try_for_each(|(b, e, h, l)| ops().split_planes(fmt, b, e, h, l))?;
    }
    Ok((exps, his, los))
}

/// One encoded chunk: its payload, index entries and checksum.
struct EncodedChunk {
    payload: Vec<u8>,
    index: [[u32; 3]; WAYS],
    crc: u32,
}

fn encode_one(
    fmt: Format,
    enc: &EncTables,
    exps: &[u8],
    his: &[u8],
    los: &[u8],
    crc: bool,
) -> Result<EncodedChunk, Error> {
    let len = exps.len();
    let hi_len = match (enc.coded, fmt.hi_bits()) {
        (true, _) => 0,
        (false, 8) => len,
        (false, b) => b as usize * plane_len(len),
    };
    let mut scratch = vec![0u16; encode_scratch_len(len)];
    let mut payload = vec![0u8; encode_bound(len, enc.coded) + hi_len + los.len()];
    let info = ops().encode_chunk(enc, exps, his, &mut scratch, &mut payload)?;
    let mut at = info.len;
    match (enc.coded, fmt.hi_bits()) {
        (true, _) => {}
        (false, 8) => payload[at..at + len].copy_from_slice(his),
        (false, b) => ops().pack_bitplanes(his, b, &mut payload[at..at + hi_len])?,
    }
    at += hi_len;
    payload[at..at + los.len()].copy_from_slice(los);
    payload.truncate(at + los.len());
    let crc = if crc { crc32c(&payload) } else { 0 };
    Ok(EncodedChunk {
        payload,
        index: info.index,
        crc,
    })
}

/// Compress a tensor given as little-endian element bytes. Floating point dtypes are entropy
/// coded, every other dtype is passed through. Coded sign|mantissa (`SmMode::Coded`) only
/// applies to formats with a full-byte high residual (BF16, F32). Falls back to raw storage
/// when coding does not pay for itself. Fails if `bytes` is not a whole number of elements.
pub fn compress_bytes(dtype: DType, bytes: &[u8], opts: &Options) -> Result<Vec<u8>, Error> {
    let width = dtype.width();
    if !bytes.len().is_multiple_of(width) {
        return Err(Error::Layout(format!("partial {} element", dtype.name())));
    }
    let n = bytes.len() / width;
    let mut out = Vec::new();
    out.extend_from_slice(&TENSOR_MAGIC);
    let Some(fmt) = dtype.format() else {
        put_u32(&mut out, dtype as u32 | TAG_PASSTHROUGH);
        put_raw(&mut out, n, bytes, opts.crc);
        return Ok(out);
    };
    put_u32(&mut out, dtype as u32);
    let raw_at = out.len();
    if n == 0 {
        put_raw(&mut out, n, bytes, opts.crc);
        return Ok(out);
    }
    let chunk = opts.chunk_elems(width);
    let mode = if fmt.hi_bits() == 8 {
        opts.mode
    } else {
        SmMode::Raw
    };
    let (exps, his, los) = split(fmt, bytes)?;
    let joint = exps
        .par_chunks(HIST)
        .zip(his.par_chunks(HIST))
        .try_fold(Joint::new, |j, (e, h)| j.add(e, h))
        .try_reduce(Joint::new, |a, b| Ok(a.merge(b)))?
        .total();
    let model = Codebook::build(&joint, mode)
        .and_then(|cb| EncTables::new(&cb).map(|enc| (cb, Box::new(enc))));
    let Some((cb, enc)) = model else {
        put_raw(&mut out, n, bytes, opts.crc);
        return Ok(out);
    };
    put_u8(&mut out, MODE_RANS);
    put_u8(&mut out, if opts.crc { FLAG_CRC } else { 0 });
    put_u64(&mut out, n as u64);
    put_u32(&mut out, chunk as u32);
    let mut table = [0u8; CODEBOOK_MAX_BYTES];
    let used = cb.write(&mut table)?;
    out.extend_from_slice(&table[..used]);
    let n_chunks = n.div_ceil(chunk);
    put_u32(&mut out, n_chunks as u32);

    let low = fmt.low_bytes();
    let chunks: Vec<EncodedChunk> = (0..n_chunks)
        .into_par_iter()
        .map(|k| {
            let (a, b) = (k * chunk, ((k + 1) * chunk).min(n));
            encode_one(
                fmt,
                &enc,
                &exps[a..b],
                &his[a..b],
                &los[a * low..b * low],
                opts.crc,
            )
        })
        .collect::<Result<_, _>>()?;

    for c in &chunks {
        for e in c.index.iter().flatten() {
            put_u32(&mut out, *e);
        }
    }
    if opts.crc {
        let head = crc32c(&out);
        put_u32(&mut out, head);
        for c in &chunks {
            put_u32(&mut out, c.crc);
        }
    }
    out.reserve(chunks.iter().map(|c| c.payload.len()).sum());
    for c in &chunks {
        out.extend_from_slice(&c.payload);
    }
    if out.len() >= raw_at + 10 + bytes.len() + 4 * usize::from(opts.crc) {
        out.truncate(raw_at);
        put_raw(&mut out, n, bytes, opts.crc);
    }
    Ok(out)
}

struct Head<'a> {
    c: Cursor<'a>,
    dtype: DType,
    passthrough: bool,
    mode: u8,
    crc: bool,
    n: usize,
}

fn read_head(data: &[u8]) -> Result<Head<'_>, Error> {
    let mut c = Cursor::new(data);
    if c.take(4)? != TENSOR_MAGIC {
        return Err(Error::BadMagic);
    }
    let tag = c.u32()?;
    if tag & TAG_BIG_ENDIAN != 0 {
        return Err(Error::Corrupt);
    }
    let dtype =
        DType::from_code(tag & !(TAG_PASSTHROUGH | TAG_BIG_ENDIAN)).ok_or(Error::Corrupt)?;
    let passthrough = tag & TAG_PASSTHROUGH != 0;
    let mode = c.u8()?;
    if (passthrough && mode != MODE_RAW) || (!passthrough && dtype.format().is_none()) {
        return Err(Error::Corrupt);
    }
    let crc = match c.u8()? {
        0 => false,
        FLAG_CRC => true,
        _ => return Err(Error::Corrupt),
    };
    let n = usize::try_from(c.u64()?).map_err(|_| Error::Corrupt)?;
    Ok(Head {
        c,
        dtype,
        passthrough,
        mode,
        crc,
        n,
    })
}

/// Number of elements a tensor blob decodes to.
pub fn decompressed_len(data: &[u8]) -> Result<usize, Error> {
    Ok(read_head(data)?.n)
}

/// Element type of a tensor blob.
pub fn dtype(data: &[u8]) -> Result<DType, Error> {
    Ok(read_head(data)?.dtype)
}

/// Whether the tensor blob carries CRC32C checksums.
pub fn has_crc(data: &[u8]) -> Result<bool, Error> {
    Ok(read_head(data)?.crc)
}

/// Whether the tensor blob stores its payload uncompressed, passthrough included.
pub fn is_raw(data: &[u8]) -> Result<bool, Error> {
    Ok(read_head(data)?.mode == MODE_RAW)
}

/// Whether the tensor blob is a passthrough of an unsupported dtype.
pub fn is_passthrough(data: &[u8]) -> Result<bool, Error> {
    Ok(read_head(data)?.passthrough)
}

/// Streams, planes and checksum of one chunk, located before decoding starts.
struct Located<'a> {
    chunk: ChunkRef<'a>,
    payload: &'a [u8],
    crc: Option<u32>,
}

/// Decompress a tensor blob into `out` (little-endian elements, `width * n` bytes), chunks in
/// parallel.
pub fn decompress_tensor(data: &[u8], out: &mut [u8]) -> Result<(), Error> {
    let Head {
        mut c,
        dtype,
        mode,
        crc,
        n,
        ..
    } = read_head(data)?;
    let width = dtype.width();
    let bytes = n.checked_mul(width).ok_or(Error::Corrupt)?;
    if bytes != out.len() {
        return Err(Error::LengthMismatch {
            expected: n as u64,
            got: (out.len() / width) as u64,
        });
    }
    match mode {
        MODE_RAW => {
            let values = c.take(bytes)?;
            if crc && crc32c(values) != c.u32()? {
                return Err(Error::Checksum);
            }
            out.copy_from_slice(values);
            return if c.rest().is_empty() {
                Ok(())
            } else {
                Err(Error::Corrupt)
            };
        }
        MODE_RANS if n > 0 => {}
        _ => return Err(Error::Corrupt),
    }
    let fmt = dtype.format().ok_or(Error::Corrupt)?;
    let chunk = c.u32()? as usize;
    if chunk == 0 || !chunk.is_multiple_of(32) {
        return Err(Error::Corrupt);
    }
    let (cb, used) = Codebook::read(c.rest())?;
    c.pos += used;
    let exp_limit = 1usize << fmt.exp_bits();
    if (cb.sm.is_some() && fmt.hi_bits() != 8)
        || cb.exp_freqs[exp_limit.min(256)..].iter().any(|&f| f > 0)
    {
        return Err(Error::Corrupt);
    }
    let n_chunks = c.u32()? as usize;
    if n_chunks != n.div_ceil(chunk) {
        return Err(Error::Corrupt);
    }
    let index = c.take(INDEX_ENTRY * WAYS * n_chunks)?;
    let sums = match crc {
        true => {
            let head = crc32c(&data[..c.pos]);
            let sums = c.take(4 * (1 + n_chunks))?;
            if head != u32::from_le_bytes([sums[0], sums[1], sums[2], sums[3]]) {
                return Err(Error::Checksum);
            }
            &sums[4..]
        }
        false => &[][..],
    };
    let tables = Box::new(DecTables::new(&cb).ok_or(Error::Corrupt)?);
    let coded = cb.sm.is_some();

    let mut located = Vec::with_capacity(n_chunks);
    for (k, entry) in index
        .as_chunks::<{ INDEX_ENTRY * WAYS }>()
        .0
        .iter()
        .enumerate()
    {
        let elems = chunk.min(n - k * chunk);
        let payload_at = c.pos;
        let mut streams = [[&[][..]; 3]; WAYS];
        for (w, s) in streams.iter_mut().enumerate() {
            let e = &entry[INDEX_ENTRY * w..INDEX_ENTRY * (w + 1)];
            let field =
                |i: usize| u32::from_le_bytes([e[4 * i], e[4 * i + 1], e[4 * i + 2], e[4 * i + 3]]);
            let [exp, sm, esc] = [0, 1, 2].map(|i| field(i) as usize);
            if (!coded && sm != 0) || (tables.esc.is_none() && esc != 0) {
                return Err(Error::Corrupt);
            }
            *s = [c.take(2 * exp)?, c.take(2 * sm)?, c.take(esc)?];
        }
        let (hi, bits) = match (coded, fmt.hi_bits()) {
            (true, _) => (&[][..], &[][..]),
            (false, 8) => (c.take(elems)?, &[][..]),
            (false, h) => (&[][..], c.take(h as usize * plane_len(elems))?),
        };
        let lo = c.take(fmt.low_bytes() * elems)?;
        located.push(Located {
            chunk: ChunkRef {
                elems,
                exp: streams.map(|s| s[0]),
                sm: streams.map(|s| s[1]),
                esc: streams.map(|s| s[2]),
                hi,
                bits,
                lo,
            },
            payload: &data[payload_at..c.pos],
            crc: crc.then(|| {
                u32::from_le_bytes([
                    sums[4 * k],
                    sums[4 * k + 1],
                    sums[4 * k + 2],
                    sums[4 * k + 3],
                ])
            }),
        });
    }
    if !c.rest().is_empty() {
        return Err(Error::Corrupt);
    }
    out.par_chunks_mut(width * chunk)
        .zip(located.par_iter())
        .try_for_each(|(dst, l)| {
            if l.crc.is_some_and(|want| crc32c(l.payload) != want) {
                return Err(Error::Checksum);
            }
            ops().decode_chunk(&tables, fmt, l.chunk, dst)?;
            Ok(())
        })
}
