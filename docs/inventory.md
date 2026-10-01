# Phase 0 inventory of `tensor-compressor`

Source: `/home/valarauca/Documents/rust_stuff/gpu/tensor-compressor` at `cba8a21`. Paths below are relative to its `src/`.

## 1. Operations

Hot paths are marked **H**. Everything else runs once per tensor or once per container and stays generic but is not a tier override target.

### Encode side

| Operation | Old location | Signature (old) | SIMD today | Notes |
| --- | --- | --- | --- | --- |
| Element split **H** | `codec.rs:75` `Format::split`, driven from `tensor.rs:238` `compress_bytes` | `fn split(self, v: u32) -> (u8, u8, u32)` per element, rayon over 64 Ki element slices | None (scalar) | Produces three planes: exponent symbol, high residual (sign + top mantissa bits, `Format::hi_bits` wide), `Format::low_bytes` raw low bytes. |
| Joint histogram **H** | `codec.rs:203` `Codebook::build` | `[[u64; 256]; 256]` over (exp, hi), rayon fold/reduce | None | 512 KiB table. Becomes a caller-provided buffer; per-chunk `u32` counts summed by the library. |
| `normalize` | `codec.rs:117` | `fn normalize(counts: &[u64; 256], bits: u32) -> [u16; 256]` | None | Greedy repair by `f64::ln` cost. Needs `core_float_math` in `no_std`. |
| Exponent model + escape | `codec.rs:203` | `Codebook::build(exps, his, mode) -> Option<Codebook>` | None | Alphabet above 64: rarest merged into an escape symbol (an absent byte value). Tier = smallest of 16/32/64 buckets that fits. |
| Alias table (Vose) | `codec.rs:441` `ExpAlias::new`, `:487` `flat` | `fn new(freqs: &[u16; 256], tier: Tier) -> Option<ExpAlias>` | None | `div[64]`, `lo[64]`, `hi[64]` packed `sym \| (f-1)<<8 \| base<<20`. Full buckets store `hi == lo`. |
| SM context model | `codec.rs:319` `sm_model` | `fn sm_model(joint, exp_counts) -> SmModel` | None | `--slow` only, formats with an 8-bit high residual (BF16, F32). Top 15 exponents with count ≥ 4096 get a context, the rest share a tail. |
| Encoder tables | `codec.rs:504` `enc_table`, `:525` `EncTables::new` | `[EncSym; 256]` + `exp_slot[4096]` | None | `exp_slot` maps (sym, rank) to alias slot. |
| `encode_chunk` **H** | `codec.rs:584`, `:546` `encode_stream` | `fn encode_chunk(t: &EncTables, exps: &[u8], his: &[u8], out: &mut ChunkStreams)` | None | 4 ways × 16 lanes. Per symbol: one `u32` divide and modulo, at most one 16-bit word out, then the stream is reversed. Worst case ≤ 1 word per symbol + 32 flush words per way. Escape bytes per way. |
| Bitplane pack | `tensor.rs` inside `compress_bytes` | per group of 16, `u16` word per bit | None | F16 (3 planes), e5m2 (3), e4m3 (4). |
| Chunk framing | `tensor.rs:238` | index `(exp_words, sm_words, esc_bytes)[n][4]`, CRC array, payloads | rayon over chunks | Format in §4. |

### Decode side

| Operation | Old location | Signature (old) | SIMD today |
| --- | --- | --- | --- |
| `decode_chunk` **H** | `avx512.rs:62`, `:337` `decode_lines` | `fn decode_chunk(t: &DecTables, st: &mut ChunkState, out: &mut [u8]) -> usize`, specialized over `<CODED, N∈{16,32,64}, ESC, W∈{1,2,4}, M∈{2,3,7,10,23}>` | AVX-512 + VBMI + VBMI2 |
| Scalar decode / tail **H** | `codec.rs:795` `ChunkState::decode_scalar` | `fn decode_scalar(&mut self, t: &DecTables, out: &mut [u8], first: usize)` | None |
| Stream reader | `codec.rs:673..704` `Reader` | `refill(x) -> u32`, branch-free single refill | None |
| Chunk validation | `codec.rs:841` `ChunkState::finish` | all lanes back at `RANS_L`, every stream consumed | None |
| Decoder tables | `codec.rs:645` `DecTables::new` | `exp[4096]`, `sm[16 << 11]` (128 KiB), `ctx_of[256]`, alias | None |

Per-group SIMD steps in `decode_lines` (16 lanes, `u32x16`):

1. Exponent decode: `slot >> shift` bucket, `vpermb` divider lookup, `vpermi2d` / `vpermd` primary and alias entries, compare-and-blend, `vpmulld` state update.
2. Refill: compare against `RANS_L`, `vpexpandw` expand-load of `popcnt(k)` words (`refill`, `avx512.rs:128`).
3. Escape patch (alphabet > 64 only): `cmpeq` with escape code, `vpexpandb` expand-load of escaped bytes.
4. Coded SM (`--slow`): two `vpermi2b` over the 256-byte context map, `vpgatherdd` from the 128 KiB table, update, refill.
5. Raw residual: byte plane `vpmovzxbd`, bitplanes via `maskz_mov_epi32`, low bytes `vpmovzxbd`/`vpmovzxwd`.
6. Join into `sign | exp | mant` with variable shifts, narrow (`vpmovdw` / `vpmovdb`) and pack into 64-byte lines.

### Integrity

| Operation | Old location | Paths |
| --- | --- | --- |
| CRC32C **H** | `crc.rs:74` `update(state, data) -> u32` | table (`:92`), SSE4.2 `crc32` 8 B/instr (`:102`), VPCLMULQDQ 4 × zmm fold of 256 B + 128-bit reduction + SSE4.2 finish (`:137`). Constants computed by `fold_pair` (`:57`). |
| BLAKE3 **H** | `cafetensor.rs` via `blake3` crate (`update_rayon`, `update_mmap_rayon`) | Crate-internal dispatch. To be ported (decision). |

### Container and I/O (library only, not tier code)

`safetensors.rs` (header parse via `anamnesis`, `write_header`), `shards.rs` (`join`, `split`), `cafetensor.rs` (v1 container, gather/scatter, verify), `bytes.rs` (`AlignedBuf`: hugetlb 1 GiB / 2 MiB / THP / small `mmap`), `main.rs` (clap CLI). These move to `cafetensor-lib`, `cafetensor-bin` and `os_linux` / `os_darwin`.

## 2. Intrinsics and CPU features

Extracted from the stdarch `target_feature` attributes on the pinned nightly.

| Intrinsic | Feature | Used in |
| --- | --- | --- |
| `_mm512_*` add, and, or, xor, shifts (`slli/srli/sll/srl/srlv`), `mullo_epi32`, `cmp*_mask`, `test_epi32_mask`, `mask_blend/mask_or/mask_mov/maskz_mov_epi32`, `maskz_expand_epi32`, `permutexvar_epi32`, `permutex2var_epi32`, `i32gather_epi32`, `cvtepu8/cvtepu16_epi32`, `cvtepi32_epi8/epi16`, `inserti32x4/inserti64x4`, `extracti32x4`, `broadcast_i32x4`, casts, `loadu/storeu`, `set1/setzero` | `avx512f` | decoder, CRC fold |
| `_mm512_permutexvar_epi8`, `_mm512_permutex2var_epi8` | `avx512vbmi` | alias divider lookup, SM context lookup |
| `_mm_maskz_expandloadu_epi8` (and `_mm256_maskz_expandloadu_epi16` in the refill path) | `avx512vbmi2` + `avx512vl` | escape patch, refill |
| `_mm512_clmulepi64_epi128` | `vpclmulqdq` + `avx512f` | CRC fold |
| `_mm_clmulepi64_si128` | `pclmulqdq` | CRC 128-bit reduction |
| `_mm_crc32_u64`, `_mm_crc32_u8` | `sse4.2` | CRC short inputs and finish |
| `_mm_extract_epi64` | `sse4.1` | CRC finish |
| `_mm256_loadu_si256`, `_mm_loadu_si128`, `_mm_set_epi64x`, `_mm_cvtsi*` | `avx` / `sse2` | loads, scalars |

Derived feature deltas over the target baseline (pinned nightly, `rustc --print cfg`):

- `x86-64-v2`: cmpxchg16b lahfsahf popcnt sse3 sse4.1 sse4.2 ssse3
- `x86-64-v3`: v2 + avx avx2 bmi1 bmi2 f16c fma lzcnt movbe xsave
- `x86-64-v4`: v3 + avx512f avx512bw avx512cd avx512dq avx512vl
- `znver5`: v4 + adx aes avx512bf16 avx512bitalg avx512ifma avx512vbmi avx512vbmi2 avx512vnni avx512vp2intersect avx512vpopcntdq avxvnni clflushopt gfni pclmulqdq prfchw rdrand rdseed sha sse4a vaes vpclmulqdq xsavec xsaveopt xsaves

No x86-64 level contains `avx512vbmi`, `avx512vbmi2`, `pclmulqdq` or `vpclmulqdq`, so the old decoder and CRC fold can only run as written on the new `amd64_v4_icl` and `amd64_9800x3d` tiers. `lahfsahf` and `prfchw` have no `is_x86_feature_detected!` name.

## 3. What the entropy model conditions on

This sets what a distribution profile must capture.

- **Per tensor, not per position.** One codebook per tensor. No neighbour, row or column conditioning: measured on Qwen3.5-27B, neighbour XOR raises entropy from 10.47 to 11.13 bpw, and H(E | previous E) saves at most 0.015 bpw.
- **Exponent field** (`Format::exp_bits`: 8 for BF16/F32, 5 for F16/e5m2, 4 for e4m3). Its histogram decides the alias tier (alphabet ≤ 16/32/64) and the escape set (alphabet > 64, rarest first).
- **High residual** (`Format::hi_bits`: 8 for BF16/F32, 3 for F16/e5m2, 4 for e4m3). Raw by default. With `--slow` and 8-bit residuals it is coded conditioned on the exponent through ≤ 16 contexts, so profiles need the joint (exp, hi) histogram.
- **Low bytes** (`Format::low_bytes`: 2 for F32, 1 for F16, 0 otherwise). Always raw. Measured ≈ 7.998 to 7.9999 bits per byte, so a per-plane histogram is enough.
- **Element count**, because of per-chunk overhead (4 ways × 2 × 64 B of flush state, 48 B of index, CRC) and the raw fallback when coding does not pay.
- **Special values** (zeros, subnormals, inf, NaN). They only matter through the exponent histogram, but profiles record their fractions for edge-case coverage.
- **Non-float dtypes** (BOOL, U8..I64, F64) are passthrough, so profiles only need their size.

## 4. On-disk format to preserve (gate 6 byte equality)

Tensor blob (`tensor.rs`): `"CAF1"`, `u32` tag (dtype code bits 0..30, bit 30 big-endian, bit 31 passthrough), `u8` mode (0 raw, 1 rANS), `u8` flags (bit 0 CRC), `u64` n. For rANS: `u32` chunk_elems, codebook (sparse exponent freqs, tag byte tier|escape, escape symbol, n_ctx, contexts, SM tables), `u32` n_chunks, index, optional `head_crc` + per-chunk CRC, then per chunk the way streams, escapes, raw high plane or bitplanes, and low bytes.

Container (`cafetensor.rs`): `u64` length + JSON with `rasn_comp` first (`version`, `crc32c`, `checksum: "blake3-<hex>"`, `block`, `sources[]` with verbatim header ranges) and tensor entries `{dtype, shape, data_offsets, source}`, then the data section.

## 5. Proposed `Operations` (object-safe, slices / scalars / plain structs, caller-provided buffers)

| Method | Purpose | Caller-side bound |
| --- | --- | --- |
| `tier_name()` | identity | |
| `split_planes(fmt, bytes, exps, his, los)` | element split | planes sized `n`, `n`, `n × low_bytes` |
| `histogram(exps, his, joint: &mut [[u32; 256]; 256])` | per chunk joint counts | chunk ≤ 2^28 elements fits `u32` |
| `encode_chunk(tables, exps, his, los, fmt, mode, out) -> Result<ChunkInfo, OpError>` | streams + planes of one chunk, CRC | `out` ≥ worst-case bound computed by a generic `const fn` |
| `decode_chunk(tables, chunk: &ChunkRef, fmt, out) -> Result<(), OpError>` | full chunk incl. CRC check and validation | `out` exactly `elems × width` |
| `crc32c_update(state, data) -> u32` | CRC over any buffer | |
| `blake3_chunks(input, chunk_counter, flags, cvs_out) -> usize` | chaining values of whole 1 KiB chunks | `cvs_out` ≥ `input.len() / 1024` |
| `blake3_parents(cvs, flags, out)` | parent node merges | |

Codebook build (`normalize`, alias, contexts, encoder and decoder tables) is plain generic code in `generic_operations`, called once per tensor by the library, returning fixed-size arrays. Tables live in caller-provided storage (`DecodeTables` ≈ 145 KiB).

## 6. Proposed `Kernels` hooks and tier overrides

Defaults are `core::simd` at a fixed `u32x16` / `u8x64` width.

| Hook | Default (`core::simd`) | v2 | v3 | v4 | v4_icl, 9800x3d |
| --- | --- | --- | --- | --- | --- |
| `gather_u32x16(table, idx)` | `Simd::gather_select_unchecked` after masking | default | AVX2 `vpgatherdd` ×2 | `vpgatherdd zmm` | `vpgatherdd zmm` |
| `lookup{16,32,64}_u32(regs, idx)` | `swizzle_dyn` on bytes, else scalar | default | default | `vpermd` / `vpermi2d` (+blend) | same as v4 |
| `lookup_div(div, idx)` (alias dividers) | scalar per lane | default | default | `vpermi2d` on dword copy | `vpermb` |
| `lookup256_u8(map, idx)` (SM contexts) | scalar per lane | default | default | gather from 1 KiB dword copy | 2 × `vpermi2b` + blend |
| `expand_u16_lanes(mask, src)` (refill) | prefix sum + `swizzle_dyn` | `pshufb` LUT | `pshufb` LUT on ymm | `vpmovzxwd` + `vpexpandd` | `vpexpandw` |
| `expand_u8_lanes(mask, src)` (escape) | prefix sum + `swizzle_dyn` | default | default | `vpmovzxbd` + `vpexpandd` | `vpexpandb` |
| `bitplane_to_lanes(word, bit)` | `Mask::from_bitmask` select | default | default | `maskz_mov_epi32` | same |
| `narrow_u32_to_u16`, `narrow_u32_to_u8` | `cast` | default | default | `vpmovdw` / `vpmovdb` | same |
| `crc32c_block(state, &[u8; 64])` | slice-by-8 table | `crc32` instr | `crc32` instr | `crc32` instr | `vpclmulqdq` fold (+ `pclmulqdq` reduce, `crc32` finish) |
| BLAKE3 `g` / round (16 chunks across `u32x16`) | rotates as shift-or | default | default | `vprord` | `vprord` |

Overrides are added one hook at a time in Phase 3, each measured against the default. The v3 `pshufb` expand and the v4 dword-table forms have no counterpart in the old project, so gate 6 records their throughput against old scalar or old AVX-512 code as noted per tier.

## 7. Port order (Phase 3)

`crc32c_update` → `split_planes` + bitplanes → `histogram` + codebook build → `encode_chunk` → `decode_chunk` → BLAKE3.
