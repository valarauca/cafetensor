# Codegen parity with tensor-compressor (gate 6)

Measured on the reference Ryzen 7 9800X3D with `scripts/compare-old.sh <op>`. Samples are 256 MiB per profile dtype, generated in memory from `testdata/profiles/`. Throughput is the best of five runs, single thread. "Old" is the old project's own dispatch on this host.

## crc32c_update

Output: identical to the old `crc::update` for every sample and tier, and to a bitwise reference for every length 0 to 65,537, at offsets 1 to 63 and for arbitrary seed states.

| Tier | Path | GB/s (range over 10 samples) | Old on this host | Delta |
| --- | --- | --- | --- | --- |
| `amd64_9800x3d` | VPCLMULQDQ 4 × zmm fold | 39.8 to 41.5 | 47.4 to 51.1 (`update_vpclmul`) | −20%, explained in DECISIONS.md |
| `amd64_v4_icl` | VPCLMULQDQ 4 × zmm fold | 51.5 to 53.5 | 47.4 to 51.1 (`update_vpclmul`) | +5% |
| `amd64_v4` | SSE4.2 `crc32q` | 13.8 to 13.9 | old `update_hw` (same instruction) | equal |
| `amd64_v3` | SSE4.2 `crc32q` | 13.8 to 13.9 | old `update_hw` | equal |
| `amd64_v2` | SSE4.2 `crc32q` | 13.8 to 13.9 | old `update_hw` | equal |
| `portable` | byte table | 0.65 | old `update_table` | equal path |

Inner loops (`llvm-objdump` on the release binary):

| Code | Loop body | Bytes per iteration |
| --- | --- | --- |
| old `update_vpclmul` | 8 `vpclmulqdq` + 4 `vpternlogq`, interleaved per accumulator | 256 |
| new `amd64_v4_icl` | 2 × (8 `vpclmulqdq` + 4 `vpternlogq`), interleaved | 512 |
| new `amd64_9800x3d` | 4 × (8 `vpclmulqdq` grouped, then 4 `vpternlogq`) | 1024 |
| new v2, v3, v4 and old `update_hw` | 8 × `crc32q` | 64 |

`llvm-mca -mcpu=znver5` predicts 4.0 cycles per 64 bytes for both new fold loops, so the static model does not explain the 9800x3d gap. Building the 9800x3d tier with `-Ztune-cpu=x86-64-v4` (znver5 features, generic scheduling) measured 50.0 GB/s against 50.6 GB/s for `amd64_v4_icl`, so the loss comes from LLVM's Zen 5 scheduling and unrolling of this loop.

## split_planes and pack_bitplanes

Output: identical to the old project's split loop (`codec::Format::split` per element, as in `compress_bytes`) and bitplane loop for every float profile sample on every tier, and to a scalar reference for lengths 0 to 1000 at offsets 0 to 63. The old project had no SIMD path for either step; it ran them per element (and split under rayon).

Throughput, GB/s of input, 256 MiB samples, single thread:

| Sample | Old split | 9800x3d | v4_icl | v4 | v3 | v2 | portable |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BF16 (Qwen3.5, Qwen3-VL) | 1.11 | 15.3 to 15.5 | 15.2 to 15.4 | 15.7 | 15.2 to 15.3 | 14.8 to 15.0 | 14.9 to 15.0 |
| F32 (SAM, Grounding DINO, Wan, Qwen3-VL) | 1.71 to 1.73 | 12.8 to 14.6 | 14.6 to 15.0 | 14.4 to 14.9 | 13.3 to 13.5 | 13.0 to 13.2 | 12.5 to 12.9 |
| F16 (SDXL) | 1.01 | 11.0 | 11.0 | 11.0 | 10.3 | 9.9 | 9.8 |
| F8_E4M3 / F8_E5M2 | 0.57 | 9.1 to 9.2 | 8.9 to 9.3 | 8.7 to 9.0 | 8.9 to 9.2 | 8.7 to 9.3 | 9.0 to 9.4 |

Bitplanes (GB/s of high-residual bytes): old 3.7 to 5.0, new 8.9 to 14.8 across tiers.

Every tier is 8 to 16 times faster than the old scalar loop, and tiers sit within about 20% of each other because a 256 MiB split streams through DRAM: the loop is bound by memory, not instructions. No tier overrides were added; the `core::simd` defaults are the code on every tier.

Inner loops of the split helpers (instantiated per tier, so each tier's copy is compiled with its own flags):

| Helper | portable | v2 | v3 | v4 / v4_icl | 9800x3d | Old (per element) |
| --- | --- | --- | --- | --- | --- | --- |
| `split_w2` (BF16, F16), 64 elements | 98 to 132 | 98 to 132 | 46 to 58 | 27 to 39 | 53 to 75 | 56 to 59 |
| `split_w4` (F32), 16 elements | 70 | 66 | 37 | 22 | 22 | 56 to 59 |
| `split_w1` (FP8), 64 elements | 47 | 47 | 21 | 20 | 36 | 56 to 59 |

## histogram and codebook build

Output: the serialized codebook (`Codebook::build` then `write`) is byte-identical to the old project's for every float profile sample, in raw mode and, for BF16 and F32, in coded sign|mantissa mode, on every tier. The new path builds the joint histogram per 1 Mi-element chunk through the selected tier and sums it in `u64`; the old path built it in one rayon fold.

Single thread, G elements/s, histogram plus build plus write (old build forced to one thread with `RAYON_NUM_THREADS=1`):

| Sample | Old | 9800x3d | v4_icl | v4 | v3 | v2 | portable |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BF16 raw (Qwen3.5) | 3.40 | 3.66 | 3.10 | 3.11 | 3.14 | 3.48 | 3.45 |
| BF16 coded (Qwen3.5) | 3.19 | 3.44 | 3.01 | 2.98 | 3.04 | 3.32 | 3.30 |
| F32 raw (SAM) | 3.46 | 3.91 | 3.39 | 3.30 | 3.45 | 3.83 | 3.90 |
| F16 raw (SDXL) | 2.95 | 3.34 | 3.05 | 3.04 | 3.01 | 3.20 | 3.20 |
| F8_E4M3 raw (Qwen3-VL) | 3.87 | 4.02 | 3.68 | 3.65 | 3.72 | 4.02 | 4.01 |

v3, v4 and v4_icl trail the old loop by up to 9% because of LLVM's SLP vectorization of the index arithmetic; explained in DECISIONS.md. Histogram inner loop: 8 instructions per element scalar (13 per two elements on v2/portable), 21 per four elements on AVX tiers (`vpmovzxbd`, `vpslld`, 3 × `vpextrd`).

## encode_chunk

Output: the per-way exponent words, sign|mantissa words and escape bytes of every 1 Mi-element chunk are byte-identical to the old `codec::encode_chunk` streams for every float profile sample (64 MiB each), raw and coded, on every tier. A test-only reference rANS decoder also recovers every symbol, escape and coded sign|mantissa byte with all final states back at `RANS_L`.

Two changes from the old encoder, neither of which alters a single output bit:

- `x / freq` and `x % freq` use Lemire's exact 64-bit reciprocal (`mulhi(ceil(2^64/freq), x)`, `freq = 1` special-cased), checked exhaustively in a unit test for every frequency 0 to 4096.
- Renormalization can be branch-free (`Kernels::ENCODE_BRANCHLESS`): the candidate word is always written below the cursor, which advances only when the word is kept. It is enabled on `amd64_v4`, `amd64_v4_icl` and `amd64_9800x3d`, where it measured 40 to 50% faster; on `amd64_v3`, `amd64_v2` and `portable` the branchy form measured 8 to 17% faster and stays the default.

Single thread, G elements/s (old forced to one thread):

| Sample | Mode | Old | 9800x3d | v4_icl | v4 | v3 | v2 | portable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BF16 (Qwen3.5) | raw | 0.37 | 0.49 | 0.52 | 0.51 | 0.40 | 0.37 | 0.36 |
| BF16 (Qwen3.5) | coded | 0.15 | 0.23 | 0.17 | 0.17 | 0.17 | 0.16 | 0.16 |
| F32 (SAM) | raw | 0.36 | 0.48 | 0.50 | 0.51 | 0.38 | 0.35 | 0.35 |
| F32 (SAM) | coded | 0.15 | 0.24 | 0.18 | 0.18 | 0.17 | 0.16 | 0.16 |
| F32 (Qwen3-VL scales, H(exp) about 1 bit) | raw | 0.51 | 0.49 | 0.53 | 0.53 | 0.59 | 0.50 | 0.52 |
| F16 (SDXL) | raw | 0.37 | 0.51 | 0.51 | 0.52 | 0.40 | 0.36 | 0.36 |
| F8_E4M3 (Qwen3-VL) | raw | 0.35 | 0.48 | 0.52 | 0.52 | 0.39 | 0.36 | 0.35 |
| F8_E5M2 (Wan) | raw | 0.36 | 0.46 | 0.49 | 0.49 | 0.39 | 0.36 | 0.36 |

The worst case is the low-entropy Qwen3-VL F32 sample on `amd64_9800x3d`, 4% below the old encoder (within the 5% rule). Encoding stays scalar per lane on every tier; a vectorized 16-lane encoder is a candidate for later work.
