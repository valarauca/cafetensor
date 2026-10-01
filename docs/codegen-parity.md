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
