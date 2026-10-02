# Decisions and deviations from the brief

Each entry records what differs from the brief and why.

## New tier `amd64_v4_icl`

The old decoder needs AVX512-VBMI (`vpermb`, `vpermi2b`) and AVX512-VBMI2 (`vpexpandw`, `vpexpandb`), and the CRC fold needs PCLMULQDQ and VPCLMULQDQ. No x86-64 level contains any of them. Per the brief, `amd64_v4` is not widened. A new tier `amd64_v4_icl` (`platform/amd64_ia64/v4_icl/`) is built with `-Ctarget-cpu=x86-64-v4 -Ctarget-feature=+avx512vbmi,+avx512vbmi2,+pclmulqdq,+vpclmulqdq`, only the features its overrides use. It covers Ice Lake, Sapphire Rapids and Zen 4.

Gate 3 expectations change accordingly: SDE `-icx` and `-spr` select `amd64_v4_icl` (the brief expected `amd64_v4` for `-spr`), `-skx` selects `amd64_v4`.

`amd64_v4` overrides use AVX-512 F/BW/CD/DQ/VL only. VBMI operations are re-expressed with dword permutes, dword gathers and `vpexpandd`.

## Identical overrides in `amd64_v4_icl` and `amd64_9800x3d`

Both tiers carry the same VBMI/VBMI2/VPCLMULQDQ overrides. Sharing code between tier crates outside `generic_operations` is prohibited, and `generic_operations` may not use `core::arch`, so the overrides are copied.

## `amd64_9800x3d` drops RDSEED

`-Ctarget-cpu=znver5` enables `rdseed`, but Linux masks RDSEED on Zen 5 hosts affected by AMD's RDSEED erratum, and `is_x86_feature_detected!("rdseed")` returns false on the reference 9800X3D. The detection set must equal the flag set, so the tier is built with `-Ctarget-cpu=znver5 -Ctarget-feature=-rdseed` and does not check RDSEED. Nothing in the codec uses RDSEED.

## Features without a detection name

`lahfsahf` (all tiers) and `prfchw` (`amd64_9800x3d`) appear in the derived lists but have no `is_x86_feature_detected!` name on the pinned nightly. LAHF/SAHF in 64-bit mode is present on every CPU with SSE4.2/POPCNT (x86-64-v2 hosts), and PREFETCHW is present on every Zen part that passes the other znver5 checks. Neither is used explicitly by the codec.

## `clflushopt` detection is feature-gated

`is_x86_feature_detected!("clflushopt")` requires `#![feature(clflushopt_target_feature)]` on the pinned nightly, enabled in `general_backend` only.

## BLAKE3 is ported into `generic_operations`

The old project used the `blake3` crate, which performs its own CPU dispatch. BLAKE3 is reimplemented in `generic_operations` with tier overrides. The crate is a dev-dependency oracle in tests only.

## `f64::ln` in `no_std`

`normalize()` needs `ln`, which `core` does not provide (`core::f64::math` has no logarithm on the pinned nightly). `generic_operations` calls `core::intrinsics::log` (what `std`'s `f64::ln` calls) under `#![feature(core_intrinsics)]`. It lowers to the same `llvm.log.f64` / libm `log` that `std`'s `f64::ln` uses, so normalized frequencies match the old project bit for bit on the same host.
## Empty OS re-exports carry `allow(unused_imports)`

`os_linux` and `os_darwin` start empty as the brief requires, so the glob re-exports in `os_common` are unused until mainline code adds items.

## Gate 2 passes tier flags to rustdoc

Profile `rustflags` reach rustc only, so documenting a tier crate trips its flag guard. `scripts/audit-public-api.sh` sets `RUSTDOCFLAGS` to the tier's flags for the `cargo public-api` run. `RUSTDOCFLAGS` is not one of the forbidden variables and does not affect compiled code.

## Gate 3 uses QEMU, Intel SDE is deferred

Intel SDE validation is deferred to a later sprint. Gate 3 runs the tests and a binary smoke run, in dev and release, under `qemu-x86_64-static -cpu <model>`, which fakes CPUID and raises SIGILL on unsupported instructions:

| QEMU CPU | Expected tier |
| --- | --- |
| `qemu64` (x86-64 baseline) | `portable` |
| `Nehalem` | `amd64_v2` |
| `Haswell` | `amd64_v3` |
| `Skylake-Server` | `amd64_v3` (QEMU TCG has no AVX-512, so those CPUID bits are masked and selection must fall back cleanly) |

AArch64 runs under `qemu-aarch64-static -L /usr/aarch64-linux-gnu` and must select `portable`. The AVX-512 tiers (`amd64_v4`, `amd64_v4_icl`, `amd64_9800x3d`) are exercised natively on the 9800X3D, where `available()` lists all of them and gate 4 compares each against `portable`. Selection of `amd64_v4` and `amd64_v4_icl` as the best tier on real Skylake-SP and Ice Lake class hosts is not emulated until SDE lands.

## llvm-mca for inner-loop throughput, not for ISA validation

`llvm-mca` (LLVM 19, `-mcpu` per tier) is the gate 6 tool for inner-loop throughput and port-pressure comparisons between old and new code. Neither `llvm-mca` nor `llvm-mc` can validate that code stays inside a tier: both accept and schedule any x86 instruction regardless of `-mcpu` and `-mattr` (verified: `vpermb` is accepted for `-mcpu=nehalem`).

## Testkit profiles pool tensors and also record per-tensor entropy

A profile holds one pooled histogram per dtype per source, which is what samples are drawn from. The codec builds one model per tensor, so pooled entropy overstates what the old project achieved (SAM 2.1 F32: pooled 26.84 bits, per tensor 26.58, old project 26.67). Profiles record both. Ratio checks on generated samples compare against the sample's own entropy, and comparisons with `old_ratio` use the per-tensor figures.

## The blake3 crate uses its pure-Rust backend only on AArch64

`blake3` (testkit file provenance hashes, and the oracle for the ported BLAKE3) enables the crate's `pure` feature only for AArch64 targets. Its C/assembly backend needs an unversioned `aarch64-linux-gnu-gcc` for AArch64 test builds. On x86-64 it keeps its assembly backend, because the gate 6 harness links the testkit next to the old project and Cargo's feature unification would otherwise force `pure` onto the old project's hasher too.

## crc32c on `amd64_9800x3d` is 20% slower than on `amd64_v4_icl`

Both tiers run the identical VPCLMULQDQ fold source. `-Ctarget-cpu=znver5` makes LLVM unroll the loop 4× and group the carry-less multiplies, which measures about 40 GB/s against about 51 GB/s with generic scheduling on the same 9800X3D (details in docs/codegen-parity.md). The tier keeps the brief's `-Ctarget-cpu=znver5` for now, because Zen 5 scheduling may help the decoder ops still to be ported. Agreed with the owner: decide after the decoder is ported, by measuring both tunings on the decode loop. LLVM's Zen 5 model (and `llvm-mca`) rates both schedules equal while the real chip does not, so the final call is validated on real hardware, including other Zen 5 and Intel hosts reached over SSH, not on the model. CRC is a small share of decode time either way, since it runs over compressed bytes only.

Measured after the decoder port (64 MiB samples, best of three runs, single thread, 9800X3D), the same `amd64_9800x3d` source built with `-Ctarget-cpu=znver5` and with `-Ctarget-cpu=znver5 -Ztune-cpu=x86-64-v4`:

| Op (GB/s) | znver5 tuning | x86-64-v4 tuning |
| --- | --- | --- |
| decode BF16 raw | 11.36 | 11.33 |
| decode BF16 coded | 4.13 | 4.13 |
| decode F32 raw | 18.37 | 18.51 |
| decode F32 coded | 7.11 | 7.23 |
| decode F16 raw | 9.09 | 9.02 |
| decode F8_E4M3 raw | 4.56 | 4.53 |
| decode F8_E5M2 raw | 4.93 | 4.91 |
| crc32c | 41.35 | 50.92 |

Decode is tuning-neutral within 2%, so Zen 5 scheduling buys nothing on the decoder, and the generic tuning recovers the crc32c loss. The tier flags are unchanged until the owner decides and the result is checked on real Zen 5 and Intel hosts.

## Histogram on v3, v4 and v4_icl is about 10% slower than the old scalar loop

The joint histogram is the same scalar loop on every tier. On AVX tiers LLVM's SLP vectorizer computes four `exp << 8` indices in an xmm register and extracts them with `vpextrd`, which measures 4.3 to 4.4 G elements/s against 4.9 on `amd64_v2`, `portable` and `amd64_9800x3d` (single thread, 256 Mi elements), and 5 to 9% below the old project's loop on one thread. A word-at-a-time variant was slower on every tier (3.6 to 4.6). The loop is kept: the histogram is one pass per tensor before the rANS encode, which dominates encode time, so the end-to-end cost is around 1 to 2% on those tiers. Codebook bytes are identical to the old project in every case.

## core::simd byte shuffles and gathers do not follow tier flags

`Simd::swizzle_dyn` chooses its instruction with `cfg(target_feature)` inside `core`, which is prebuilt at the baseline, so on x86-64 it is a scalar byte loop on every tier (on AArch64 it is `tbl`). `Simd::gather_or_default` lowers to `llvm.masked.gather`, which LLVM scalarized into `vpextrq`/`vpinsrd` sequences for the v3, v4, v4_icl and znver5 builds. With both as the only decode path, every x86 tier decoded at 0.7 GB/s (BF16 raw).

The decoder therefore has small hooks with portable defaults: `Kernels::shuffle_bytes` (overridden with `pshufb` on `amd64_v2` and `amd64_v3`), `exp_entries`, `sm_bases`, `sm_entries` (register alias tables and explicit `vpgatherdd` on the AVX-512 tiers), `refill` (`vpexpandd` on the AVX-512 tiers) and `escapes` (`vpexpandb` on the VBMI2 tiers).

## amd64_v3 decodes without hardware gathers

Explicit AVX2 `vpgatherdd` in the `amd64_v3` decode hooks measured 10 to 18% faster on Zen 5, but they are not used. The installed QEMU 8.2.2 (gate 3) decodes a VSIB index register of `ymm4` as "no index" (SIB index 100b), so every lane gathers the table base; a five-instruction reproducer passes natively and fails under `qemu-x86_64-static -cpu Haswell`. Register allocation decides when LLVM picks `ymm4`, so the gathers could not pass gate 3 reliably. Intel cores carrying the Gather Data Sampling microcode mitigation also run `vpgatherdd` far slower than Zen 5 does, so the gain was not expected to hold on Intel AVX2 hosts. Revisit with SDE or a fixed QEMU in the SDE sprint, measured on real Intel and AMD AVX2 hardware.

## amd64_v4 decodes without VBMI

The old decoder needed VBMI and VBMI2. `amd64_v4` (AVX-512 F/BW/CD/DQ/VL) widens the alias dividers to dwords and looks them up with `vpermd`/`vpermi2d`, reads the sign|mantissa context from `DecTables::ctx_nibbles` (eight 4-bit contexts per dword, one `vpermi2d` and a variable shift) instead of a gather, and patches escapes with the scalar default since they are rare. It decodes at the speed of the VBMI tiers on the 9800X3D.

## BLAKE3 lane count is a per-tier constant

`Kernels::BLAKE3_LANES` (4, 8 or 16) sets how many chunks or parents the ported BLAKE3 compresses at once. It is a constant rather than a hook because the whole hashing loop is generic over the lane count. Measured on the 9800X3D, wider was faster on every x86 tier even when the 16-word state and message spill: 16 lanes on `amd64_v3` and the AVX-512 tiers, 8 on `amd64_v2`. `portable` keeps 4, which fits AArch64 NEON registers without spills; it is revisited when an AArch64 host is measured.

## Helpers called from tier code are generic over the kernel type

A non-generic `#[inline]` helper (the scalar BLAKE3 `chunk_cv` at first) is copied into every crate that calls it, so the tier crates emitted AVX2 and AVX-512 copies under one untiered symbol name, and gate 5 rejected them. Every `generic_operations` function reachable from an `Engine` method is therefore either `#[inline(always)]` or generic over `K: Kernels`, which puts the tier into its mangled name; helpers that do not otherwise use `K` carry `#[allow(clippy::extra_unused_type_parameters)]`.

## Library and binary (Phase 4)

- **Formats are byte-identical to the old project.** The tensor blob and the `.cafetensor` container are reproduced field for field, including the JSON key order, which needs `serde_json`'s `preserve_order` feature as the old project had. `scripts/compare-old.sh e2e` checks identical bytes and cross-decoding on the local checkpoints.
- **Safetensors headers are parsed with `anamnesis` 0.6.9**, the old project's parser, so both accept and reject the same files.
- **Huge-page buffers use `memmap2`** in `os_linux` (hugetlb 1 GiB and 2 MiB, then `MADV_HUGEPAGE`, then small pages, the old policy) and plain anonymous maps in `os_darwin`. Both expose the same `HugeBuf` and `Backing` items, and neither needs `unsafe` code of its own.
- **BLAKE3 streams through the tier.** `cafetensor_lib::hash::Hasher` keeps the reference stack of subtree chaining values. Each update cuts its input into the largest aligned power-of-two subtrees (leaving the last chunk buffered), hashes all of them at once on the rayon pool in 256 KiB leaves through `blake3_subtree`, and merges each subtree's parents in order through `blake3_parent`. A recursive `rayon::join` tree with 128 KiB leaves measured 25% slower; 256 KiB leaves run within 1% of one large call. `b3sum` reads the file on one extra thread into two alternating 16 MiB huge-page buffers, which stay in cache while the pool hashes them; mapping the file like the old `update_mmap_rayon` would need `unsafe`. Partial groups of whole chunks in a batch now go through the SIMD lanes padded, instead of the scalar path.
- **`--backend` is gone from `decompress`.** The old flag chose between its scalar and AVX-512 decoders; tier choice is now process-wide through `CAFETENSOR_TIER`. Running the binary without a subcommand runs the self-test, which gate 3 uses as its smoke run.
- **`compress_bytes` returns `Result`** instead of panicking on a partial element.
- **Gate 4 ratio check compares two ways.** A generated sample's ratio must be within the profile's tolerance of the sample's own order-0 bound (coded exponent plus raw residual). Against the old project's ratio, the sample is compressed in pieces of the profile's mean tensor size, so fixed per-tensor overhead counts as it did on the real file (it dominates the Qwen3-VL F32 scales, 149 tensors of about 1,700 elements), and the old ratio is shifted by the pooled minus per-tensor exponent entropy. Every profile agrees within 0.004.
- **Gate 5 admits memchr's runtime-dispatched AVX2 functions.** `serde_json` (needed for the container header and by `anamnesis`) depends on `memchr`, whose `std` feature compiles `memchr::arch::x86_64::*::*_avx2` at baseline with `#[target_feature(enable = "avx2")]` and calls them only after its own `is_x86_feature_detected!` check. They are third-party code outside the tier mechanism and cannot leak tier flags. The audit allows exactly those symbols, matched on the raw v0 path `6memchr4arch6x86_64` and an `_avx2` function name, and only `vex` and `ymm` use; anything else outside a tier still fails.
