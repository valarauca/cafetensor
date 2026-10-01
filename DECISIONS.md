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

`normalize()` needs `ln`. `generic_operations` enables `#![feature(core_float_math)]`. If a result ever differs from the old `std` build, gate 6 falls back to cross-decoding and the case is recorded here.

## Empty OS re-exports carry `allow(unused_imports)`

`os_linux` and `os_darwin` start empty as the brief requires, so the glob re-exports in `os_common` are unused until mainline code adds items.
