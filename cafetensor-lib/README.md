# cafetensor-lib

Lossless rANS compression of safetensors checkpoints, with CPU-tier specific code selected once at startup.

## Building cafetensor-lib as a dependency

Cargo reads profiles only from the root manifest of the workspace being built, so a consumer must reproduce this workspace's tier configuration. Every tier crate carries a compile-time flag guard, so a consumer that skips these steps fails to build instead of silently running every tier at baseline.

1. Use a nightly toolchain (this workspace pins `nightly-2026-09-30` in `rust-toolchain.toml`).
2. Add the block below to the consumer's root `Cargo.toml`, verbatim. `cargo-features` goes at the very top of the file.
3. Pass `-Zshare-generics=n` in the consumer's build flags, for example in its `.cargo/config.toml`:

   ```toml
   [build]
   rustflags = ["-Zshare-generics=n"]
   ```

4. Do not set any ambient `target-cpu` or `target-feature` flag: not in any `.cargo/config.toml`, not in `[target.*] rustflags`, and not through `RUSTFLAGS`, `CARGO_ENCODED_RUSTFLAGS` or `CARGO_TARGET_*_RUSTFLAGS`. Cargo drops `[build] rustflags` whenever `RUSTFLAGS` is set.

<!-- BEGIN tier-profiles -->
```toml
cargo-features = ["profile-rustflags"]

# ---- tier flags: release ----
[profile.release.package.amd64_v2]
rustflags = ["-Ctarget-cpu=x86-64-v2"]
[profile.release.package.amd64_v3]
rustflags = ["-Ctarget-cpu=x86-64-v3"]
[profile.release.package.amd64_v4]
rustflags = ["-Ctarget-cpu=x86-64-v4"]
[profile.release.package.amd64_v4_icl]
rustflags = ["-Ctarget-cpu=x86-64-v4", "-Ctarget-feature=+avx512vbmi,+avx512vbmi2,+pclmulqdq,+vpclmulqdq"]
[profile.release.package.amd64_9800x3d]
rustflags = ["-Ctarget-cpu=znver5", "-Ctarget-feature=-rdseed"]

# ---- tier flags: dev (opt-level 3 is load-bearing, see leak path 1) ----
[profile.dev.package.amd64_v2]
opt-level = 3
rustflags = ["-Ctarget-cpu=x86-64-v2"]
[profile.dev.package.amd64_v3]
opt-level = 3
rustflags = ["-Ctarget-cpu=x86-64-v3"]
[profile.dev.package.amd64_v4]
opt-level = 3
rustflags = ["-Ctarget-cpu=x86-64-v4"]
[profile.dev.package.amd64_v4_icl]
opt-level = 3
rustflags = ["-Ctarget-cpu=x86-64-v4", "-Ctarget-feature=+avx512vbmi,+avx512vbmi2,+pclmulqdq,+vpclmulqdq"]
[profile.dev.package.amd64_9800x3d]
opt-level = 3
rustflags = ["-Ctarget-cpu=znver5", "-Ctarget-feature=-rdseed"]
```
<!-- END tier-profiles -->
