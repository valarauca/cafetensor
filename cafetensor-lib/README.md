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

## Using the library

Every call runs on the CPU tier `general_backend` selects once per process; `tier_name()` reports it, and `CAFETENSOR_TIER=<name>` forces one (an unsupported name is a hard error). Parallel work runs on the global rayon pool.

| Module | Purpose |
| --- | --- |
| `container` | `compress` gathers `.safetensors` files into one `.cafetensor`; `decompress` restores them byte for byte, optionally checking each file's BLAKE3; `read_header` parses the `rasn_comp` header. |
| `tensor` | The per-tensor blob: `compress_bytes` and `decompress_tensor`, with `Options` for block size, coded sign\|mantissa and CRC32C. |
| `safetensors` | `Source::open` parses a header without reading tensor data; `write_header` serializes one. |
| `shards` | `join` merges sharded files; `split` writes shards plus a Hugging Face style index. |
| `hash` | Streaming BLAKE3 (`Hasher`, `hash`, `b3sum`) in the container's `blake3-<hex>` notation. |

`HugeBuf` is the decode output buffer: an anonymous mapping backed by 1 GiB or 2 MiB huge pages, or transparent huge pages, when the Linux kernel allows, and ordinary pages otherwise. Darwin takes no page size request for anonymous memory, so it always gets ordinary pages.

The `.cafetensor` container and the tensor blobs are byte-identical to those of the `tensor-compressor` project, so either tool decodes the other's files.

## Command line

`cafetensor-bin` builds the `cafetensor` binary, which logs the selected tier on stderr and offers `compress`, `decompress`, `inspect`, `b3sum`, `join`, `split` and `bench`. Run without a subcommand it runs a self-test on the selected tier. `-j N` limits the worker threads.
