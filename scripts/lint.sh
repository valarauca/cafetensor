#!/usr/bin/env bash
# rustfmt and clippy on every supported target: x86-64 Linux (every crate, fuzz included),
# AArch64 Linux and AArch64 macOS (every crate but the x86-64 tiers).
set -euo pipefail
cd "$(dirname "$0")/.."
fail() { echo "lint: FAIL: $*" >&2; exit 1; }
x86=(-p cafetensor-bin -p cafetensor-lib -p general_backend -p generic_operations
     -p amd64_v2 -p amd64_v3 -p amd64_v4 -p amd64_v4_icl -p amd64_9800x3d
     -p os_common -p cafetensor-testkit -p cafetensor-fuzz)
arm=(-p cafetensor-bin -p cafetensor-lib -p general_backend -p generic_operations
     -p os_common -p cafetensor-testkit)

cargo fmt --all --check || fail "fmt"
echo "fmt: ok"
cargo clippy -q "${x86[@]}" --all-targets -- -D warnings || fail "clippy x86_64"
echo "clippy x86_64: ok"
cargo clippy -q "${arm[@]}" --all-targets --target aarch64-unknown-linux-gnu -- -D warnings || fail "clippy aarch64"
echo "clippy aarch64: ok"
cargo clippy -q "${arm[@]}" --all-targets --target aarch64-apple-darwin -- -D warnings || fail "clippy aarch64-apple-darwin"
echo "clippy aarch64-apple-darwin: ok"
