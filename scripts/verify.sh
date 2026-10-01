#!/usr/bin/env bash
# Run every verification gate. A step is done only when this passes.
set -euo pipefail
cd "$(dirname "$0")/.."
fail() { echo "verify: FAIL: $*" >&2; exit 1; }
x86=(-p cafetensor-bin -p cafetensor-lib -p general_backend -p generic_operations
     -p amd64_v2 -p amd64_v3 -p amd64_v4 -p amd64_v4_icl -p amd64_9800x3d
     -p os_common -p os_linux -p os_darwin -p cafetensor-testkit)
arm=(-p cafetensor-bin -p cafetensor-lib -p general_backend -p generic_operations
     -p os_common -p os_linux -p cafetensor-testkit)
arm_cfg=(--target aarch64-unknown-linux-gnu --config 'target.aarch64-unknown-linux-gnu.linker="aarch64-linux-gnu-gcc-13"')

scripts/audit-flags.sh || fail "gate 1"
scripts/audit-public-api.sh || fail "gate 2"
scripts/emulated-tests.sh || fail "gate 3"
cargo test -q -p general_backend >/dev/null || fail "gate 4 (dev)"
cargo test -q --release -p general_backend >/dev/null || fail "gate 4 (release)"
cargo test -q --release -p cafetensor-testkit >/dev/null || fail "gate 4 (testkit self-test)"
echo "gate 4 (sample tests, every tier): ok"
scripts/audit-disasm.sh | tail -1 || fail "gate 5"
[ "${PIPESTATUS[0]}" = 0 ] || fail "gate 5"
cargo fmt --all --check || fail "fmt"
echo "fmt: ok"
cargo clippy -q "${x86[@]}" --all-targets -- -D warnings || fail "clippy x86_64"
echo "clippy x86_64: ok"
cargo clippy -q "${arm[@]}" --all-targets "${arm_cfg[@]}" -- -D warnings || fail "clippy aarch64"
echo "clippy aarch64: ok"
echo "verify: all gates passed"
