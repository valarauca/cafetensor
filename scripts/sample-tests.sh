#!/usr/bin/env bash
# Gate 4: sample tests on every tier this host can run, dev and release, plus the testkit
# self-test and the library tests.
set -euo pipefail
cd "$(dirname "$0")/.."
fail() { echo "gate 4: FAIL: $*" >&2; exit 1; }
cargo test -q -p general_backend >/dev/null || fail "dev"
cargo test -q --release -p general_backend >/dev/null || fail "release"
cargo test -q --release -p cafetensor-testkit >/dev/null || fail "testkit self-test"
cargo test -q --release -p cafetensor-lib >/dev/null || fail "library"
echo "gate 4 (sample tests, every tier): ok"
