#!/usr/bin/env bash
# Run every verification gate. A step is done only when this passes.
set -euo pipefail
cd "$(dirname "$0")/.."
fail() { echo "verify: FAIL: $*" >&2; exit 1; }

scripts/audit-flags.sh || fail "gate 1"
scripts/audit-public-api.sh || fail "gate 2"
scripts/emulated-tests.sh || fail "gate 3"
scripts/sample-tests.sh || fail "gate 4"
scripts/audit-disasm.sh | tail -1 || fail "gate 5"
[ "${PIPESTATUS[0]}" = 0 ] || fail "gate 5"
scripts/lint.sh || fail "lint"
echo "verify: all gates passed"
