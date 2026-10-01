#!/usr/bin/env bash
# Gate 3: tier selection and tests on native, SDE-emulated and AArch64 hosts, dev and release.
# Rows whose emulator is missing are reported as PENDING, not passed.
set -euo pipefail
cd "$(dirname "$0")/.."
pending=0

run_row() {  # name expect runner-json target
    local name="$1" expect="$2" runner="$3" target="$4"
    for pflag in "" "--release"; do
        local cfg=() tgt=()
        [ -n "$runner" ] && cfg=(--config "target.$target.runner=$runner")
        [ "$target" != x86_64-unknown-linux-gnu ] && tgt=(--target "$target") && cfg+=(--config "target.$target.linker=\"aarch64-linux-gnu-gcc-13\"")
        CAFETENSOR_EXPECT_TIER="$expect" cargo test -q $pflag "${tgt[@]}" "${cfg[@]}" -p general_backend >/dev/null \
            || { echo "gate 3: FAIL: $name tests ${pflag:-dev}"; exit 1; }
        local out
        out="$(cargo run -q $pflag "${tgt[@]}" "${cfg[@]}" -p cafetensor-bin 2>&1 >/dev/null)" \
            || { echo "gate 3: FAIL: $name smoke ${pflag:-dev}: $out"; exit 1; }
        [ "$out" = "cafetensor: tier $expect" ] || { echo "gate 3: FAIL: $name selected '$out', expected $expect"; exit 1; }
    done
    echo "$name: $expect (dev, release)"
}

native="$(cargo run -q -p cafetensor-bin 2>&1 >/dev/null | sed 's/^cafetensor: tier //')"
expect_native="${CAFETENSOR_NATIVE_TIER:-amd64_9800x3d}"
[ "$native" = "$expect_native" ] || { echo "gate 3: FAIL: native selected $native, expected $expect_native"; exit 1; }
run_row native "$expect_native" "" x86_64-unknown-linux-gnu

if command -v sde64 >/dev/null; then
    for row in "nhm amd64_v2" "hsw amd64_v3" "skx amd64_v4" "icx amd64_v4_icl" "spr amd64_v4_icl"; do
        set -- $row
        run_row "sde -$1" "$2" "[\"sde64\",\"-$1\",\"--\"]" x86_64-unknown-linux-gnu
    done
else
    echo "sde rows: PENDING (sde64 not installed)"; pending=1
fi

if command -v qemu-aarch64 >/dev/null; then
    run_row aarch64 portable '["qemu-aarch64","-L","/usr/aarch64-linux-gnu"]' aarch64-unknown-linux-gnu
else
    cargo build -q -p cafetensor-bin --target aarch64-unknown-linux-gnu --config 'target.aarch64-unknown-linux-gnu.linker="aarch64-linux-gnu-gcc-13"' \
        || { echo "gate 3: FAIL: aarch64 build"; exit 1; }
    echo "aarch64: builds; run PENDING (qemu-aarch64 not installed)"; pending=1
fi

[ "$pending" = 0 ] && echo "gate 3 (hosts): ok" || echo "gate 3 (hosts): ok with PENDING rows"
