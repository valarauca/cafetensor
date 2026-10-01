#!/usr/bin/env bash
# Gate 3: tier selection and tests on native, SDE-emulated and AArch64 hosts, dev and release.
# Rows whose emulator is missing are reported as PENDING, not passed. Emulated rows check
# selection and SIGILL safety, so their sample tests run on EMULATED_TEST_BYTES (256 KiB).
set -euo pipefail
cd "$(dirname "$0")/.."
pending=0

run_row() {  # name expect runner-json target
    local name="$1" expect="$2" runner="$3" target="$4"
    for pflag in "" "--release"; do
        local cfg=() tgt=()
        [ -n "$runner" ] && cfg=(--config "target.$target.runner=$runner")
        [ "$target" != x86_64-unknown-linux-gnu ] && tgt=(--target "$target") && cfg+=(--config "target.$target.linker=\"aarch64-linux-gnu-gcc-13\"")
        local bytes=()
        [ -n "$runner" ] && bytes=(CAFETENSOR_TEST_BYTES="${EMULATED_TEST_BYTES:-262144}")
        env "${bytes[@]}" CAFETENSOR_EXPECT_TIER="$expect" cargo test -q $pflag "${tgt[@]}" "${cfg[@]}" -p general_backend >/dev/null 2>&1 \
            || { echo "gate 3: FAIL: $name tests ${pflag:-dev}"; exit 1; }
        local out
        out="$(cargo run -q $pflag "${tgt[@]}" "${cfg[@]}" -p cafetensor-bin 2>&1 >/dev/null)" \
            || { echo "gate 3: FAIL: $name smoke ${pflag:-dev}: $out"; exit 1; }
        out="$(grep '^cafetensor: ' <<<"$out" || true)"
        [ "$out" = "cafetensor: tier $expect" ] || { echo "gate 3: FAIL: $name selected '$out', expected $expect"; exit 1; }
    done
    echo "$name: $expect (dev, release)"
}

native="$(cargo run -q -p cafetensor-bin 2>&1 >/dev/null | sed 's/^cafetensor: tier //')"
expect_native="${CAFETENSOR_NATIVE_TIER:-amd64_9800x3d}"
[ "$native" = "$expect_native" ] || { echo "gate 3: FAIL: native selected $native, expected $expect_native"; exit 1; }
run_row native "$expect_native" "" x86_64-unknown-linux-gnu

# QEMU user mode fakes CPUID and raises SIGILL on unsupported instructions. Its TCG emulator has
# no AVX-512, so AVX-512 guests mask those features and must fall back to amd64_v3 cleanly.
# Intel SDE rows (skx/icx/spr) are deferred to a later sprint, see DECISIONS.md.
if command -v qemu-x86_64-static >/dev/null; then
    for row in "qemu64 portable" "Nehalem amd64_v2" "Haswell amd64_v3" "Skylake-Server amd64_v3"; do
        set -- $row
        run_row "qemu -cpu $1" "$2" "[\"qemu-x86_64-static\",\"-cpu\",\"$1\"]" x86_64-unknown-linux-gnu
    done
else
    echo "qemu x86-64 rows: PENDING (qemu-x86_64-static not installed)"; pending=1
fi

if command -v qemu-aarch64-static >/dev/null; then
    run_row aarch64 portable '["qemu-aarch64-static","-L","/usr/aarch64-linux-gnu"]' aarch64-unknown-linux-gnu
else
    cargo build -q -p cafetensor-bin --target aarch64-unknown-linux-gnu --config 'target.aarch64-unknown-linux-gnu.linker="aarch64-linux-gnu-gcc-13"' \
        || { echo "gate 3: FAIL: aarch64 build"; exit 1; }
    echo "aarch64: builds; run PENDING (qemu-aarch64-static not installed)"; pending=1
fi

[ "$pending" = 0 ] && echo "gate 3 (hosts): ok" || echo "gate 3 (hosts): ok with PENDING rows"
