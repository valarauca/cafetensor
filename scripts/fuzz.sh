#!/usr/bin/env bash
# Run fuzz targets for a number of seconds each, from the root workspace so the tier overrides
# apply.
#
#   scripts/fuzz.sh [seconds] [target ...]     default: 60 seconds, every target
#
# cargo-fuzz sets RUSTFLAGS itself, which is the one exemption from the RUSTFLAGS rule (see
# DECISIONS.md). Fat LTO drops the sanitizer coverage counters at link time, so fuzz builds
# turn it off through the profile environment. Corpora and crashes stay under fuzz/, which
# git ignores.
set -euo pipefail
cd "$(dirname "$0")/.."
secs="${1:-60}"
shift || true
targets=("$@")
[ "${#targets[@]}" -gt 0 ] || mapfile -t targets < <(cargo fuzz list)
export CARGO_PROFILE_RELEASE_LTO=false
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
for t in "${targets[@]}"; do
    mkdir -p "fuzz/corpus/$t"
    echo "fuzz: $t for ${secs}s"
    cargo fuzz run "$t" "fuzz/corpus/$t" -- -max_total_time="$secs" -rss_limit_mb=4096 -print_final_stats=1 2>&1 \
        | grep -E '^(stat::number_of_executed_units|stat::peak_rss_mb|==[0-9]+==|SUMMARY|thread .* panicked|Failing input|Done )' || true
    [ "${PIPESTATUS[0]}" = 0 ] || { echo "fuzz: $t FAILED"; exit 1; }
done
echo "fuzz: ok"
