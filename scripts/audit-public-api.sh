#!/usr/bin/env bash
# Gate 2: every tier crate exposes exactly one public item, `fn operations`.
set -euo pipefail
cd "$(dirname "$0")/.."
mapfile -t tiers < <(python3 -c 'import sys; sys.path.insert(0,"scripts"); from workspace import tier_flags; print("\n".join(tier_flags()))')
for t in "${tiers[@]}"; do
    if command -v cargo-public-api >/dev/null; then
        # rustdoc never sees profile rustflags, so the flag guard would fire. Pass the tier's
        # flags to rustdoc only; RUSTDOCFLAGS does not affect codegen.
        flags="$(python3 -c "import sys; sys.path.insert(0,'scripts'); from workspace import tier_flags; print(' '.join(tier_flags()['$t']))")"
        api="$(RUSTDOCFLAGS="$flags" cargo public-api --package "$t" 2>/dev/null | grep -v '^$' || true)"
        want="$(printf 'pub mod %s\npub fn %s::operations() -> &'"'"'static dyn generic_operations::Operations' "$t" "$t")"
        if [ "$api" != "$want" ]; then
            echo "gate 2: FAIL: $t public API is:"; echo "$api"; exit 1
        fi
    else
        dir="$(python3 -c "import sys; sys.path.insert(0,'scripts'); from workspace import tier_dir; print(tier_dir('$t'))")"
        others="$(grep -rnE '^\s*pub(\(crate\))?\s' "$dir/src" | grep -v 'pub fn operations() -> &'"'"'static dyn Operations' || true)"
        [ -z "$others" ] || { echo "gate 2: FAIL: $t has other pub items:"; echo "$others"; exit 1; }
    fi
    echo "$t: one public function"
done
echo "gate 2 (public surface): ok"
