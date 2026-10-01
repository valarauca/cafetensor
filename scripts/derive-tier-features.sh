#!/usr/bin/env bash
# Rewrite scripts/tier-features/<tier>.txt from the tier rustflags in the root Cargo.toml.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 - <<'PY'
import sys
sys.path.insert(0, "scripts")
from workspace import ROOT, derive_features, tier_flags
for tier, flags in tier_flags("release").items():
    (ROOT / "scripts" / "tier-features" / f"{tier}.txt").write_text("\n".join(derive_features(flags)) + "\n")
    print(tier, " ".join(flags))
PY
