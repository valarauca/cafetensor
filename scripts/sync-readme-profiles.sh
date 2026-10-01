#!/usr/bin/env bash
# Regenerate the tier override block in cafetensor-lib/README.md from the root Cargo.toml.
# With --check, fail instead of writing when the README is out of date.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 - "${1:-}" <<'PY'
import sys
sys.path.insert(0, "scripts")
from workspace import ROOT, profile_block
readme = ROOT / "cafetensor-lib" / "README.md"
text = readme.read_text()
begin, end = "<!-- BEGIN tier-profiles -->\n```toml\n", "```\n<!-- END tier-profiles -->"
head, rest = text.split(begin, 1)
_, tail = rest.split(end, 1)
new = head + begin + profile_block() + end + tail
if sys.argv[1] == "--check":
    if new != text:
        sys.exit("cafetensor-lib/README.md tier block differs from Cargo.toml; run scripts/sync-readme-profiles.sh")
    print("readme tier block: ok")
else:
    readme.write_text(new)
    print("readme tier block: written")
PY
