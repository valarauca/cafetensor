#!/usr/bin/env bash
# Gate 1: tier flags reach exactly the tier crates, every rustc line carries
# -Zshare-generics=n, and the derived lists, flag guards, detection and README agree.
set -euo pipefail
cd "$(dirname "$0")/.."

fail() { echo "audit-flags: FAIL: $*" >&2; exit 1; }

for v in RUSTFLAGS CARGO_ENCODED_RUSTFLAGS; do
    [ -z "${!v+x}" ] || fail "$v is set"
done
if env | grep -qE '^CARGO_TARGET_[A-Z0-9_]+_RUSTFLAGS='; then fail "a CARGO_TARGET_*_RUSTFLAGS variable is set"; fi

dir="$PWD"
configs=()
while :; do
    for f in "$dir/.cargo/config.toml" "$dir/.cargo/config"; do [ -f "$f" ] && configs+=("$f"); done
    [ "$dir" = / ] && break
    dir="$(dirname "$dir")"
done
for f in "${CARGO_HOME:-$HOME/.cargo}/config.toml" "${CARGO_HOME:-$HOME/.cargo}/config"; do [ -f "$f" ] && configs+=("$f"); done
for f in "${configs[@]}"; do
    if grep -qE 'target-cpu|target-feature' "$f"; then fail "$f mentions target-cpu or target-feature"; fi
done

python3 - <<'PY'
import re, sys
sys.path.insert(0, "scripts")
from workspace import (ROOT, NO_DETECTION_NAME, committed_features, derive_features,
                       tier_dir, tier_flags)
errors = []
release, dev = tier_flags("release"), tier_flags("dev")
if release != dev:
    errors.append(f"release and dev tier flags differ: {release} vs {dev}")
detect = (ROOT / "platform/general_backend/src/detect.rs").read_text()
fns = dict(re.findall(r"pub fn (\w+)\(\) -> bool \{(.*?)\n\}", detect, re.S))
for tier, flags in release.items():
    derived = derive_features(flags)
    if derived != sorted(committed_features(tier)):
        errors.append(f"{tier}: derived features differ from scripts/tier-features/{tier}.txt")
    src = (tier_dir(tier) / "src" / "lib.rs").read_text()
    guard = re.search(r"#\[cfg\(not\(all\((.*?)\)\)\)\]\s*compile_error!", src, re.S)
    guarded = sorted(re.findall(r'target_feature = "([^"]+)"', guard.group(1))) if guard else []
    if guarded != derived:
        errors.append(f"{tier}: flag guard does not list exactly the derived features")
    if len(re.findall(r"target_feature\s*=", src)) != len(guarded) or "cfg!(target_feature" in src:
        errors.append(f"{tier}: cfg(target_feature) used outside the flag guard")
    if tier not in fns:
        errors.append(f"{tier}: no detection function")
        continue
    checked, f = set(), tier
    while f:
        body = fns[f]
        checked |= set(re.findall(r'is_x86_feature_detected!\("([^"]+)"\)', body))
        nxt = re.search(r"^\s*(amd64_\w+)\(\)", body, re.M)
        f = nxt.group(1) if nxt else None
    want = set(derived) - NO_DETECTION_NAME
    if checked != want:
        errors.append(f"{tier}: detection {sorted(checked ^ want)} differs from the flag set")
if errors:
    sys.exit("audit-flags: FAIL:\n  " + "\n  ".join(errors))
print("feature lists, guards, detection: ok")
PY

scripts/sync-readme-profiles.sh --check

mapfile -t members < <(python3 -c 'import sys; sys.path.insert(0,"scripts"); from workspace import members; print("\n".join(members()))')
mapfile -t tiers < <(python3 -c 'import sys; sys.path.insert(0,"scripts"); from workspace import tier_flags; print("\n".join(tier_flags()))')

for profile in dev release; do
    pflag=(); [ "$profile" = release ] && pflag=(--release)
    for m in "${members[@]}"; do cargo clean -q "${pflag[@]}" -p "$m" 2>/dev/null || true; done
    log="$(mktemp)"
    cargo build -v --color never "${pflag[@]}" -p cafetensor-bin -p general_backend --all-targets >"$log" 2>&1 || { cat "$log"; fail "build ($profile)"; }
    TIERS="${tiers[*]}" PROFILE="$profile" python3 - "$log" <<'PY'
import os, re, shlex, sys
sys.path.insert(0, "scripts")
from workspace import tier_flags
flags = tier_flags(os.environ["PROFILE"])
lines = [l for l in open(sys.argv[1]) if "Running `" in l and "rustc" in l and "--crate-name" in l]
errors, seen = [], set()
for l in lines:
    cmd = l.split("Running `", 1)[1].rsplit("`", 1)[0]
    args = shlex.split(cmd)
    crate = args[args.index("--crate-name") + 1]
    cpu = [a for a in args if a.startswith(("-Ctarget-cpu", "-Ctarget-feature"))]
    cpu += [args[i] + args[i + 1] for i, a in enumerate(args) if a == "-C" and args[i + 1].startswith(("target-cpu", "target-feature"))]
    if "-Zshare-generics=n" not in args:
        errors.append(f"{crate}: missing -Zshare-generics=n")
    if crate in flags:
        seen.add(crate)
        if cpu != flags[crate]:
            errors.append(f"{crate}: flags {cpu}, expected {flags[crate]}")
    elif cpu:
        errors.append(f"{crate}: carries CPU flags {cpu}")
missing = set(flags) - seen
if missing:
    errors.append(f"tier crates not rebuilt: {sorted(missing)}")
if errors:
    sys.exit(f"audit-flags ({os.environ['PROFILE']}): FAIL:\n  " + "\n  ".join(errors))
print(f"rustc lines ({os.environ['PROFILE']}): {len(lines)} checked, ok")
PY
    rm -f "$log"
done
echo "gate 1 (flag audit): ok"
