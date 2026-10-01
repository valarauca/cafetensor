#!/usr/bin/env bash
# Gate 5: in the release binary, every function using ymm/zmm/k registers or VEX/EVEX
# instructions must belong to a tier allowed them, judged from its raw v0-mangled symbol.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build -q --release -p cafetensor-bin
llvm-objdump -d --no-show-raw-insn target/release/cafetensor > target/cafetensor.dis
python3 - target/cafetensor.dis <<'PY'
import re, sys
# Tier identifiers as they appear length-prefixed in v0 symbols, with what each may use.
ALLOW = {
    "8amd64_v3": {"vex", "ymm"},
    "8amd64_v4": {"vex", "ymm", "zmm", "k"},
    "12amd64_v4_icl": {"vex", "ymm", "zmm", "k"},
    "13amd64_9800x3d": {"vex", "ymm", "zmm", "k"},
}
VEX_GPR = {"andn", "bextr", "blsi", "blsmsk", "blsr", "bzhi", "mulx", "pdep", "pext", "rorx", "sarx", "shlx", "shrx"}
def classes(line):
    parts = line.split("\t")
    if len(parts) < 2:
        return set()
    mnem = parts[1].split()[0] if parts[1].split() else ""
    ops = parts[2] if len(parts) > 2 else ""
    c = set()
    if (mnem.startswith("v") and mnem not in ("verr", "verw")) or mnem in VEX_GPR:
        c.add("vex")
    if "%ymm" in ops: c.add("ymm")
    if "%zmm" in ops: c.add("zmm")
    if re.search(r"%k[0-7]\b", ops): c.add("k")
    return c
used, sym = {}, None
for line in open(sys.argv[1]):
    m = re.match(r"^[0-9a-f]+ <(.+)>:$", line)
    if m:
        sym = m.group(1); continue
    if sym:
        c = classes(line)
        if c:
            used.setdefault(sym, set()).update(c)
bad = []
for sym, c in sorted(used.items()):
    allowed = set().union(*[a for k, a in ALLOW.items() if k in sym]) if any(k in sym for k in ALLOW) else set()
    if not c <= allowed:
        bad.append(f"{sym}: uses {sorted(c - allowed)}")
print(f"functions with vector extensions: {len(used)}")
for sym in sorted(used):
    print(f"  {sorted(used[sym])} {sym}")
if bad:
    sys.exit("gate 5: FAIL:\n  " + "\n  ".join(bad))
print("gate 5 (disassembly): ok")
PY
