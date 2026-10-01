#!/usr/bin/env bash
# Gate 6: compare one ported operation with the old project.
#
#   scripts/compare-old.sh <op>
#
# Builds scripts/compare-harness in a temporary directory outside both repositories, as a
# consumer of this workspace (copying its tier overrides and build flags), runs it on samples
# generated in memory from the committed profiles, prints a markdown table of output equality
# and throughput per tier, then reports inner-loop instruction counts from the harness binary.
# The temporary directory is deleted afterwards.
set -euo pipefail
cd "$(dirname "$0")/.."
op="${1:?usage: compare-old.sh <op>}"
NEW="$PWD"
OLD="${OLD_PROJECT:-$HOME/Documents/rust_stuff/gpu/tensor-compressor}"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/cafetensor-compare-XXXXXX")"
trap 'rm -rf "$tmp"' EXIT

mkdir -p "$tmp/src" "$tmp/.cargo"
cp scripts/compare-harness/main.rs "$tmp/src/main.rs"
cp rust-toolchain.toml "$tmp/"
cp .cargo/config.toml "$tmp/.cargo/config.toml"
python3 - "$tmp" "$NEW" "$OLD" <<'PY'
import sys
sys.path.insert(0, "scripts")
from workspace import profile_block
tmp, new, old = sys.argv[1:]
block = profile_block()
head, tiers = block.split("\n", 1)
open(f"{tmp}/Cargo.toml", "w").write(f"""{head}

[package]
name = "compare-harness"
version = "0.0.0"
edition = "2024"
publish = false

[workspace]

[dependencies]
tensor-compressor = {{ path = "{old}" }}
general_backend = {{ path = "{new}/platform/general_backend" }}
cafetensor-testkit = {{ path = "{new}/testkit" }}

[profile.release]
lto = "fat"
codegen-units = 1
debug = "line-tables-only"
{tiers}""")
PY
(cd "$tmp" && cargo build -q --release)
"$tmp/target/release/compare-harness" "$op"

echo
echo "inner loops (instructions per iteration, from llvm-objdump):"
llvm-objdump -d --no-show-raw-insn --no-leading-addr "$tmp/target/release/compare-harness" > "$tmp/h.dis"
llvm-objdump -d --no-show-raw-insn "$tmp/target/release/compare-harness" > "$tmp/h.adis"
OP="$op" python3 - "$tmp/h.adis" <<'PY'
import os, re, sys
op = os.environ["OP"]
targets = {
    "crc32c": [("old update_vpclmul", "update_vpclmul"), ("old update_hw", "update_hw"), ("old update_table", "update_table")],
    "split": [("old split loop", "old_split"), ("old bitplanes loop", "old_bitplanes")],
}[op]
new_syms = {"crc32c": ["13crc32c_update"], "split": ["8split_w1", "8split_w2", "8split_w4", "14pack_bitplanes"]}[op]
funcs, cur = {}, None
for line in open(sys.argv[1]):
    m = re.match(r"^([0-9a-f]+) <(.+)>:$", line)
    if m:
        cur = m.group(2); funcs[cur] = []; continue
    m = re.match(r"^\s+([0-9a-f]+):\s+(\S+)\s*(.*)$", line)
    if cur and m:
        funcs[cur].append((int(m.group(1), 16), m.group(2), m.group(3)))
def loops(ins):
    out = []
    for i, (addr, mn, ops) in enumerate(ins):
        if mn.startswith("j"):
            t = re.match(r"^(?:0x)?([0-9a-f]+)\b", ops)
            if t and int(t.group(1), 16) < addr:
                start = int(t.group(1), 16)
                out.append(sum(1 for a, _, _ in ins if start <= a <= addr))
    return out
def report(label, sym):
    l = loops(funcs[sym])
    print(f"  {label}: {len(funcs[sym])} instructions, loops {sorted(l)}")
tiers = {"8amd64_v2": "amd64_v2", "8amd64_v3": "amd64_v3", "8amd64_v4": "amd64_v4",
         "12amd64_v4_icl": "amd64_v4_icl", "13amd64_9800x3d": "amd64_9800x3d", "8Portable": "portable",
         "15general_backend": "portable"}
for new_sym in new_syms:
    for sym in sorted(funcs):
        if new_sym in sym and ("Engine" in sym or "planes" in sym):
            tier = next((v for k, v in tiers.items() if k in sym), sym)
            report(f"new {tier} {new_sym.lstrip('0123456789')}", sym)
for label, frag in targets:
    for sym in funcs:
        if frag in sym:
            report(label, sym)
            break
PY
