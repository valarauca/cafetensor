#!/usr/bin/env bash
# Gate 6: compare one ported operation with the old project.
#
#   scripts/compare-old.sh <op>
#   scripts/compare-old.sh e2e [checkpoint.safetensors ...]
#
# Builds scripts/compare-harness in a temporary directory outside both repositories, as a
# consumer of this workspace (copying its tier overrides and build flags), runs it on samples
# generated in memory from the committed profiles, prints a markdown table of output equality
# and throughput per tier, then reports inner-loop instruction counts from the harness binary.
# The temporary directory is deleted afterwards.
#
# `e2e` instead builds the old `tcz` (with the old project's own toolchain, into the temporary
# directory) and the new `cafetensor`, then on each local checkpoint compresses with both,
# requires byte-identical containers, decodes each container with the other tool, requires the
# restored file to equal the input, and reports wall times. Both tools run with `-j $E2E_JOBS`
# (default 2). Every file they write lives in the temporary directory.
set -euo pipefail
cd "$(dirname "$0")/.."
op="${1:?usage: compare-old.sh <op>}"
NEW="$PWD"
OLD="${OLD_PROJECT:-$HOME/Documents/rust_stuff/gpu/tensor-compressor}"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/cafetensor-compare-XXXXXX")"
trap 'rm -rf "$tmp"' EXIT

if [ "$op" = e2e ]; then
    shift
    jobs="${E2E_JOBS:-2}"
    first() { ls $1 2>/dev/null | head -1; }
    if [ "$#" -eq 0 ]; then
        set -- \
            "$HOME/models/Qwen3.5-27B/model.safetensors-00002-of-00011.safetensors" \
            "$(first '/srv/hub/models--facebook--sam2.1-hiera-large/snapshots/*/model.safetensors')" \
            "$(first '/srv/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/*/unet/diffusion_pytorch_model.fp16.safetensors')" \
            "$(first '/srv/hub/models--Qwen--Qwen3-VL-8B-Thinking-FP8/snapshots/*/model-00002-of-00002.safetensors')" \
            "$(first '/srv/hub/models--Kijai--WanVideo_comfy_fp8_scaled/snapshots/*/TI2V/Wan2_2-TI2V-5B_fp8_e5m2_scaled_KJ.safetensors')" \
            "$(first '/srv/hub/models--IDEA-Research--grounding-dino-base/snapshots/*/model.safetensors')"
    fi
    (cd "$OLD" && cargo build -q --release --locked --target-dir "$tmp/old-target")
    cargo build -q --release -p cafetensor-bin
    old_bin="$tmp/old-target/release/tcz"
    new_bin="$NEW/target/release/cafetensor"
    now() { date +%s.%N; }
    # Best of two runs per tool, alternating old and new, so page cache and write-back state
    # favour neither. Prints the best wall time in seconds.
    best() {
        local which="$1"; shift
        local t0 t1 b=1e9
        for _ in 1 2; do
            t0=$(now); "$@" >/dev/null 2>&1 || { echo "$which failed: $*" >&2; exit 1; }; t1=$(now)
            b=$(python3 -c "print(min($b, $t1 - $t0))")
        done
        python3 -c "print(f'{$b:.2f}')"
    }
    echo "| checkpoint | bytes | ratio | identical | old compress s | new compress s | old decompress s | new decompress s | old b3sum s | new b3sum s |"
    echo "|---|---|---|---|---|---|---|---|---|---|"
    for f in "$@"; do
        [ -f "$f" ] || { echo "missing checkpoint $f" >&2; exit 1; }
        name="$(basename "$f")"
        oc=$(best old "$old_bin" -j "$jobs" compress "$f" -o "$tmp/old.cafetensor")
        nc=$(best new "$new_bin" -j "$jobs" compress "$f" -o "$tmp/new.cafetensor")
        cmp -s "$tmp/old.cafetensor" "$tmp/new.cafetensor" || { echo "containers differ for $f" >&2; exit 1; }
        od=$(best old "$old_bin" -j "$jobs" decompress "$tmp/new.cafetensor" -o "$tmp/by-old.safetensors" --verify --force)
        nd=$(best new "$new_bin" -j "$jobs" decompress "$tmp/old.cafetensor" -o "$tmp/by-new.safetensors" --verify --force)
        cmp -s "$f" "$tmp/by-old.safetensors" || { echo "old tcz restored $f wrongly" >&2; exit 1; }
        cmp -s "$f" "$tmp/by-new.safetensors" || { echo "cafetensor restored $f wrongly" >&2; exit 1; }
        want="$("$old_bin" -j "$jobs" b3sum "$f" | cut -d' ' -f1)"
        got="$("$new_bin" -j "$jobs" b3sum "$f" 2>/dev/null | cut -d' ' -f1)"
        [ "$want" = "$got" ] || { echo "b3sum differs for $f" >&2; exit 1; }
        ob=$(best old "$old_bin" -j "$jobs" b3sum "$f")
        nb=$(best new "$new_bin" -j "$jobs" b3sum "$f")
        bytes=$(stat -L -c %s "$f")
        ratio=$(python3 -c "print(f'{$(stat -c %s "$tmp/new.cafetensor") / $bytes:.4f}')")
        echo "| $name | $bytes | $ratio | yes | $oc | $nc | $od | $nd | $ob | $nb |"
        rm -f "${tmp:?}"/*.cafetensor "${tmp:?}"/*.safetensors
    done
    shards=()
    for s in /srv/hub/models--Qwen--Qwen3-VL-8B-Thinking-FP8/snapshots/*/model-0000?-of-00002.safetensors; do
        [ -f "$s" ] && shards+=("$s")
    done
    if [ "${#shards[@]}" -eq 2 ]; then
        "$old_bin" -j "$jobs" compress "${shards[@]}" -o "$tmp/old.cafetensor" >/dev/null
        "$new_bin" -j "$jobs" compress "${shards[@]}" -o "$tmp/new.cafetensor" >/dev/null 2>&1
        cmp -s "$tmp/old.cafetensor" "$tmp/new.cafetensor" || { echo "multi-source containers differ" >&2; exit 1; }
        mkdir -p "$tmp/by-old" "$tmp/by-new"
        "$old_bin" -j "$jobs" decompress "$tmp/new.cafetensor" -o "$tmp/by-old" --verify >/dev/null
        "$new_bin" -j "$jobs" decompress "$tmp/old.cafetensor" -o "$tmp/by-new" --verify >/dev/null 2>&1
        for s in "${shards[@]}"; do
            cmp -s "$s" "$tmp/by-old/$(basename "$s")" && cmp -s "$s" "$tmp/by-new/$(basename "$s")" \
                || { echo "multi-source restore of $s differs" >&2; exit 1; }
        done
        echo
        echo "multi-source (Qwen3-VL-8B-Thinking-FP8, 2 shards): identical containers, cross-decoded, restored"
    fi
    exit 0
fi

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
blake3 = "1.8"

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
    "codebook": [],
    "encode": [("old encode_stream", "encode_stream")],
    "decode": [("old decode_lines", "decode_lines")],
    "blake3": [("old blake3_hash_many_avx512", "blake3_hash_many_avx512")],
}[op]
new_syms = {"crc32c": ["13crc32c_update"], "split": ["8split_w1", "8split_w2", "8split_w4", "14pack_bitplanes"], "codebook": ["9histogram"], "encode": ["13encode_stream", "12encode_chunk"], "decode": ["12decode_lines"], "blake3": ["8batch_cv"]}[op]
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
        if new_sym in sym and ("Engine" in sym or "planes" in sym or "algorithms" in sym or "4rans" in sym or "6decode" in sym or "6blake3" in sym):
            tier = next((v for k, v in tiers.items() if k in sym), sym)
            report(f"new {tier} {new_sym.lstrip('0123456789')}", sym)
for label, frag in targets:
    for sym in funcs:
        if frag in sym:
            report(label, sym)
            break
PY
