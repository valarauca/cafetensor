"""Shared helpers for the gate scripts: tier flags come from the root Cargo.toml only."""
import pathlib
import re
import subprocess
import tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent
TARGET = "x86_64-unknown-linux-gnu"
NO_DETECTION_NAME = {"lahfsahf", "prfchw"}


def manifest():
    return tomllib.loads((ROOT / "Cargo.toml").read_text())


def tier_flags(profile="release"):
    """{tier package: [rustflags]} from [profile.<profile>.package.*]."""
    pkgs = manifest()["profile"][profile]["package"]
    return {name: cfg.get("rustflags", []) for name, cfg in pkgs.items() if name.startswith("amd64_")}


def members():
    """Workspace package names, read from each member manifest."""
    out = []
    for m in manifest()["workspace"]["members"]:
        out.append(tomllib.loads((ROOT / m / "Cargo.toml").read_text())["package"]["name"])
    return out


def derive_features(flags):
    """Target features the flags add over the target baseline, sorted."""
    def cfg(extra):
        text = subprocess.run(
            ["rustc", "--print", "cfg", "--target", TARGET, *extra],
            capture_output=True, text=True, check=True, cwd=ROOT,
        ).stdout
        return set(re.findall(r'target_feature="([^"]+)"', text))
    return sorted(cfg(flags) - cfg([]))


def committed_features(tier):
    return (ROOT / "scripts" / "tier-features" / f"{tier}.txt").read_text().split()


def tier_dir(tier):
    for m in manifest()["workspace"]["members"]:
        if tomllib.loads((ROOT / m / "Cargo.toml").read_text())["package"]["name"] == tier:
            return ROOT / m
    raise KeyError(tier)


def profile_block():
    """The tier override blocks of the root manifest, verbatim."""
    text = (ROOT / "Cargo.toml").read_text()
    start = text.index("# ---- tier flags: release ----")
    return 'cargo-features = ["profile-rustflags"]\n\n' + text[start:].rstrip() + "\n"
