#!/usr/bin/env bash
# Regenerate testdata/profiles/ from the local reference checkpoints. Only histograms and
# summary statistics are written; the checkpoints themselves never enter the repository.
# OLD_TCZ points at the old project's release binary for the old_ratio field.
set -euo pipefail
cd "$(dirname "$0")/.."
OLD_TCZ="${OLD_TCZ:-$HOME/Documents/rust_stuff/gpu/tensor-compressor/target/release/tcz}"
first() { ls $1 | head -1; }
cargo build -q --release -p cafetensor-testkit --bin profile-extract
./target/release/profile-extract --old-tcz "$OLD_TCZ" \
    "$HOME/models/Qwen3.5-27B/model.safetensors-00002-of-00011.safetensors" \
    "$(first '/srv/hub/models--facebook--sam2.1-hiera-large/snapshots/*/model.safetensors')" \
    "$(first '/srv/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/*/unet/diffusion_pytorch_model.fp16.safetensors')" \
    "$(first '/srv/hub/models--Qwen--Qwen3-VL-8B-Thinking-FP8/snapshots/*/model-00002-of-00002.safetensors')" \
    "$(first '/srv/hub/models--Kijai--WanVideo_comfy_fp8_scaled/snapshots/*/TI2V/Wan2_2-TI2V-5B_fp8_e5m2_scaled_KJ.safetensors')" \
    "$(first '/srv/hub/models--IDEA-Research--grounding-dino-base/snapshots/*/model.safetensors')"
