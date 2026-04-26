#!/usr/bin/env bash
# scripts/hf_job.sh
#
# One-shot bootstrap for running JailbreakArena GRPO training inside HF Jobs.
#
# Usage (the typical "hf jobs run" path expects HF_TOKEN to be present in env):
#
#   hf jobs run \
#       --flavor a10g-large \
#       --secrets HF_TOKEN \
#       pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime \
#       bash -c "curl -sL https://raw.githubusercontent.com/GHPRNV/jailbreak-arena/main/scripts/hf_job.sh | bash"
#
# After completion, plots/*.png are uploaded BACK to the HF Space repo so the
# live Space's README shows the real training curves, and the trained model is
# pushed to the Hugging Face Hub.

set -eo pipefail
exec 2>&1
trap 'rc=$?; echo "[hf_job] EXIT rc=$rc at line $LINENO"' EXIT
trap 'rc=$?; echo "[hf_job] ERR  rc=$rc at line $LINENO. cmd: ${BASH_COMMAND}"' ERR
trap 'echo "[hf_job] SIGTERM received"; exit 143' TERM
trap 'echo "[hf_job] SIGINT received"; exit 130' INT

# Helper: run a long pip command with a heartbeat so HF Jobs never thinks
# we're hung during silent dependency resolution.
heartbeat() {
    while true; do
        echo "[heartbeat] $(date -Iseconds) still working..."
        sleep 10
    done
}

# -------- Tunables ---------------------------------------------------------- #
GIT_REPO="${GIT_REPO:-https://github.com/GHPRNV/jailbreak-arena}"
GIT_BRANCH="${GIT_BRANCH:-main}"
SPACE_REPO="${SPACE_REPO:-M134pra/jailbreak-arena}"
HUB_MODEL_ID="${HUB_MODEL_ID:-M134pra/jailbreak-arena-defender-qwen2.5-0.5b}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
DATASET_SIZE="${DATASET_SIZE:-600}"
SELF_PLAY_POOL_SIZE="${SELF_PLAY_POOL_SIZE:-32}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
MAX_TURNS="${MAX_TURNS:-3}"

# ---------------------------------------------------------------------------- #
echo "::group::Environment"
nvidia-smi || echo "(no GPU?)"
python --version
python -m pip --version
df -h /
free -h
echo "::endgroup::"

WORKDIR="${WORKDIR:-/workspace/jailbreak-arena}"
if [ -d "$WORKDIR/.git" ]; then
    echo "[hf_job] reusing existing clone at $WORKDIR"
    cd "$WORKDIR"
else
    rm -rf "$WORKDIR"
    git clone --depth 1 --branch "$GIT_BRANCH" "$GIT_REPO" "$WORKDIR"
    cd "$WORKDIR"
fi

echo "::group::pip install (uv-powered, no vllm)"
# uv is ~10x faster than pip and prints continuously => HF Jobs log heartbeat
# stays alive during resolution. We deliberately DO NOT install vllm here:
# trl[vllm] upgrades torch 2.4 -> 2.10 which conflicts with the base image's
# pre-built CUDA stack and SIGKILLs the container at `import torch`.
heartbeat &
HB=$!
python -m pip install --quiet --no-cache-dir uv
python -m uv pip install --system --no-cache \
    "trl>=0.13.0" "transformers>=4.45" "datasets>=2.20" \
    "accelerate>=0.34" "matplotlib" "huggingface_hub>=0.25" \
    "openenv-core[core]>=0.2.2" "fastapi>=0.115" "pydantic>=2" \
    "uvicorn>=0.24" "fastmcp>=0.1" "pytest>=8"
python -m uv pip install --system --no-cache --no-deps -e .
kill "$HB" 2>/dev/null || true
echo "::endgroup::"

echo "::group::Sanity check (rubric tests, no GPU needed)"
python -m pytest tests/test_rubric.py -q --no-header
echo "::endgroup::"

echo "::group::Train"
mkdir -p outputs/run0
python scripts/train_grpo_defender.py \
    --model-id "$MODEL_ID" \
    --dataset-size "$DATASET_SIZE" \
    --self-play-pool-size "$SELF_PLAY_POOL_SIZE" \
    --num-generations "$NUM_GENERATIONS" \
    --per-device-batch-size "$PER_DEVICE_BATCH_SIZE" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --learning-rate "$LEARNING_RATE" \
    --max-turns "$MAX_TURNS" \
    --vllm-mode off \
    --output-dir outputs/run0 \
    --plots-dir plots \
    2>&1 | tee outputs/run0/train.log
echo "::endgroup::"

echo "::group::Upload trained model to Hub"
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='outputs/run0',
    repo_id='${HUB_MODEL_ID}',
    repo_type='model',
    commit_message='Trained JailbreakArena Defender (GRPO self-play, Qwen2.5-0.5B)',
    allow_patterns=['*.safetensors', '*.json', '*.py', 'tokenizer*', 'merges.txt', 'vocab*', 'special_tokens*'],
)
print('uploaded model ->', '${HUB_MODEL_ID}')
"
echo "::endgroup::"

echo "::group::Upload plots BACK to HF Space"
python scripts/upload_plots_to_space.py \
    --plots-dir plots \
    --space-repo "$SPACE_REPO" \
    --include-outputs outputs/run0 \
    --commit-message "Real training plots from HF Jobs run"
echo "::endgroup::"

echo "[hf_job] DONE. Live Space: https://huggingface.co/spaces/${SPACE_REPO}"
echo "[hf_job] Trained model: https://huggingface.co/${HUB_MODEL_ID}"
