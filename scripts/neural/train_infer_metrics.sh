#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
DEVICE="${DEVICE:-cuda}"
BACKEND="${BACKEND:-gpu}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CROP_SIZE="${CROP_SIZE:-256}"
TRAIN_MAX_SIDE="${TRAIN_MAX_SIDE:-1280}"
WIDTH="${WIDTH:-24}"
LR="${LR:-2e-4}"
LOG_EVERY="${LOG_EVERY:-10}"
MAX_SIDE="${MAX_SIDE:-480}"
MODEL="${MODEL:-models/neural_dehazer.pt}"
RESUME="${RESUME:-}"
SAVE_OPTIMIZER="${SAVE_OPTIMIZER:-1}"
OUT_DIR="${OUT_DIR:-datasets/results/frames/neural-eval}"
METRICS="${METRICS:-datasets/results/metrics/neural_eval.csv}"
TRAIN_LOG="${TRAIN_LOG:-datasets/results/metrics/neural_train_log.csv}"

if ! command -v uv >/dev/null 2>&1; then
  echo "missing uv; install uv first or create .venv manually with PyTorch installed" >&2
  exit 1
fi

mkdir -p models "$OUT_DIR" "$(dirname "$METRICS")"

if [[ ! -x .venv/bin/python ]]; then
  uv python install "$PYTHON_VERSION"
  uv venv --python "$PYTHON_VERSION" .venv
fi

if ! .venv/bin/python - <<'PY' >/dev/null 2>&1
import torch
import torchvision
PY
then
  if [[ "$DEVICE" == "cuda" ]]; then
    uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu128
  else
    uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cpu
  fi
fi

export PATH="$ROOT_DIR/.venv/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

train_args=()
if [[ -n "$RESUME" ]]; then
  train_args+=(--resume "$RESUME")
fi
if [[ "$SAVE_OPTIMIZER" == "1" ]]; then
  train_args+=(--save-optimizer)
fi

.venv/bin/python scripts/neural/train_neural_dehazer.py \
  --hazy-dir datasets/Train/hazy \
  --gt-dir datasets/Train/gt \
  --output "$MODEL" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --crop-size "$CROP_SIZE" \
  --max-side "$TRAIN_MAX_SIDE" \
  --width "$WIDTH" \
  --lr "$LR" \
  --device "$DEVICE" \
  --amp \
  --log-every "$LOG_EVERY" \
  --log-csv "$TRAIN_LOG" \
  "${train_args[@]}"

cargo run --release -p dehaze-cli -- eval-sequences \
  --hazy-dir datasets/Test/hazy \
  --gt-dir datasets/Test/gt \
  --output-dir "$OUT_DIR" \
  --csv "$METRICS" \
  --method neural \
  --backend "$BACKEND" \
  --model "$MODEL" \
  --neural-script scripts/neural/infer_neural_dehazer.py \
  --max-side "$MAX_SIDE"

echo "metrics: $METRICS"
