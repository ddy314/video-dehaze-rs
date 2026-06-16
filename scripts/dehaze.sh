#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

pick_model() {
  if [[ -n "${MODEL:-}" ]]; then
    printf '%s\n' "$MODEL"
  elif [[ -f models/neural_dehazer_finetune_plus10.pt ]]; then
    printf '%s\n' models/neural_dehazer_finetune_plus10.pt
  elif [[ -f models/neural_dehazer_finetune.pt ]]; then
    printf '%s\n' models/neural_dehazer_finetune.pt
  elif [[ -f models/neural_dehazer.pt ]]; then
    printf '%s\n' models/neural_dehazer.pt
  else
    printf '%s\n' models/neural_dehazer.pt
  fi
}

usage() {
  cat <<'EOF'
Usage:
  scripts/dehaze.sh image <input> <output> [extra dehaze args...]
  scripts/dehaze.sh video <input> <output> [extra dehaze args...]
  scripts/dehaze.sh eval [extra eval-sequences args...]
  scripts/dehaze.sh train

Environment:
  METHOD=neural|improved-dcp|original-dcp   default: neural
  BACKEND=gpu|cpu|python-cuda               default: gpu
  MODEL=models/neural_dehazer_*.pt          default: best existing neural checkpoint
  MAX_SIDE=480                              default eval resize

Examples:
  scripts/dehaze.sh image hazy.jpg out.png
  METHOD=improved-dcp BACKEND=cpu scripts/dehaze.sh video input.mp4 out.mp4
  MODEL=models/neural_dehazer_finetune_plus10.pt scripts/dehaze.sh eval
EOF
}

cmd="${1:-help}"
if [[ "$cmd" == "help" || "$cmd" == "-h" || "$cmd" == "--help" ]]; then
  usage
  exit 0
fi
shift || true

METHOD="${METHOD:-neural}"
BACKEND="${BACKEND:-gpu}"
MODEL_PATH="$(pick_model)"
MAX_SIDE="${MAX_SIDE:-480}"
NEURAL_SCRIPT="${NEURAL_SCRIPT:-scripts/neural/infer_neural_dehazer.py}"

base_args=(--method "$METHOD" --backend "$BACKEND")
if [[ "$METHOD" == "neural" ]]; then
  base_args+=(--model "$MODEL_PATH" --neural-script "$NEURAL_SCRIPT")
fi

case "$cmd" in
  image)
    if [[ $# -lt 2 ]]; then
      usage >&2
      exit 2
    fi
    input="$1"
    output="$2"
    shift 2
    cargo run --release -p dehaze-cli -- image "$input" -o "$output" "${base_args[@]}" "$@"
    ;;
  video)
    if [[ $# -lt 2 ]]; then
      usage >&2
      exit 2
    fi
    input="$1"
    output="$2"
    shift 2
    cargo run --release -p dehaze-cli -- video "$input" -o "$output" "${base_args[@]}" "$@"
    ;;
  eval)
    cargo run --release -p dehaze-cli -- eval-sequences \
      --hazy-dir "${HAZY_DIR:-datasets/Test/hazy}" \
      --gt-dir "${GT_DIR:-datasets/Test/gt}" \
      --output-dir "${OUT_DIR:-datasets/results/frames/${METHOD}-eval}" \
      --csv "${METRICS:-datasets/results/metrics/${METHOD}_eval.csv}" \
      --max-side "$MAX_SIDE" \
      "${base_args[@]}" \
      "$@"
    ;;
  train)
    scripts/neural/train_infer_metrics.sh "$@"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
