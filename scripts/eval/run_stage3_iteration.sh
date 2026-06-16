#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HAZY_DIR="${STAGE3_HAZY_DIR:-$ROOT_DIR/datasets/Test/hazy}"
GT_DIR="${STAGE3_GT_DIR:-$ROOT_DIR/datasets/Test/gt}"
OUT_DIR="${1:-$ROOT_DIR/datasets/results/frames/stage3-improved-dcp}"
METRICS="${2:-$ROOT_DIR/datasets/results/metrics/stage3-iteration.csv}"
METHOD="${STAGE3_METHOD:-improved-dcp}"
BACKEND="${STAGE3_BACKEND:-cpu}"
MAX_SIDE="${STAGE3_MAX_SIDE:-480}"
SIMD="${STAGE3_SIMD:-auto}"

mkdir -p "$OUT_DIR" "$(dirname "$METRICS")"
cd "$ROOT_DIR"

cargo run --release -p dehaze-cli -- eval-sequences \
  --hazy-dir "$HAZY_DIR" \
  --gt-dir "$GT_DIR" \
  --output-dir "$OUT_DIR" \
  --csv "$METRICS" \
  --method "$METHOD" \
  --backend "$BACKEND" \
  --max-side "$MAX_SIDE" \
  --simd "$SIMD"

echo "wrote $METRICS"
