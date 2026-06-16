#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GENERATED_INPUT="demo/c005_hazy_input.mp4"
OUTPUT="${2:-demo/c005_neural_dehazed.mp4}"
INPUT="${1:-}"

METHOD="${METHOD:-neural}"
BACKEND="${BACKEND:-auto}"
FPS="${FPS:-6}"
DEMO_REGENERATE_INPUT="${DEMO_REGENERATE_INPUT:-0}"

usage() {
  cat <<'EOF'
Usage:
  scripts/demo_video.sh [input.mp4] [output.mp4]

Demo flow:
  1. Open the hazy input video.
  2. Wait until you close the video window.
  3. Run dehazing.
  4. Open the output video automatically.

Environment:
  METHOD=neural|improved-dcp|original-dcp   default: neural
  BACKEND=auto|gpu|cpu|python-cuda          default: auto
  DEMO_PLAYER="mpv --fs"                    optional custom player command
  FPS=6                                     used only when auto-generating input
  DEMO_REGENERATE_INPUT=1                   rebuild the default input video
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ensure_input_video() {
  if [[ -n "$INPUT" ]]; then
    if [[ ! -f "$INPUT" ]]; then
      printf 'input video not found: %s\n' "$INPUT" >&2
      exit 1
    fi
    return
  fi

  if [[ "$DEMO_REGENERATE_INPUT" != "1" && -f "$GENERATED_INPUT" ]]; then
    INPUT="$GENERATED_INPUT"
    return
  fi

  if ! command -v ffmpeg >/dev/null 2>&1; then
    printf 'ffmpeg is required to generate the fallback demo input video.\n' >&2
    exit 1
  fi

  mkdir -p "$(dirname "$GENERATED_INPUT")"
  printf 'Generating fallback demo input: %s\n' "$GENERATED_INPUT"
  ffmpeg -y \
    -framerate "$FPS" \
    -pattern_type glob \
    -i 'datasets/Test/hazy/C005/*.JPG' \
    -vf 'scale=720:-2,format=yuv420p' \
    -movflags +faststart \
    "$GENERATED_INPUT"
  INPUT="$GENERATED_INPUT"
}

python_bin() {
  if [[ -x .venv/bin/python ]]; then
    printf '%s\n' .venv/bin/python
  else
    printf '%s\n' python3
  fi
}

pick_backend() {
  if [[ "$BACKEND" != "auto" ]]; then
    printf '%s\n' "$BACKEND"
    return
  fi

  if [[ "$METHOD" != "neural" ]]; then
    printf '%s\n' cpu
    return
  fi

  if "$(python_bin)" - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
  then
    printf '%s\n' gpu
  else
    printf '%s\n' cpu
  fi
}

play_and_wait() {
  local file="$1"
  local label="$2"

  printf '\n%s\n%s\n' "$label" "$file"
  printf 'Close the video window to continue.\n'

  if [[ -n "${DEMO_PLAYER:-}" ]]; then
    # shellcheck disable=SC2086
    $DEMO_PLAYER "$file"
  elif command -v mpv >/dev/null 2>&1; then
    mpv --force-window=immediate --keep-open=yes --title="$label" "$file"
  elif command -v ffplay >/dev/null 2>&1; then
    ffplay -window_title "$label" "$file"
  elif command -v vlc >/dev/null 2>&1; then
    vlc --play-and-exit "$file"
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$file" >/dev/null 2>&1 || true
    read -r -p "After closing the video window, press Enter to continue..."
  else
    printf 'No video player found. Open this file manually: %s\n' "$file" >&2
    read -r -p "Press Enter to continue..."
  fi
}

ensure_input_video
BACKEND="$(pick_backend)"
mkdir -p "$(dirname "$OUTPUT")"

play_and_wait "$INPUT" "Input hazy video"

printf '\nRunning dehaze demo...\n'
printf 'METHOD=%s BACKEND=%s\n' "$METHOD" "$BACKEND"
METHOD="$METHOD" BACKEND="$BACKEND" scripts/dehaze.sh video "$INPUT" "$OUTPUT"

play_and_wait "$OUTPUT" "Output dehazed video"

printf '\nDemo complete.\nInput:  %s\nOutput: %s\n' "$INPUT" "$OUTPUT"
