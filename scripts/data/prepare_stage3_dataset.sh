#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_DIR="${STAGE3_RAW_DIR:-$ROOT_DIR/datasets/raw/stage3}"
PREPARED_DIR="${STAGE3_PREPARED_DIR:-$ROOT_DIR/datasets/prepared/stage3}"
MANIFEST="${STAGE3_MANIFEST:-$ROOT_DIR/datasets/manifests/stage3-eval.csv}"
DATASET="${STAGE3_DATASET:-lite}"

MAX_MB="${STAGE3_MAX_PREPARED_MB:-2048}"
RAW_DOWNLOAD_MAX_MB="${STAGE3_RAW_DOWNLOAD_MAX_MB:-2048}"
HEIGHT="${STAGE3_HEIGHT:-720}"
FPS="${STAGE3_FPS:-12}"
CLIP_SECONDS="${STAGE3_CLIP_SECONDS:-8}"
STILL_SECONDS="${STAGE3_STILL_SECONDS:-2}"
CRF="${STAGE3_CRF:-28}"
CLEAN="${STAGE3_CLEAN:-1}"
DOWNLOADER="${STAGE3_DOWNLOADER:-auto}"
ARIA2_CONNECTIONS="${STAGE3_ARIA2_CONNECTIONS:-8}"
ARIA2_SPLIT="${STAGE3_ARIA2_SPLIT:-8}"
ARIA2_MIN_SPLIT_SIZE="${STAGE3_ARIA2_MIN_SPLIT_SIZE:-16M}"

VIREDA_URL="${STAGE3_VIREDA_URL:-https://entrepot.recherche.data.gouv.fr/api/access/dataset/:persistentId/?persistentId=doi:10.57745/IZKSF9}"
HAZEBENCH_SAMPLE_URL="${STAGE3_HAZEBENCH_SAMPLE_URL:-https://zenodo.org/api/records/14954622/files/HazyVid%20Samples.zip/content}"
HAZEBENCH_URL="${STAGE3_HAZEBENCH_URL:-https://zenodo.org/api/records/14954622/files/HazyVid.zip/content}"
VIREDA_SIZE_BYTES="${STAGE3_VIREDA_SIZE_BYTES:-8278776237}"
HAZEBENCH_SAMPLE_SIZE_BYTES="${STAGE3_HAZEBENCH_SAMPLE_SIZE_BYTES:-3025220}"
HAZEBENCH_SIZE_BYTES="${STAGE3_HAZEBENCH_SIZE_BYTES:-13697815094}"

CLIP_DIR="$PREPARED_DIR/videos"
INDEX="$PREPARED_DIR/index.csv"

need_tool() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required tool: $1" >&2
    exit 1
  fi
}

download() {
  local url="$1"
  local out="$2"
  local out_dir
  local out_name
  out_dir="$(dirname "$out")"
  out_name="$(basename "$out")"
  mkdir -p "$(dirname "$out")"
  if [[ -s "$out" ]]; then
    echo "exists $out"
    return
  fi
  echo "download $url"

  case "$DOWNLOADER" in
    auto)
      if command -v aria2c >/dev/null 2>&1 && [[ "$url" != file://* ]]; then
        aria2_download "$url" "$out_dir" "$out_name"
      else
        curl_download "$url" "$out"
      fi
      ;;
    aria2)
      if [[ "$url" == file://* ]]; then
        echo "aria2 downloader does not handle file:// smoke-test URLs; use STAGE3_DOWNLOADER=auto or curl" >&2
        exit 1
      fi
      need_tool aria2c
      aria2_download "$url" "$out_dir" "$out_name"
      ;;
    curl)
      need_tool curl
      curl_download "$url" "$out"
      ;;
    *)
      echo "STAGE3_DOWNLOADER must be one of: auto, aria2, curl" >&2
      exit 1
      ;;
  esac
}

download_checked() {
  local url="$1"
  local out="$2"
  local label="$3"
  local size_bytes="$4"
  local max_bytes

  if [[ -s "$out" ]]; then
    download "$url" "$out"
    return
  fi

  max_bytes=$((RAW_DOWNLOAD_MAX_MB * 1024 * 1024))
  if (( RAW_DOWNLOAD_MAX_MB > 0 && size_bytes > max_bytes )); then
    echo "skip $label: archive is about $((size_bytes / 1024 / 1024)) MB, above STAGE3_RAW_DOWNLOAD_MAX_MB=${RAW_DOWNLOAD_MAX_MB}" >&2
    echo "raise STAGE3_RAW_DOWNLOAD_MAX_MB explicitly if you want this full source" >&2
    exit 1
  fi

  download "$url" "$out"
}

aria2_download() {
  local url="$1"
  local out_dir="$2"
  local out_name="$3"
  aria2c \
    --continue=true \
    --max-tries=0 \
    --retry-wait=5 \
    --max-connection-per-server="$ARIA2_CONNECTIONS" \
    --split="$ARIA2_SPLIT" \
    --min-split-size="$ARIA2_MIN_SPLIT_SIZE" \
    --auto-file-renaming=false \
    --allow-overwrite=true \
    --dir="$out_dir" \
    --out="$out_name" \
    "$url"
}

curl_download() {
  local url="$1"
  local out="$2"
  curl -L --fail --continue-at - --output "$out" "$url"
}

extract_zip() {
  local archive="$1"
  local out_dir="$2"
  local marker="$out_dir/.extract-complete"
  mkdir -p "$out_dir"
  if [[ -f "$marker" ]]; then
    echo "extracted $archive"
    return
  fi
  echo "extract $archive"
  unzip -q -n "$archive" -d "$out_dir"
  touch "$marker"
}

prepared_mb() {
  if [[ ! -d "$CLIP_DIR" ]]; then
    echo 0
    return
  fi
  du -sm "$CLIP_DIR" | awk '{print $1}'
}

sanitize_name() {
  basename "${1%.*}" |
    tr '[:upper:]' '[:lower:]' |
    sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//'
}

match_key() {
  sanitize_name "$1" |
    sed -E 's/(hazy|haze|foggy|fog|brume|brouillard|clear|gt|groundtruth|ground_truth|reference|ref|sans|without)//g; s/_+/_/g; s/^_+//; s/_+$//'
}

role_for_path() {
  local lower
  lower="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
  if [[ "$lower" =~ (clear|gt|groundtruth|ground_truth|reference|ref|sans[_-]?fog|without[_-]?fog) ]]; then
    echo "gt"
  else
    echo "hazy"
  fi
}

append_source_note() {
  mkdir -p "$RAW_DIR"
  cat > "$RAW_DIR/SOURCES.md" <<'SOURCES'
# Stage 3 Local Data Sources

This directory is local-only and ignored by Git.

- HazeBench/HazyVid samples, DOI 10.5281/zenodo.14954622, CC-BY-4.0. Default lite source under the raw download budget.
- VIREDA, DOI 10.57745/IZKSF9, Etalab Open License 2.0. Optional paired full source; it is larger than the default raw budget.
- HazeBench/HazyVid full archive, DOI 10.5281/zenodo.14954622, CC-BY-4.0. Optional real-world stress set without clean ground truth; it is larger than the default raw budget.

The repository tracks scripts, documentation, and example manifests only. Raw
archives, extracted media, prepared videos, and generated result videos remain
outside Git.
SOURCES
}

prepare_media() {
  local dataset="$1"
  local source="$2"
  local role="$3"
  local safe
  local key
  local out
  local rel
  local ext
  local used

  used="$(prepared_mb)"
  if (( used >= MAX_MB )); then
    return
  fi

  safe="$(sanitize_name "$source")"
  key="$(match_key "$source")"
  out="$CLIP_DIR/${dataset}__${role}__${safe}_${HEIGHT}p.mp4"
  ext="${source##*.}"
  ext="$(echo "$ext" | tr '[:upper:]' '[:lower:]')"

  if [[ ! -s "$out" ]]; then
    echo "prepare $dataset $role $source"
    if [[ "$ext" =~ ^(jpg|jpeg|png|bmp|tif|tiff)$ ]]; then
      ffmpeg -hide_banner -loglevel error -y \
        -loop 1 -t "$STILL_SECONDS" -i "$source" \
        -vf "scale=-2:'min(${HEIGHT},ih)',fps=${FPS},pad=ceil(iw/2)*2:ceil(ih/2)*2" \
        -an -crf "$CRF" -pix_fmt yuv420p "$out"
    else
      ffmpeg -hide_banner -loglevel error -y \
        -t "$CLIP_SECONDS" -i "$source" \
        -vf "scale=-2:'min(${HEIGHT},ih)',fps=${FPS},pad=ceil(iw/2)*2:ceil(ih/2)*2" \
        -an -crf "$CRF" -pix_fmt yuv420p "$out"
    fi
  fi

  rel="${out#"$ROOT_DIR"/}"
  echo "$role,$key,${dataset}__${safe},$rel" >> "$INDEX"
}

collect_dataset() {
  local dataset="$1"
  local root="$2"
  if [[ ! -d "$root" ]]; then
    echo "missing extracted directory: $root" >&2
    return
  fi

  find "$root" -type f \( \
      -iname '*.mp4' -o -iname '*.mov' -o -iname '*.avi' -o -iname '*.mkv' -o \
      -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o \
      -iname '*.tif' -o -iname '*.tiff' \
    \) |
    sort |
    while read -r media; do
      prepare_media "$dataset" "$media" "$(role_for_path "$media")"
    done
}

clean_prepared_dir() {
  case "$PREPARED_DIR" in
    ""|"/"|"$ROOT_DIR"|"$ROOT_DIR/"|"."|"./")
      echo "refusing unsafe STAGE3_PREPARED_DIR: $PREPARED_DIR" >&2
      exit 1
      ;;
  esac
  rm -rf "$PREPARED_DIR"
}

write_manifest() {
  mkdir -p "$(dirname "$MANIFEST")"
  {
    echo "name,input,gt,method"
    awk -F',' '
      BEGIN { OFS = "," }
      $1 == "gt" && !gt[$2] { gt[$2] = $4 }
      { rows[NR] = $0 }
      END {
        for (i = 1; i <= NR; i++) {
          split(rows[i], row, ",")
          if (row[1] == "hazy") {
            print row[3], row[4], gt[row[2]], "improved-dcp"
          }
        }
      }
    ' "$INDEX"
  } > "$MANIFEST"
}

need_tool unzip
need_tool ffmpeg
need_tool awk

case "$DATASET" in
  lite|vireda|hazebench|both) ;;
  *)
    echo "STAGE3_DATASET must be one of: lite, vireda, hazebench, both" >&2
    exit 1
    ;;
esac

if [[ "$CLEAN" == "1" ]]; then
  clean_prepared_dir
fi

mkdir -p "$RAW_DIR" "$CLIP_DIR"
: > "$INDEX"
append_source_note

if [[ "$DATASET" == "lite" ]]; then
  download_checked "$HAZEBENCH_SAMPLE_URL" "$RAW_DIR/hazebench-samples/HazyVid_Samples.zip" "HazeBench samples" "$HAZEBENCH_SAMPLE_SIZE_BYTES"
  extract_zip "$RAW_DIR/hazebench-samples/HazyVid_Samples.zip" "$RAW_DIR/hazebench-samples/extracted"
  collect_dataset "hazebench_sample" "$RAW_DIR/hazebench-samples/extracted"
fi

if [[ "$DATASET" == "vireda" || "$DATASET" == "both" ]]; then
  download_checked "$VIREDA_URL" "$RAW_DIR/vireda/VIREDA.zip" "VIREDA full" "$VIREDA_SIZE_BYTES"
  extract_zip "$RAW_DIR/vireda/VIREDA.zip" "$RAW_DIR/vireda/extracted"
  collect_dataset "vireda" "$RAW_DIR/vireda/extracted"
fi

if [[ "$DATASET" == "hazebench" || "$DATASET" == "both" ]]; then
  download_checked "$HAZEBENCH_URL" "$RAW_DIR/hazebench/HazyVid.zip" "HazeBench full" "$HAZEBENCH_SIZE_BYTES"
  extract_zip "$RAW_DIR/hazebench/HazyVid.zip" "$RAW_DIR/hazebench/extracted"
  collect_dataset "hazebench" "$RAW_DIR/hazebench/extracted"
fi

write_manifest

echo "wrote $MANIFEST"
echo "prepared budget: $(prepared_mb) MB / ${MAX_MB} MB"
du -sh "$RAW_DIR" "$PREPARED_DIR" 2>/dev/null || true
