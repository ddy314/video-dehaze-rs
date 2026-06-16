# video-dehaze-rs

Rust image/video dehazing project with three clear routes:

- `original-dcp`: plain Dark Channel Prior baseline for explanation and ablation.
- `improved-dcp`: Rust CPU traditional route with robust airlight, sky/highlight protection, CAP-assisted transmission, temporal stabilization, and SIMD-friendly operators.
- `neural`: RGB-only DCP-inspired PyTorch route. Rust handles IO and video assembly; Python/PyTorch handles neural inference and CUDA.

The current recommended final demo path is `neural` with the finetuned checkpoint under `models/`.

## Quick Use

Small wrapper:

```bash
scripts/dehaze.sh image input.jpg output.png
scripts/dehaze.sh video input.mp4 output.mp4
scripts/dehaze.sh eval
```

Traditional CPU baseline:

```bash
METHOD=original-dcp BACKEND=cpu scripts/dehaze.sh image input.jpg original.png
METHOD=improved-dcp BACKEND=cpu scripts/dehaze.sh video input.mp4 improved.mp4
```

Neural GPU route:

```bash
MODEL=models/neural_dehazer_finetune_plus10.pt scripts/dehaze.sh image input.jpg neural.png
MODEL=models/neural_dehazer_finetune_plus10.pt scripts/dehaze.sh video input.mp4 neural.mp4
```

Direct CLI is still available:

```bash
cargo run --release -p dehaze-cli -- image input.jpg -o output.png \
  --method neural \
  --backend gpu \
  --model models/neural_dehazer_finetune_plus10.pt
```

## Evaluation

Default evaluation:

```bash
MODEL=models/neural_dehazer_finetune_plus10.pt scripts/dehaze.sh eval
```

Full command:

```bash
cargo run --release -p dehaze-cli -- eval-sequences \
  --hazy-dir datasets/Test/hazy \
  --gt-dir datasets/Test/gt \
  --output-dir datasets/results/frames/neural-eval \
  --csv datasets/results/metrics/neural_eval.csv \
  --method neural \
  --backend gpu \
  --model models/neural_dehazer_finetune_plus10.pt \
  --max-side 480
```

Metrics CSV includes:

- `input_psnr`, `input_ssim`
- `output_psnr`, `output_ssim`
- `color_delta`
- `flicker`
- `elapsed_ms`
- `method`, `backend`

Latest local neural result on `datasets/Test`:

```text
input_psnr=15.205  input_ssim=0.6608
output_psnr=18.803 output_ssim=0.8570
color_delta=0.0952 flicker=0.0599
elapsed_ms=93/frame
```

## Training

One-command training and evaluation:

```bash
scripts/neural/train_infer_metrics.sh
```

Continue from a checkpoint:

```bash
RESUME=models/neural_dehazer_finetune.pt \
MODEL=models/neural_dehazer_finetune_plus10.pt \
EPOCHS=10 \
LR=2e-5 \
scripts/neural/train_infer_metrics.sh
```

The training script uses crop training, AMP on CUDA, checkpoint resume, and CSV logging.

## Project Layout

- `crates/dehaze-core/src/lib.rs`: public Rust API exports.
- `crates/dehaze-core/src/dcp.rs`: original/improved DCP, temporal stabilization, SIMD-friendly filters.
- `crates/dehaze-core/src/metrics.rs`: PSNR, SSIM, color delta, flicker.
- `crates/dehaze-cli/src/main.rs`: CLI arguments and top-level dispatch.
- `crates/dehaze-cli/src/runtime.rs`: image/video IO, ffmpeg, evaluation loops, Python neural backend.
- `scripts/dehaze.sh`: lightweight user-facing wrapper.
- `scripts/neural/`: PyTorch model, training, and inference.

## Validation

```bash
cargo check
cargo test
cargo run -p dehaze-cli -- video --help
bash -n scripts/dehaze.sh scripts/neural/train_infer_metrics.sh
python3 -m py_compile scripts/neural/*.py
```
