#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from model import DcpInspiredDehazer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RGB-only DCP-inspired neural dehazer inference")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=Path)
    input_group.add_argument("--input-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--debug-dir", type=Path)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--max-side", type=int)
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device, width: int) -> DcpInspiredDehazer:
    if not path.exists():
        raise FileNotFoundError(f"neural model not found: {path}")
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "model" in payload:
        width = int(payload.get("width", width))
        state = payload["model"]
    else:
        state = payload
    model = DcpInspiredDehazer(width=width).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def save_debug(debug_dir: Path, outputs: dict[str, torch.Tensor]) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    to_pil_image(outputs["t_hat"][0].detach().cpu()).save(debug_dir / "t_hat.png")
    to_pil_image(outputs["confidence"][0].detach().cpu()).save(debug_dir / "confidence.png")
    to_pil_image(outputs["j_phy"][0].detach().cpu()).save(debug_dir / "j_phy.png")
    residual = (outputs["residual"][0].detach().cpu() * 2.0 + 0.5).clamp(0.0, 1.0)
    to_pil_image(residual).save(debug_dir / "residual.png")
    a_hat = outputs["a_hat"][0, :, 0, 0].detach().cpu().tolist()
    (debug_dir / "airlight.txt").write_text(",".join(f"{v:.6f}" for v in a_hat) + "\n")


def resize_image(img: Image.Image, max_side: int | None) -> Image.Image:
    if not max_side:
        return img
    w, h = img.size
    current = max(w, h)
    if current <= max_side:
        return img
    scale = max_side / current
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return img.resize(size, Image.Resampling.BICUBIC)


def run_one(model: DcpInspiredDehazer, input_path: Path, output_path: Path, device: torch.device, max_side: int | None) -> None:
    img = resize_image(Image.open(input_path).convert("RGB"), max_side)
    hazy = pil_to_tensor(img).float().unsqueeze(0).to(device) / 255.0
    with torch.inference_mode():
        outputs = model(hazy)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    to_pil_image(outputs["clean"][0].detach().cpu()).save(output_path)


def image_files(root: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg"}
    return [p for p in sorted(root.rglob("*")) if p.suffix.lower() in exts]


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested by Rust --backend gpu/python-cuda, but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    model = load_checkpoint(args.model, device, args.width)
    if args.input_dir:
        if not args.output_dir:
            raise ValueError("--output-dir is required with --input-dir")
        files = image_files(args.input_dir)
        print(f"infer files={len(files)} device={device}", flush=True)
        start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        if start and end:
            start.record()
        for idx, path in enumerate(files, start=1):
            rel = path.relative_to(args.input_dir)
            out = args.output_dir / rel.with_suffix(".png")
            run_one(model, path, out, device, args.max_side)
            if idx == 1 or idx % 25 == 0 or idx == len(files):
                print(f"infer step={idx}/{len(files)}", flush=True)
        if start and end:
            end.record()
            torch.cuda.synchronize()
            print(f"infer_cuda_ms={start.elapsed_time(end):.1f}", flush=True)
        return
    if not args.output:
        raise ValueError("--output is required with --input")
    img = resize_image(Image.open(args.input).convert("RGB"), args.max_side)
    hazy = pil_to_tensor(img).float().unsqueeze(0).to(device) / 255.0
    with torch.inference_mode():
        outputs = model(hazy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    to_pil_image(outputs["clean"][0].detach().cpu()).save(args.output)
    if args.debug_dir:
        save_debug(args.debug_dir, outputs)


if __name__ == "__main__":
    main()
