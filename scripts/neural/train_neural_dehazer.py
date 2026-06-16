#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor

from model import DcpInspiredDehazer


class PairedImages(Dataset):
    def __init__(self, hazy_dir: Path, gt_dir: Path, crop_size: int, max_side: int | None) -> None:
        self.hazy_dir = hazy_dir
        self.gt_dir = gt_dir
        self.crop_size = crop_size
        self.max_side = max_side
        self.items = [p.relative_to(hazy_dir) for p in sorted(hazy_dir.rglob("*")) if p.suffix.lower() in {".png", ".jpg", ".jpeg"} and (gt_dir / p.relative_to(hazy_dir)).exists()]
        if not self.items:
            raise RuntimeError(f"no paired images under {hazy_dir} and {gt_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rel = self.items[idx]
        hazy = Image.open(self.hazy_dir / rel).convert("RGB")
        gt = Image.open(self.gt_dir / rel).convert("RGB")
        hazy, gt = resize_pair(hazy, gt, self.max_side)
        hazy, gt = random_crop_pair(hazy, gt, self.crop_size)
        return pil_to_tensor(hazy).float() / 255.0, pil_to_tensor(gt).float() / 255.0


def resize_pair(hazy: Image.Image, gt: Image.Image, max_side: int | None) -> tuple[Image.Image, Image.Image]:
    if hazy.size != gt.size:
        gt = gt.resize(hazy.size, Image.Resampling.BICUBIC)
    if not max_side:
        return hazy, gt
    w, h = hazy.size
    current = max(w, h)
    if current <= max_side:
        return hazy, gt
    scale = max_side / current
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return hazy.resize(size, Image.Resampling.BICUBIC), gt.resize(size, Image.Resampling.BICUBIC)


def random_crop_pair(hazy: Image.Image, gt: Image.Image, crop_size: int) -> tuple[Image.Image, Image.Image]:
    if crop_size <= 0:
        return hazy, gt
    w, h = hazy.size
    if w < crop_size or h < crop_size:
        scale = crop_size / min(w, h)
        size = (max(crop_size, round(w * scale)), max(crop_size, round(h * scale)))
        hazy = hazy.resize(size, Image.Resampling.BICUBIC)
        gt = gt.resize(size, Image.Resampling.BICUBIC)
        w, h = hazy.size
    x = random.randint(0, w - crop_size)
    y = random.randint(0, h - crop_size)
    box = (x, y, x + crop_size, y + crop_size)
    return hazy.crop(box), gt.crop(box)


def gradient_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    gt_dx = gt[..., :, 1:] - gt[..., :, :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    gt_dy = gt[..., 1:, :] - gt[..., :-1, :]
    return F.l1_loss(pred_dx, gt_dx) + F.l1_loss(pred_dy, gt_dy)


def ssim_like_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    mu_x = F.avg_pool2d(pred, 7, stride=1, padding=3)
    mu_y = F.avg_pool2d(gt, 7, stride=1, padding=3)
    sigma_x = F.avg_pool2d(pred * pred, 7, stride=1, padding=3) - mu_x * mu_x
    sigma_y = F.avg_pool2d(gt * gt, 7, stride=1, padding=3) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * gt, 7, stride=1, padding=3) - mu_x * mu_y
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2))
    return 1.0 - ssim.clamp(0.0, 1.0).mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RGB-only DCP-inspired neural dehazer")
    parser.add_argument("--hazy-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/neural_dehazer.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--max-side", type=int, default=1280)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--log-csv", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--save-optimizer", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dataset = PairedImages(args.hazy_dir, args.gt_dir, crop_size=args.crop_size, max_side=args.max_side)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    start_epoch = 1
    checkpoint = None
    if args.resume:
        if not args.resume.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        if isinstance(checkpoint, dict):
            args.width = int(checkpoint.get("width", args.width))
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
    model = DcpInspiredDehazer(width=args.width).to(device)
    if checkpoint is not None:
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if isinstance(checkpoint, dict) and "optimizer" in checkpoint:
        opt.load_state_dict(checkpoint["optimizer"])
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    if isinstance(checkpoint, dict) and "scaler" in checkpoint and args.amp and device.type == "cuda":
        scaler.load_state_dict(checkpoint["scaler"])
    print(
        f"train samples={len(dataset)} batches_per_epoch={len(loader)} epochs={args.epochs} start_epoch={start_epoch} "
        f"batch_size={args.batch_size} crop_size={args.crop_size} width={args.width} "
        f"device={device} amp={args.amp and device.type == 'cuda'}",
        flush=True,
    )
    if args.log_csv:
        args.log_csv.parent.mkdir(parents=True, exist_ok=True)
        if not args.log_csv.exists() or start_epoch == 1:
            args.log_csv.write_text(
                "epoch,avg_loss,elapsed_s,lr,batch_size,crop_size,width,device,amp\n"
            )
    final_epoch = start_epoch + args.epochs - 1
    for epoch in range(start_epoch, final_epoch + 1):
        epoch_start = time.perf_counter()
        total = 0.0
        for step, (hazy, gt) in enumerate(loader, start=1):
            step_start = time.perf_counter()
            hazy = hazy.to(device)
            gt = gt.to(device)
            with torch.autocast(device_type="cuda", enabled=args.amp and device.type == "cuda"):
                out = model(hazy)
                clean = out["clean"]
                color = (clean.mean(dim=(-2, -1)) - gt.mean(dim=(-2, -1))).abs().mean()
                loss = F.l1_loss(clean, gt) + 0.2 * ssim_like_loss(clean, gt) + 0.1 * gradient_loss(clean, gt) + 0.05 * color
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_value = float(loss.detach().cpu())
            total += loss_value
            if args.log_every > 0 and (step == 1 or step % args.log_every == 0 or step == len(loader)):
                elapsed = time.perf_counter() - epoch_start
                step_s = time.perf_counter() - step_start
                avg_s = elapsed / step
                eta_s = avg_s * (len(loader) - step + (final_epoch - epoch) * len(loader))
                if device.type == "cuda":
                    mem_gb = torch.cuda.max_memory_allocated(device) / 1024**3
                    mem_text = f" gpu_mem={mem_gb:.2f}GB"
                else:
                    mem_text = ""
                print(
                    f"epoch={epoch}/{final_epoch} step={step}/{len(loader)} "
                    f"loss={loss_value:.6f} avg_loss={total / step:.6f} "
                    f"step_s={step_s:.2f} eta_min={eta_s / 60:.1f}{mem_text}",
                    flush=True,
                )
        avg_loss = total / max(1, len(loader))
        elapsed_s = time.perf_counter() - epoch_start
        print(f"epoch={epoch} done avg_loss={avg_loss:.6f} elapsed_s={elapsed_s:.1f}", flush=True)
        if args.log_csv:
            with args.log_csv.open("a") as f:
                f.write(
                    f"{epoch},{avg_loss:.8f},{elapsed_s:.3f},{args.lr:.8g},{args.batch_size},{args.crop_size},{args.width},{device},{args.amp and device.type == 'cuda'}\n"
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "width": args.width, "epoch": final_epoch}
    if args.save_optimizer:
        payload["optimizer"] = opt.state_dict()
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, args.output)
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
