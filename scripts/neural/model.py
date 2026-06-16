from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DcpInspiredDehazer(nn.Module):
    """RGB-only neural dehazer with interpretable DCP-inspired heads."""

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.enc1 = ConvBlock(3, width)
        self.enc2 = ConvBlock(width, width * 2)
        self.enc3 = ConvBlock(width * 2, width * 4)
        self.dec2 = ConvBlock(width * 4 + width * 2, width * 2)
        self.dec1 = ConvBlock(width * 2 + width, width)
        self.transmission = nn.Conv2d(width, 1, 3, padding=1)
        self.confidence = nn.Conv2d(width, 1, 3, padding=1)
        self.residual = nn.Conv2d(width, 3, 3, padding=1)
        self.airlight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(width * 4, width, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, hazy: torch.Tensor) -> dict[str, torch.Tensor]:
        e1 = self.enc1(hazy)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        d2 = F.interpolate(e3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        t_hat = torch.sigmoid(self.transmission(d1)).clamp(0.08, 1.0)
        confidence = torch.sigmoid(self.confidence(d1))
        residual = torch.tanh(self.residual(d1)) * 0.25
        a_hat = self.airlight(e3)
        j_phy = a_hat + (hazy - a_hat) / t_hat
        clean = (j_phy + confidence * residual).clamp(0.0, 1.0)
        return {
            "clean": clean,
            "t_hat": t_hat,
            "a_hat": a_hat,
            "confidence": confidence,
            "residual": residual,
            "j_phy": j_phy.clamp(0.0, 1.0),
        }
