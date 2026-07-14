import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DetectionHead(nn.Module):
    def __init__(self, in_dim: int = 768, patch_grid: int = 14):
        super().__init__()
        self.patch_grid = patch_grid

        # project patch tokens from 768 down to 256 channels
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
        )

        # progressive upsampling: 14 → 28 → 56 → 112 → 224
        self.up1 = ConvBlock(256, 128)
        self.up2 = ConvBlock(128, 64)
        self.up3 = ConvBlock(64,  32)
        self.up4 = ConvBlock(32,  16)

        # final 1×1 conv — collapses to single channel heatmap
        self.out = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        B = patch_tokens.shape[0]
        g = self.patch_grid

        # step 1: project 768 → 256
        x = self.proj(patch_tokens)                        # (B, 196, 256)

        # step 2: reshape flat sequence into spatial grid
        x = x.transpose(1, 2).reshape(B, 256, g, g)       # (B, 256, 14, 14)

        # step 3: upsample 4 times, halving channels each time
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up1(x)   # (B, 128, 28, 28)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up2(x)   # (B, 64, 56, 56)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up3(x)   # (B, 32, 112, 112)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up4(x)   # (B, 16, 224, 224)

        # step 4: collapse to single channel — raw logits, no sigmoid yet
        return self.out(x).squeeze(1)   # (B, 224, 224)