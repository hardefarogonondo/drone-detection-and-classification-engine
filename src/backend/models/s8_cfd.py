from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from src.backend.config.detector_config import DetectorConfig, DEFAULT_DETECTOR_CONFIG


def _num_groups(channels: int, preferred: int = 8) -> int:
    return preferred if channels % preferred == 0 else 1


class ConvGNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int = 3, stride: int = 1, activation: bool = True) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.norm = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = ConvGNAct(channels, channels)
        self.conv2 = ConvGNAct(channels, channels, activation=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.conv2(self.conv1(x)))


class S8CFD(nn.Module):
    """Stride-8 context-fused drone detector built from primitive PyTorch layers."""

    def __init__(self, config: DetectorConfig = DEFAULT_DETECTOR_CONFIG) -> None:
        super().__init__()
        self.config = config

        self.stem = ConvGNAct(3, 24, stride=2)
        self.stage2_down = ConvGNAct(24, 48, stride=2)
        self.stage2_blocks = nn.Sequential(ResidualBlock(48))
        self.stage3_down = ConvGNAct(48, 96, stride=2)
        self.stage3_blocks = nn.Sequential(ResidualBlock(96), ResidualBlock(96))
        self.stage4_down = ConvGNAct(96, 160, stride=2)
        self.stage4_blocks = nn.Sequential(ResidualBlock(160), ResidualBlock(160))
        self.stage5_down = ConvGNAct(160, 224, stride=2)
        self.stage5_blocks = nn.Sequential(ResidualBlock(224))

        self.proj8 = ConvGNAct(96, 96, kernel_size=1)
        self.proj16 = ConvGNAct(160, 96, kernel_size=1)
        self.proj32 = ConvGNAct(224, 96, kernel_size=1)
        self.fuse = ConvGNAct(96 * 3, 128)
        self.head = nn.Sequential(
            ConvGNAct(128, 128),
            nn.Conv2d(128, config.num_outputs, kernel_size=1),
        )
        self._initialize_head_biases()

    def _initialize_head_biases(self) -> None:
        final = self.head[-1]
        if isinstance(final, nn.Conv2d) and final.bias is not None:
            nn.init.constant_(final.bias, 0.0)
            final.bias.data[0] = -6.0

    def initialize_size_bias(self, median_width_norm: float, median_height_norm: float) -> None:
        """Optionally initialize log-size outputs from train-set box statistics."""
        final = self.head[-1]
        if not isinstance(final, nn.Conv2d) or final.bias is None:
            return
        final.bias.data[3] = torch.log(torch.tensor(float(median_width_norm)).clamp(min=1e-8))
        final.bias.data[4] = torch.log(torch.tensor(float(median_height_norm)).clamp(min=1e-8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage2_blocks(self.stage2_down(x))
        p8 = self.stage3_blocks(self.stage3_down(x))
        p16 = self.stage4_blocks(self.stage4_down(p8))
        p32 = self.stage5_blocks(self.stage5_down(p16))

        p8_proj = self.proj8(p8)
        p16_proj = F.interpolate(self.proj16(p16), size=p8_proj.shape[-2:], mode="bilinear", align_corners=False)
        p32_proj = F.interpolate(self.proj32(p32), size=p8_proj.shape[-2:], mode="bilinear", align_corners=False)
        fused = torch.cat([p8_proj, p16_proj, p32_proj], dim=1)
        return self.head(self.fuse(fused))


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
