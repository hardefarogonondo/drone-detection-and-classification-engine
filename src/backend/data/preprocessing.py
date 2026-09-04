from __future__ import annotations

from dataclasses import dataclass

import torch
import numpy as np
from PIL import Image, ImageOps

from src.backend.config.detector_config import DetectorConfig, DEFAULT_DETECTOR_CONFIG


@dataclass(frozen=True)
class CoordinateTransform:
    """Map boxes between original image coordinates and padded model canvas."""

    original_width: int
    original_height: int
    input_width: int
    content_height: int
    input_height: int

    @property
    def scale_x(self) -> float:
        return self.input_width / self.original_width

    @property
    def scale_y(self) -> float:
        return self.content_height / self.original_height

    @property
    def pad_top(self) -> float:
        return (self.input_height - self.content_height) / 2

    def yolo_xywh_to_canvas_xywh(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert normalized original-image xywh boxes to normalized canvas xywh."""
        if boxes.numel() == 0:
            return boxes.reshape(0, 4)
        out = boxes.to(torch.float32).clone()
        out[:, 0] = boxes[:, 0]
        out[:, 1] = (boxes[:, 1] * self.content_height + self.pad_top) / self.input_height
        out[:, 2] = boxes[:, 2]
        out[:, 3] = boxes[:, 3] * self.content_height / self.input_height
        return out

    def canvas_xywh_to_original_xywh(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert normalized canvas xywh boxes back to normalized original-image xywh."""
        if boxes.numel() == 0:
            return boxes.reshape(0, 4)
        out = boxes.to(torch.float32).clone()
        out[:, 0] = boxes[:, 0]
        out[:, 1] = (boxes[:, 1] * self.input_height - self.pad_top) / self.content_height
        out[:, 2] = boxes[:, 2]
        out[:, 3] = boxes[:, 3] * self.input_height / self.content_height
        return out


def build_transform(
    original_width: int,
    original_height: int,
    config: DetectorConfig = DEFAULT_DETECTOR_CONFIG,
) -> CoordinateTransform:
    return CoordinateTransform(
        original_width=original_width,
        original_height=original_height,
        input_width=config.input_width,
        content_height=config.content_height,
        input_height=config.input_height,
    )


def preprocess_image(image: Image.Image, config: DetectorConfig = DEFAULT_DETECTOR_CONFIG) -> torch.Tensor:
    """Convert RGB/RGBA PIL input to a normalized CHW tensor on the padded canvas."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((config.input_width, config.content_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (config.input_width, config.input_height), (0, 0, 0))
    canvas.paste(image, (0, int(config.pad_top)))
    data = torch.from_numpy(np.asarray(canvas, dtype="float32")).permute(2, 0, 1)
    return data / 255.0
