from __future__ import annotations

from dataclasses import dataclass

import torch

from src.backend.config.detector_config import DetectorConfig, DEFAULT_DETECTOR_CONFIG
from src.backend.models.boxes import xywh_to_xyxy


@dataclass(frozen=True)
class EncodedTargets:
    target: torch.Tensor
    positive_mask: torch.Tensor
    collision_count: int


def encode_anchor_free_targets(
    canvas_xywh: torch.Tensor,
    config: DetectorConfig = DEFAULT_DETECTOR_CONFIG,
    *,
    raise_on_collision: bool = True,
) -> EncodedTargets:
    """Encode normalized canvas xywh boxes into the stride-8 S8-CFD target map."""
    boxes = canvas_xywh.to(torch.float32)
    target = torch.zeros((config.num_outputs, config.grid_height, config.grid_width), dtype=torch.float32)
    positive_mask = torch.zeros((config.grid_height, config.grid_width), dtype=torch.bool)
    collision_count = 0

    for box_index, box in enumerate(boxes):
        x, y, w, h = box.tolist()
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and w > 0.0 and h > 0.0):
            raise ValueError(f"Invalid normalized canvas box at index {box_index}: {box.tolist()}")
        cell_x = min(config.grid_width - 1, max(0, int(torch.floor(box[0] * config.grid_width).item())))
        cell_y = min(config.grid_height - 1, max(0, int(torch.floor(box[1] * config.grid_height).item())))
        if positive_mask[cell_y, cell_x]:
            collision_count += 1
            if raise_on_collision:
                raise ValueError(f"Multiple objects assigned to grid cell ({cell_y}, {cell_x})")
            continue
        offset_x = box[0] * config.grid_width - cell_x
        offset_y = box[1] * config.grid_height - cell_y
        target[0, cell_y, cell_x] = 1.0
        target[1, cell_y, cell_x] = offset_x
        target[2, cell_y, cell_x] = offset_y
        target[3, cell_y, cell_x] = torch.log(torch.clamp(box[2], min=1e-8))
        target[4, cell_y, cell_x] = torch.log(torch.clamp(box[3], min=1e-8))
        positive_mask[cell_y, cell_x] = True

    return EncodedTargets(target=target, positive_mask=positive_mask, collision_count=collision_count)


def decode_prediction_map(
    predictions: torch.Tensor,
    config: DetectorConfig = DEFAULT_DETECTOR_CONFIG,
    *,
    apply_sigmoid: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode Bx5xHxW outputs into normalized canvas xyxy boxes and objectness scores."""
    if predictions.ndim != 4 or predictions.shape[1] != config.num_outputs:
        raise ValueError(f"Expected Bx5xHxW predictions, got {tuple(predictions.shape)}")

    batch_size, _, grid_h, grid_w = predictions.shape
    device = predictions.device
    y_idx, x_idx = torch.meshgrid(
        torch.arange(grid_h, device=device, dtype=predictions.dtype),
        torch.arange(grid_w, device=device, dtype=predictions.dtype),
        indexing="ij",
    )
    if apply_sigmoid:
        scores = torch.sigmoid(predictions[:, 0])
        offset_x = torch.sigmoid(predictions[:, 1])
        offset_y = torch.sigmoid(predictions[:, 2])
    else:
        scores = predictions[:, 0]
        offset_x = predictions[:, 1]
        offset_y = predictions[:, 2]

    size_logits = predictions[:, 3:5].clamp(config.size_logit_min, config.size_logit_max)
    width = torch.exp(size_logits[:, 0])
    height = torch.exp(size_logits[:, 1])
    center_x = (x_idx.unsqueeze(0) + offset_x) / grid_w
    center_y = (y_idx.unsqueeze(0) + offset_y) / grid_h
    xywh = torch.stack([center_x, center_y, width, height], dim=-1).reshape(batch_size, -1, 4)
    return xywh_to_xyxy(xywh), scores.reshape(batch_size, -1)


def decode_target_map(
    target: torch.Tensor,
    positive_mask: torch.Tensor,
    config: DetectorConfig = DEFAULT_DETECTOR_CONFIG,
) -> torch.Tensor:
    """Decode positive target cells back to normalized canvas xywh boxes."""
    if target.shape != (config.num_outputs, config.grid_height, config.grid_width):
        raise ValueError(f"Unexpected target shape {tuple(target.shape)}")
    locations = positive_mask.nonzero(as_tuple=False)
    boxes = []
    for cell_y, cell_x in locations:
        x = (cell_x.to(torch.float32) + target[1, cell_y, cell_x]) / config.grid_width
        y = (cell_y.to(torch.float32) + target[2, cell_y, cell_x]) / config.grid_height
        w = torch.exp(target[3, cell_y, cell_x].clamp(config.size_logit_min, config.size_logit_max))
        h = torch.exp(target[4, cell_y, cell_x].clamp(config.size_logit_min, config.size_logit_max))
        boxes.append(torch.stack([x, y, w, h]))
    if not boxes:
        return torch.empty((0, 4), dtype=torch.float32)
    return torch.stack(boxes, dim=0)
