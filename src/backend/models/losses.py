from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from src.backend.config.detector_config import DetectorConfig, DEFAULT_DETECTOR_CONFIG
from src.backend.models.boxes import xywh_to_xyxy


@dataclass(frozen=True)
class DetectionLossOutput:
    total: torch.Tensor
    objectness: torch.Tensor
    objectness_pos: torch.Tensor
    objectness_neg: torch.Tensor
    center: torch.Tensor
    size: torch.Tensor
    iou: torch.Tensor
    num_positive: int


class S8CFDLoss(nn.Module):
    def __init__(self, config: DetectorConfig = DEFAULT_DETECTOR_CONFIG) -> None:
        super().__init__()
        self.config = config

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, positive_mask: torch.Tensor) -> DetectionLossOutput:
        obj_logits = predictions[:, 0]
        obj_target = targets[:, 0]
        bce = F.binary_cross_entropy_with_logits(obj_logits, obj_target, reduction="none")
        pos_mask = positive_mask.bool()
        neg_mask = ~pos_mask

        zero = predictions.sum() * 0.0
        pos_loss = bce[pos_mask].mean() if pos_mask.any() else zero
        neg_loss = bce[neg_mask].mean() if neg_mask.any() else zero
        obj_loss = 0.5 * pos_loss + 0.5 * neg_loss

        if pos_mask.any():
            center_pred = torch.sigmoid(predictions[:, 1:3].permute(0, 2, 3, 1)[pos_mask])
            center_target = targets[:, 1:3].permute(0, 2, 3, 1)[pos_mask]
            center_loss = F.smooth_l1_loss(center_pred, center_target)

            size_pred = predictions[:, 3:5].clamp(self.config.size_logit_min, self.config.size_logit_max).permute(0, 2, 3, 1)[pos_mask]
            size_target = targets[:, 3:5].permute(0, 2, 3, 1)[pos_mask]
            size_loss = F.smooth_l1_loss(size_pred, size_target)
        else:
            center_loss = zero
            size_loss = zero

        iou_loss = zero
        total = (
            self.config.lambda_obj * obj_loss
            + self.config.lambda_center * center_loss
            + self.config.lambda_size * size_loss
        )
        if self.config.lambda_iou > 0.0:
            iou_loss = iou_loss_for_positive_predictions(predictions, targets, pos_mask, self.config)
            total = total + self.config.lambda_iou * iou_loss
        return DetectionLossOutput(
            total=total,
            objectness=obj_loss,
            objectness_pos=pos_loss,
            objectness_neg=neg_loss,
            center=center_loss,
            size=size_loss,
            iou=iou_loss,
            num_positive=int(pos_mask.sum().item()),
        )


def decode_positive_prediction_boxes(
    predictions: torch.Tensor,
    positive_mask: torch.Tensor,
    config: DetectorConfig = DEFAULT_DETECTOR_CONFIG,
) -> torch.Tensor:
    """Decode predicted normalized canvas xyxy boxes for positive assigned cells."""
    pos_mask = positive_mask.to(device=predictions.device).bool()
    if not pos_mask.any():
        return torch.empty((0, 4), dtype=predictions.dtype, device=predictions.device)

    _, cell_y, cell_x = pos_mask.nonzero(as_tuple=True)
    positive_predictions = predictions.permute(0, 2, 3, 1)[pos_mask]
    offset_x = torch.sigmoid(positive_predictions[:, 1])
    offset_y = torch.sigmoid(positive_predictions[:, 2])
    width = torch.exp(positive_predictions[:, 3].clamp(config.size_logit_min, config.size_logit_max))
    height = torch.exp(positive_predictions[:, 4].clamp(config.size_logit_min, config.size_logit_max))
    center_x = (cell_x.to(dtype=predictions.dtype) + offset_x) / config.grid_width
    center_y = (cell_y.to(dtype=predictions.dtype) + offset_y) / config.grid_height
    return xywh_to_xyxy(torch.stack([center_x, center_y, width, height], dim=-1))


def decode_positive_target_boxes(
    targets: torch.Tensor,
    positive_mask: torch.Tensor,
    config: DetectorConfig = DEFAULT_DETECTOR_CONFIG,
) -> torch.Tensor:
    """Decode target normalized canvas xyxy boxes for positive assigned cells."""
    pos_mask = positive_mask.to(device=targets.device).bool()
    if not pos_mask.any():
        return torch.empty((0, 4), dtype=targets.dtype, device=targets.device)

    _, cell_y, cell_x = pos_mask.nonzero(as_tuple=True)
    positive_targets = targets.permute(0, 2, 3, 1)[pos_mask]
    center_x = (cell_x.to(dtype=targets.dtype) + positive_targets[:, 1]) / config.grid_width
    center_y = (cell_y.to(dtype=targets.dtype) + positive_targets[:, 2]) / config.grid_height
    width = torch.exp(positive_targets[:, 3].clamp(config.size_logit_min, config.size_logit_max))
    height = torch.exp(positive_targets[:, 4].clamp(config.size_logit_min, config.size_logit_max))
    return xywh_to_xyxy(torch.stack([center_x, center_y, width, height], dim=-1))


def aligned_box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Aligned IoU for matching Nx4 xyxy box tensors."""
    if boxes1.shape != boxes2.shape:
        raise ValueError(f"Expected matching box shapes, got {tuple(boxes1.shape)} and {tuple(boxes2.shape)}")
    if boxes1.numel() == 0:
        return torch.empty((0,), dtype=boxes1.dtype, device=boxes1.device)

    lt = torch.maximum(boxes1[:, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    intersection = wh[:, 0] * wh[:, 1]

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1 + area2 - intersection
    return intersection / union.clamp(min=eps)


def iou_loss_for_positive_predictions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    positive_mask: torch.Tensor,
    config: DetectorConfig = DEFAULT_DETECTOR_CONFIG,
) -> torch.Tensor:
    """Mean standard IoU loss, 1 - IoU, over positive assigned cells only."""
    pos_mask = positive_mask.to(device=predictions.device).bool()
    if not pos_mask.any():
        return predictions.sum() * 0.0
    pred_boxes = decode_positive_prediction_boxes(predictions, pos_mask, config)
    target_boxes = decode_positive_target_boxes(targets.to(device=predictions.device, dtype=predictions.dtype), pos_mask, config)
    iou = aligned_box_iou_xyxy(pred_boxes, target_boxes)
    return (1.0 - iou).mean()
