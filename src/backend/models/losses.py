from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from src.backend.config.detector_config import DetectorConfig, DEFAULT_DETECTOR_CONFIG
from src.backend.data.targets import decode_prediction_map
from src.backend.models.boxes import box_iou_xyxy, xywh_to_xyxy


@dataclass(frozen=True)
class DetectionLossOutput:
    total: torch.Tensor
    objectness: torch.Tensor
    objectness_pos: torch.Tensor
    objectness_neg: torch.Tensor
    center: torch.Tensor
    size: torch.Tensor
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

        total = (
            self.config.lambda_obj * obj_loss
            + self.config.lambda_center * center_loss
            + self.config.lambda_size * size_loss
        )
        return DetectionLossOutput(
            total=total,
            objectness=obj_loss,
            objectness_pos=pos_loss,
            objectness_neg=neg_loss,
            center=center_loss,
            size=size_loss,
            num_positive=int(pos_mask.sum().item()),
        )


def iou_loss_for_positive_predictions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    positive_mask: torch.Tensor,
    config: DetectorConfig = DEFAULT_DETECTOR_CONFIG,
) -> torch.Tensor:
    """Optional ablation loss, implemented but not used in the first baseline objective."""
    if not positive_mask.any():
        return predictions.sum() * 0.0
    pred_boxes, _ = decode_prediction_map(predictions, config)
    target_xywh = []
    batch_size = targets.shape[0]
    for batch_index in range(batch_size):
        locations = positive_mask[batch_index].nonzero(as_tuple=False)
        for cell_y, cell_x in locations:
            x = (cell_x.to(torch.float32) + targets[batch_index, 1, cell_y, cell_x]) / config.grid_width
            y = (cell_y.to(torch.float32) + targets[batch_index, 2, cell_y, cell_x]) / config.grid_height
            w = torch.exp(targets[batch_index, 3, cell_y, cell_x])
            h = torch.exp(targets[batch_index, 4, cell_y, cell_x])
            target_xywh.append(torch.stack([x, y, w, h]))
    target_boxes = xywh_to_xyxy(torch.stack(target_xywh).to(predictions.device))
    flat_mask = positive_mask.reshape(batch_size, -1)
    positive_pred_boxes = pred_boxes[flat_mask]
    iou = box_iou_xyxy(positive_pred_boxes, target_boxes).diag()
    return (1.0 - iou).mean()
