from __future__ import annotations

from dataclasses import dataclass

import torch

from src.backend.config.detector_config import DetectorConfig, DEFAULT_DETECTOR_CONFIG
from src.backend.data.targets import decode_prediction_map
from src.backend.models.boxes import nms_xyxy


@dataclass(frozen=True)
class DetectionResult:
    boxes_xyxy: torch.Tensor
    scores: torch.Tensor


def postprocess_predictions(
    predictions: torch.Tensor,
    config: DetectorConfig = DEFAULT_DETECTOR_CONFIG,
    *,
    confidence_threshold: float = 0.25,
    nms_iou_threshold: float = 0.5,
    top_k: int = 200,
    max_detections: int = 50,
) -> list[DetectionResult]:
    """Decode dense predictions and apply confidence filtering plus direct NMS."""
    boxes, scores = decode_prediction_map(predictions, config)
    results: list[DetectionResult] = []
    for image_boxes, image_scores in zip(boxes, scores):
        valid_size = (image_boxes[:, 2] > image_boxes[:, 0]) & (image_boxes[:, 3] > image_boxes[:, 1])
        valid_coord = torch.isfinite(image_boxes).all(dim=1) & torch.isfinite(image_scores)
        keep_mask = valid_size & valid_coord & (image_scores >= confidence_threshold)
        candidate_boxes = image_boxes[keep_mask]
        candidate_scores = image_scores[keep_mask]
        if candidate_boxes.numel() == 0:
            results.append(DetectionResult(candidate_boxes.reshape(0, 4), candidate_scores.reshape(0)))
            continue
        if candidate_scores.numel() > top_k:
            order = candidate_scores.argsort(descending=True)[:top_k]
            candidate_boxes = candidate_boxes[order]
            candidate_scores = candidate_scores[order]
        keep = nms_xyxy(candidate_boxes, candidate_scores, iou_threshold=nms_iou_threshold, max_detections=max_detections)
        results.append(DetectionResult(candidate_boxes[keep], candidate_scores[keep]))
    return results
