from __future__ import annotations

from dataclasses import dataclass

import torch

from src.backend.models.boxes import box_iou_xyxy


@dataclass(frozen=True)
class DetectionMetrics:
    precision: float
    recall: float
    f1: float
    ap50: float
    ap75: float
    map50_95: float


@dataclass(frozen=True)
class OperatingPointMetrics:
    precision: float
    recall: float
    f1: float
    true_positives: float
    false_positives: float
    false_negatives: float
    predictions: float


def gt_size_category_from_original_dimensions(width_px: float, height_px: float) -> str:
    """Classify GT size using original 2560x1440 dimensions for ablation-invariant reporting."""
    min_dim = min(width_px, height_px)
    if min_dim < 16:
        return "lt16px"
    if min_dim < 32:
        return "16to31px"
    if min_dim < 64:
        return "32to63px"
    return "ge64px"


def match_detections(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    order = pred_scores.argsort(descending=True)
    pred_boxes = pred_boxes[order]
    pred_scores = pred_scores[order]
    matched_gt = torch.zeros((gt_boxes.shape[0],), dtype=torch.bool, device=gt_boxes.device)
    tp = torch.zeros((pred_boxes.shape[0],), dtype=torch.float32, device=pred_boxes.device)
    fp = torch.zeros_like(tp)
    if gt_boxes.numel() == 0:
        fp[:] = 1
        return tp, fp
    ious = box_iou_xyxy(pred_boxes, gt_boxes)
    for pred_index in range(pred_boxes.shape[0]):
        best_iou, best_gt = ious[pred_index].max(dim=0)
        if best_iou >= iou_threshold and not matched_gt[best_gt]:
            tp[pred_index] = 1
            matched_gt[best_gt] = True
        else:
            fp[pred_index] = 1
    return tp, fp


def recall_by_original_size_category(
    pred_boxes_by_image: list[torch.Tensor],
    pred_scores_by_image: list[torch.Tensor],
    gt_boxes_by_image: list[torch.Tensor],
    gt_original_wh_by_image: list[torch.Tensor],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute recall by GT size category while preserving original-resolution thresholds."""
    totals = {"lt16px": 0, "16to31px": 0, "32to63px": 0, "ge64px": 0}
    matched = {key: 0 for key in totals}
    for pred_boxes, pred_scores, gt_boxes, gt_wh in zip(
        pred_boxes_by_image,
        pred_scores_by_image,
        gt_boxes_by_image,
        gt_original_wh_by_image,
    ):
        tp, _ = match_detections(pred_boxes, pred_scores, gt_boxes, iou_threshold)
        order = pred_scores.argsort(descending=True)
        gt_matched = torch.zeros((gt_boxes.shape[0],), dtype=torch.bool, device=gt_boxes.device)
        if pred_boxes.numel() and gt_boxes.numel():
            ious = box_iou_xyxy(pred_boxes[order], gt_boxes)
            for pred_index in range(pred_boxes.shape[0]):
                best_iou, best_gt = ious[pred_index].max(dim=0)
                if best_iou >= iou_threshold and not gt_matched[best_gt]:
                    gt_matched[best_gt] = True
        for gt_index, wh in enumerate(gt_wh):
            category = gt_size_category_from_original_dimensions(float(wh[0]), float(wh[1]))
            totals[category] += 1
            matched[category] += int(gt_matched[gt_index].item())
    return {
        category: matched[category] / totals[category] if totals[category] else 0.0
        for category in totals
    }



def average_precision(
    pred_boxes_by_image: list[torch.Tensor],
    pred_scores_by_image: list[torch.Tensor],
    gt_boxes_by_image: list[torch.Tensor],
    iou_threshold: float,
) -> float:
    rows = []
    total_gt = sum(boxes.shape[0] for boxes in gt_boxes_by_image)
    if total_gt == 0:
        return 0.0
    for image_index, (boxes, scores) in enumerate(zip(pred_boxes_by_image, pred_scores_by_image)):
        for box, score in zip(boxes, scores):
            rows.append((float(score.item()), image_index, box))
    if not rows:
        return 0.0
    rows.sort(key=lambda item: item[0], reverse=True)
    matched = [torch.zeros((gt.shape[0],), dtype=torch.bool, device=gt.device) for gt in gt_boxes_by_image]
    tp = []
    fp = []
    for _, image_index, box in rows:
        gt = gt_boxes_by_image[image_index]
        if gt.numel() == 0:
            tp.append(0.0)
            fp.append(1.0)
            continue
        ious = box_iou_xyxy(box.unsqueeze(0), gt).squeeze(0)
        best_iou, best_gt = ious.max(dim=0)
        if best_iou >= iou_threshold and not matched[image_index][best_gt]:
            tp.append(1.0)
            fp.append(0.0)
            matched[image_index][best_gt] = True
        else:
            tp.append(0.0)
            fp.append(1.0)
    tp_cum = torch.tensor(tp).cumsum(0)
    fp_cum = torch.tensor(fp).cumsum(0)
    recalls = tp_cum / total_gt
    precisions = tp_cum / (tp_cum + fp_cum).clamp(min=1e-8)
    recall_grid = torch.linspace(0, 1, 101)
    precision_samples = []
    for recall_level in recall_grid:
        valid = precisions[recalls >= recall_level]
        precision_samples.append(valid.max() if valid.numel() else torch.tensor(0.0))
    return float(torch.stack(precision_samples).mean().item())


def evaluate_single_class_detections(
    pred_boxes_by_image: list[torch.Tensor],
    pred_scores_by_image: list[torch.Tensor],
    gt_boxes_by_image: list[torch.Tensor],
    iou_threshold_for_pr: float = 0.5,
) -> DetectionMetrics:
    total_tp = 0.0
    total_fp = 0.0
    total_gt = sum(boxes.shape[0] for boxes in gt_boxes_by_image)
    for boxes, scores, gt_boxes in zip(pred_boxes_by_image, pred_scores_by_image, gt_boxes_by_image):
        tp, fp = match_detections(boxes, scores, gt_boxes, iou_threshold_for_pr)
        total_tp += float(tp.sum().item())
        total_fp += float(fp.sum().item())
    precision = total_tp / max(total_tp + total_fp, 1e-8)
    recall = total_tp / max(total_gt, 1e-8)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    thresholds = [round(float(t), 2) for t in torch.arange(0.50, 1.00, 0.05)]
    aps = {
        threshold: average_precision(pred_boxes_by_image, pred_scores_by_image, gt_boxes_by_image, threshold)
        for threshold in thresholds
    }
    return DetectionMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        ap50=aps[0.50],
        ap75=aps[0.75],
        map50_95=sum(aps.values()) / len(aps),
    )


def evaluate_at_confidence_threshold(
    pred_boxes_by_image: list[torch.Tensor],
    pred_scores_by_image: list[torch.Tensor],
    gt_boxes_by_image: list[torch.Tensor],
    *,
    confidence_threshold: float,
    iou_threshold_for_pr: float = 0.5,
) -> OperatingPointMetrics:
    total_tp = 0.0
    total_fp = 0.0
    total_predictions = 0.0
    total_gt = sum(boxes.shape[0] for boxes in gt_boxes_by_image)
    for boxes, scores, gt_boxes in zip(pred_boxes_by_image, pred_scores_by_image, gt_boxes_by_image):
        keep = scores >= confidence_threshold
        filtered_boxes = boxes[keep]
        filtered_scores = scores[keep]
        tp, fp = match_detections(filtered_boxes, filtered_scores, gt_boxes, iou_threshold_for_pr)
        total_tp += float(tp.sum().item())
        total_fp += float(fp.sum().item())
        total_predictions += float(filtered_scores.numel())
    false_negatives = max(float(total_gt) - total_tp, 0.0)
    precision = total_tp / max(total_tp + total_fp, 1e-8)
    recall = total_tp / max(total_gt, 1e-8)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return OperatingPointMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=total_tp,
        false_positives=total_fp,
        false_negatives=false_negatives,
        predictions=total_predictions,
    )
