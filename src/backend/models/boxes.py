from __future__ import annotations

import torch


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    out = boxes.clone()
    out[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    out[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    out[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
    out[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
    return out


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    out = boxes.clone()
    out[..., 0] = (boxes[..., 0] + boxes[..., 2]) / 2
    out[..., 1] = (boxes[..., 1] + boxes[..., 3]) / 2
    out[..., 2] = boxes[..., 2] - boxes[..., 0]
    out[..., 3] = boxes[..., 3] - boxes[..., 1]
    return out


def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Pairwise IoU for xyxy boxes."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=boxes1.dtype, device=boxes1.device)

    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=eps)


def nms_xyxy(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5, max_detections: int | None = None) -> torch.Tensor:
    """Pure PyTorch non-maximum suppression returning kept indices."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        current = order[0]
        keep.append(current)
        if max_detections is not None and len(keep) >= max_detections:
            break
        if order.numel() == 1:
            break
        ious = box_iou_xyxy(boxes[current].unsqueeze(0), boxes[order[1:]]).squeeze(0)
        order = order[1:][ious <= iou_threshold]
    return torch.stack(keep).to(torch.long)
