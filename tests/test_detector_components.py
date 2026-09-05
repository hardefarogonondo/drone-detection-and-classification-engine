from __future__ import annotations

import math

import torch

from src.backend.config.detector_config import DEFAULT_DETECTOR_CONFIG, DetectorConfig
from src.backend.config.settings import AppConfig
from src.backend.data.loaders import data_loader_kwargs
from src.backend.data.preprocessing import CoordinateTransform
from src.backend.data.targets import decode_prediction_map, decode_target_map, encode_anchor_free_targets
from src.backend.models.boxes import box_iou_xyxy, nms_xyxy, xywh_to_xyxy
from src.backend.models.inference import postprocess_predictions
from src.backend.models.losses import (
    S8CFDLoss,
    aligned_box_iou_xyxy,
    decode_positive_prediction_boxes,
    decode_positive_target_boxes,
    iou_loss_for_positive_predictions,
)
from src.backend.models.metrics import (
    average_precision,
    evaluate_at_confidence_threshold,
    gt_size_category_from_original_dimensions,
    recall_by_original_size_category,
)
from src.backend.models.s8_cfd import S8CFD
from src.backend.models.training_utils import finite_gradients


def _tiny_iou_config(lambda_iou: float = 0.0) -> DetectorConfig:
    return DetectorConfig(input_width=16, content_height=16, input_height=16, stride=8, lambda_iou=lambda_iou)


def _prediction_target_pair(
    target_xywh: torch.Tensor,
    pred_xywh: torch.Tensor,
    config: DetectorConfig,
    *,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = encode_anchor_free_targets(target_xywh.unsqueeze(0), config)
    predictions = torch.zeros((1, config.num_outputs, config.grid_height, config.grid_width), dtype=torch.float32)
    cell_y, cell_x = encoded.positive_mask.nonzero(as_tuple=False)[0].tolist()
    offset_x = pred_xywh[0] * config.grid_width - cell_x
    offset_y = pred_xywh[1] * config.grid_height - cell_y
    assert 0.0 < float(offset_x) < 1.0
    assert 0.0 < float(offset_y) < 1.0
    predictions[0, 1, cell_y, cell_x] = torch.logit(offset_x)
    predictions[0, 2, cell_y, cell_x] = torch.logit(offset_y)
    predictions[0, 3, cell_y, cell_x] = torch.log(pred_xywh[2])
    predictions[0, 4, cell_y, cell_x] = torch.log(pred_xywh[3])
    if requires_grad:
        predictions.requires_grad_(True)
    return predictions, encoded.target.unsqueeze(0), encoded.positive_mask.unsqueeze(0)


def test_resize_padding_coordinate_transform() -> None:
    transform = CoordinateTransform(
        original_width=2560,
        original_height=1440,
        input_width=960,
        content_height=540,
        input_height=544,
    )
    original = torch.tensor([[0.5, 0.5, 0.1, 0.2]], dtype=torch.float32)
    canvas = transform.yolo_xywh_to_canvas_xywh(original)
    expected = torch.tensor([[0.5, 272.0 / 544.0, 0.1, 108.0 / 544.0]], dtype=torch.float32)
    assert torch.allclose(canvas, expected, atol=1e-6)
    roundtrip = transform.canvas_xywh_to_original_xywh(canvas)
    assert torch.allclose(roundtrip, original, atol=1e-6)


def test_target_encoding_numeric_example() -> None:
    config = DEFAULT_DETECTOR_CONFIG
    canvas_box = torch.tensor([[0.552148, 0.357651, 0.017578, 0.037913]], dtype=torch.float32)
    encoded = encode_anchor_free_targets(canvas_box, config)
    assert encoded.collision_count == 0
    assert encoded.positive_mask.sum().item() == 1
    cell_y, cell_x = encoded.positive_mask.nonzero(as_tuple=False)[0].tolist()
    assert (cell_y, cell_x) == (24, 66)
    assert math.isclose(float(encoded.target[1, cell_y, cell_x]), 0.25776, rel_tol=1e-4, abs_tol=1e-4)
    assert math.isclose(float(encoded.target[2, cell_y, cell_x]), 0.320268, rel_tol=1e-4, abs_tol=1e-4)
    assert math.isclose(float(encoded.target[3, cell_y, cell_x]), math.log(0.017578), rel_tol=1e-5)
    assert math.isclose(float(encoded.target[4, cell_y, cell_x]), math.log(0.037913), rel_tol=1e-5)


def test_encode_decode_roundtrip() -> None:
    config = DEFAULT_DETECTOR_CONFIG
    boxes = torch.tensor(
        [
            [0.25, 0.35, 0.02, 0.04],
            [0.80, 0.75, 0.05, 0.08],
        ],
        dtype=torch.float32,
    )
    encoded = encode_anchor_free_targets(boxes, config)
    decoded = decode_target_map(encoded.target, encoded.positive_mask, config)
    decoded = decoded[decoded[:, 0].argsort()]
    boxes = boxes[boxes[:, 0].argsort()]
    assert torch.allclose(decoded, boxes, atol=1e-6)


def test_decode_prediction_map_shape() -> None:
    config = DEFAULT_DETECTOR_CONFIG
    predictions = torch.zeros((2, 5, config.grid_height, config.grid_width), dtype=torch.float32)
    boxes, scores = decode_prediction_map(predictions, config)
    assert boxes.shape == (2, config.grid_height * config.grid_width, 4)
    assert scores.shape == (2, config.grid_height * config.grid_width)
    assert torch.isfinite(boxes).all()
    assert torch.isfinite(scores).all()


def test_iou_identical_disjoint_and_partial() -> None:
    box = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    assert torch.allclose(box_iou_xyxy(box, box), torch.tensor([[1.0]]))

    disjoint = torch.tensor([[2.0, 2.0, 3.0, 3.0]])
    assert torch.allclose(box_iou_xyxy(box, disjoint), torch.tensor([[0.0]]))

    partial = torch.tensor([[0.5, 0.5, 1.5, 1.5]])
    expected = torch.tensor([[0.25 / 1.75]])
    assert torch.allclose(box_iou_xyxy(box, partial), expected, atol=1e-6)


def test_positive_iou_loss_identical_boxes_is_zero() -> None:
    config = _tiny_iou_config(lambda_iou=1.0)
    box = torch.tensor([0.25, 0.25, 0.2, 0.2], dtype=torch.float32)
    predictions, targets, positive_mask = _prediction_target_pair(box, box, config)

    pred_boxes = decode_positive_prediction_boxes(predictions, positive_mask, config)
    target_boxes = decode_positive_target_boxes(targets, positive_mask, config)
    iou = aligned_box_iou_xyxy(pred_boxes, target_boxes)
    loss = iou_loss_for_positive_predictions(predictions, targets, positive_mask, config)

    assert torch.allclose(iou, torch.tensor([1.0]), atol=1e-6)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_positive_iou_loss_non_overlapping_boxes_is_one() -> None:
    config = _tiny_iou_config(lambda_iou=1.0)
    target_box = torch.tensor([0.25, 0.25, 0.1, 0.1], dtype=torch.float32)
    pred_box = torch.tensor([0.45, 0.25, 0.1, 0.1], dtype=torch.float32)
    predictions, targets, positive_mask = _prediction_target_pair(target_box, pred_box, config)

    iou = aligned_box_iou_xyxy(
        decode_positive_prediction_boxes(predictions, positive_mask, config),
        decode_positive_target_boxes(targets, positive_mask, config),
    )
    loss = iou_loss_for_positive_predictions(predictions, targets, positive_mask, config)

    assert torch.allclose(iou, torch.tensor([0.0]), atol=1e-6)
    assert torch.allclose(loss, torch.tensor(1.0), atol=1e-6)


def test_positive_iou_loss_partial_overlap_matches_expected_value() -> None:
    config = _tiny_iou_config(lambda_iou=1.0)
    target_box = torch.tensor([0.25, 0.25, 0.5, 0.5], dtype=torch.float32)
    pred_box = torch.tensor([0.375, 0.25, 0.5, 0.5], dtype=torch.float32)
    predictions, targets, positive_mask = _prediction_target_pair(target_box, pred_box, config)

    iou = aligned_box_iou_xyxy(
        decode_positive_prediction_boxes(predictions, positive_mask, config),
        decode_positive_target_boxes(targets, positive_mask, config),
    )
    loss = iou_loss_for_positive_predictions(predictions, targets, positive_mask, config)

    assert torch.allclose(iou, torch.tensor([0.6]), atol=1e-6)
    assert torch.allclose(loss, torch.tensor(0.4), atol=1e-6)


def test_positive_iou_loss_has_finite_gradients() -> None:
    config = _tiny_iou_config(lambda_iou=1.0)
    target_box = torch.tensor([0.25, 0.25, 0.5, 0.5], dtype=torch.float32)
    pred_box = torch.tensor([0.375, 0.25, 0.5, 0.5], dtype=torch.float32)
    predictions, targets, positive_mask = _prediction_target_pair(target_box, pred_box, config, requires_grad=True)

    loss = iou_loss_for_positive_predictions(predictions, targets, positive_mask, config)
    loss.backward()

    assert torch.isfinite(loss)
    assert predictions.grad is not None
    assert torch.isfinite(predictions.grad).all()


def test_positive_iou_loss_ignores_negative_cells() -> None:
    config = _tiny_iou_config(lambda_iou=1.0)
    target_box = torch.tensor([0.25, 0.25, 0.5, 0.5], dtype=torch.float32)
    pred_box = torch.tensor([0.375, 0.25, 0.5, 0.5], dtype=torch.float32)
    predictions, targets, positive_mask = _prediction_target_pair(target_box, pred_box, config)
    baseline_loss = iou_loss_for_positive_predictions(predictions, targets, positive_mask, config)

    changed_predictions = predictions.clone()
    changed_predictions[0, 1:, 1, 1] = torch.tensor([20.0, -20.0, -8.0, -8.0])
    changed_loss = iou_loss_for_positive_predictions(changed_predictions, targets, positive_mask, config)

    assert torch.allclose(changed_loss, baseline_loss, atol=1e-7)


def test_iou_loss_weight_zero_preserves_baseline_total_loss() -> None:
    config = _tiny_iou_config(lambda_iou=0.0)
    target_box = torch.tensor([0.25, 0.25, 0.5, 0.5], dtype=torch.float32)
    pred_box = torch.tensor([0.375, 0.25, 0.5, 0.5], dtype=torch.float32)
    predictions, targets, positive_mask = _prediction_target_pair(target_box, pred_box, config)

    loss = S8CFDLoss(config)(predictions, targets, positive_mask)
    expected_total = (
        config.lambda_obj * loss.objectness
        + config.lambda_center * loss.center
        + config.lambda_size * loss.size
    )

    assert torch.allclose(loss.iou, torch.tensor(0.0), atol=0.0)
    assert torch.equal(loss.total, expected_total)


def test_iou_loss_weight_positive_contributes_to_total_loss() -> None:
    config = _tiny_iou_config(lambda_iou=1.25)
    target_box = torch.tensor([0.25, 0.25, 0.5, 0.5], dtype=torch.float32)
    pred_box = torch.tensor([0.375, 0.25, 0.5, 0.5], dtype=torch.float32)
    predictions, targets, positive_mask = _prediction_target_pair(target_box, pred_box, config)

    loss = S8CFDLoss(config)(predictions, targets, positive_mask)
    baseline_total = (
        config.lambda_obj * loss.objectness
        + config.lambda_center * loss.center
        + config.lambda_size * loss.size
    )

    assert torch.allclose(loss.iou, torch.tensor(0.4), atol=1e-6)
    assert torch.allclose(loss.total, baseline_total + config.lambda_iou * loss.iou, atol=1e-6)


def test_nms_suppression_and_retention() -> None:
    boxes = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.1, 0.1, 1.1, 1.1],
            [3.0, 3.0, 4.0, 4.0],
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.9, 0.8, 0.7])
    keep = nms_xyxy(boxes, scores, iou_threshold=0.5)
    assert keep.tolist() == [0, 2]


def test_model_output_dimensions() -> None:
    model = S8CFD()
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros((1, 3, 544, 960), dtype=torch.float32))
    assert output.shape == (1, 5, 68, 120)
    assert torch.isfinite(output).all()


def test_finite_loss_and_backward_gradients() -> None:
    config = DetectorConfig()
    model = S8CFD(config)
    criterion = S8CFDLoss(config)
    images = torch.randn((1, 3, config.input_height, config.input_width), dtype=torch.float32)
    boxes = torch.tensor([[0.3, 0.4, 0.02, 0.04], [0.7, 0.6, 0.03, 0.05]], dtype=torch.float32)
    encoded = encode_anchor_free_targets(boxes, config)
    predictions = model(images)
    loss_output = criterion(predictions, encoded.target.unsqueeze(0), encoded.positive_mask.unsqueeze(0))
    assert torch.isfinite(loss_output.total)
    loss_output.total.backward()
    assert finite_gradients(model)


def test_xywh_to_xyxy() -> None:
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.4]], dtype=torch.float32)
    expected = torch.tensor([[0.4, 0.3, 0.6, 0.7]], dtype=torch.float32)
    assert torch.allclose(xywh_to_xyxy(boxes), expected, atol=1e-6)


def test_original_size_category_and_recall_breakdown() -> None:
    assert gt_size_category_from_original_dimensions(15, 100) == "lt16px"
    assert gt_size_category_from_original_dimensions(31, 100) == "16to31px"
    assert gt_size_category_from_original_dimensions(63, 100) == "32to63px"
    assert gt_size_category_from_original_dimensions(64, 100) == "ge64px"

    gt_boxes = [torch.tensor([[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]], dtype=torch.float32)]
    pred_boxes = [torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)]
    pred_scores = [torch.tensor([0.9], dtype=torch.float32)]
    gt_wh = [torch.tensor([[15.0, 20.0], [80.0, 80.0]], dtype=torch.float32)]
    recall = recall_by_original_size_category(pred_boxes, pred_scores, gt_boxes, gt_wh)
    assert recall["lt16px"] == 1.0
    assert recall["ge64px"] == 0.0


def test_ap_uses_ranked_predictions_not_operating_threshold() -> None:
    gt_boxes = [torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)]
    pred_boxes = [torch.tensor([[2.0, 2.0, 3.0, 3.0], [0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)]
    pred_scores = [torch.tensor([0.9, 0.2], dtype=torch.float32)]

    ap_with_all_candidates = average_precision(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5)
    filtered = pred_scores[0] >= 0.3
    ap_after_operating_filter = average_precision([pred_boxes[0][filtered]], [pred_scores[0][filtered]], gt_boxes, iou_threshold=0.5)
    operating = evaluate_at_confidence_threshold(
        pred_boxes,
        pred_scores,
        gt_boxes,
        confidence_threshold=0.3,
    )

    assert ap_with_all_candidates > 0.0
    assert ap_after_operating_filter == 0.0
    assert operating.true_positives == 0.0
    assert operating.false_positives == 1.0


def test_dataloader_cuda_auto_settings() -> None:
    config = AppConfig(num_workers=2)
    kwargs = data_loader_kwargs(config, torch.device("cuda"), shuffle=False)
    assert kwargs["pin_memory"] is True
    assert kwargs["persistent_workers"] is True
    assert kwargs["prefetch_factor"] == 2


def test_dataloader_cpu_default_settings() -> None:
    config = AppConfig(num_workers=0)
    kwargs = data_loader_kwargs(config, torch.device("cpu"), shuffle=False)
    assert kwargs["pin_memory"] is False
    assert "persistent_workers" not in kwargs
    assert "prefetch_factor" not in kwargs


def test_postprocess_prediction_diagnostics() -> None:
    config = DEFAULT_DETECTOR_CONFIG
    predictions = torch.zeros((1, 5, config.grid_height, config.grid_width), dtype=torch.float32)
    results = postprocess_predictions(predictions, config, confidence_threshold=0.25, top_k=10, max_detections=5)
    assert len(results) == 1
    assert results[0].raw_prediction_count == config.grid_height * config.grid_width
    assert results[0].valid_prediction_count == config.grid_height * config.grid_width
    assert results[0].score_filtered_count == config.grid_height * config.grid_width
    assert results[0].nms_candidate_count == 10
    assert results[0].scores.numel() <= 5
