from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from src.backend.config.detector_config import DEFAULT_DETECTOR_CONFIG
from src.backend.config.settings import AppConfig, load_dotenv, select_device
from src.backend.data.drone_dataset import DroneDetectionDataset, detection_collate
from src.backend.data.preprocessing import build_transform
from src.backend.data.targets import decode_target_map
from src.backend.models.boxes import xywh_to_xyxy, xyxy_to_xywh
from src.backend.models.checkpoints import load_checkpoint
from src.backend.models.inference import postprocess_predictions
from src.backend.models.metrics import evaluate_single_class_detections
from src.backend.models.s8_cfd import S8CFD, count_trainable_parameters
from src.backend.utils.reproducibility import seed_everything


def _font(size: int = 14) -> ImageFont.ImageFont:
    for candidate in ["Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"]:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _normalized_original_xyxy_to_pixel(boxes_xyxy: torch.Tensor, original_width: int, original_height: int) -> torch.Tensor:
    scale = boxes_xyxy.new_tensor([original_width, original_height, original_width, original_height])
    return boxes_xyxy * scale


def _canvas_xyxy_to_original_pixel_xyxy(
    boxes_canvas_xyxy: torch.Tensor,
    original_width: int,
    original_height: int,
) -> torch.Tensor:
    transform = build_transform(original_width, original_height, DEFAULT_DETECTOR_CONFIG)
    boxes_canvas_xywh = xyxy_to_xywh(boxes_canvas_xyxy.detach().cpu())
    boxes_original_xywh = transform.canvas_xywh_to_original_xywh(boxes_canvas_xywh)
    boxes_original_xyxy = xywh_to_xyxy(boxes_original_xywh)
    return _normalized_original_xyxy_to_pixel(boxes_original_xyxy, original_width, original_height)


def _render_prediction_image(
    image_path: Path,
    gt_boxes_original_xywh: torch.Tensor,
    pred_boxes_canvas_xyxy: torch.Tensor,
    pred_scores: torch.Tensor,
    output_path: Path,
) -> None:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        original_width, original_height = image.size
    draw = ImageDraw.Draw(image)
    font = _font(18)

    gt_xyxy = _normalized_original_xyxy_to_pixel(xywh_to_xyxy(gt_boxes_original_xywh), original_width, original_height)
    pred_xyxy = _canvas_xyxy_to_original_pixel_xyxy(pred_boxes_canvas_xyxy, original_width, original_height)

    for box in gt_xyxy:
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 45, 85), width=4)
        draw.text((x1, max(0, y1 - 22)), "gt drone", fill=(255, 45, 85), font=font)

    for box, score in zip(pred_xyxy, pred_scores.detach().cpu()):
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 190, 90), width=4)
        draw.text((x1, min(original_height - 24, y2 + 4)), f"pred {float(score):.2f}", fill=(0, 150, 70), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Save a smaller view while preserving native-coordinate rendering.
    preview = image.resize((1280, 720), Image.Resampling.LANCZOS)
    preview.save(output_path)


def _write_html_gallery(output_dir: Path, image_files: list[str]) -> None:
    items = "\n".join(
        f'<figure><img src="{name}" alt="{name}"><figcaption>{name}</figcaption></figure>'
        for name in image_files
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Drone Detection Predictions</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #1f2933; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 18px; }}
    figure {{ margin: 0; }}
    img {{ width: 100%; border: 1px solid #cbd5e1; }}
    figcaption {{ font-size: 12px; margin-top: 4px; word-break: break-all; }}
  </style>
</head>
<body>
  <h1>Drone Detection Predictions</h1>
  <p>Pink boxes are ground truth. Green boxes are model predictions.</p>
  <div class="grid">{items}</div>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def run_inference(config: AppConfig) -> dict[str, Any]:
    seed_everything(config.seed)
    if config.inference_split == "test" and not config.allow_test_inference:
        raise RuntimeError("INFERENCE_SPLIT=test requested but ALLOW_TEST_INFERENCE=false. The sealed test split is protected.")

    device = select_device(config.device)
    print(f"Selected device: {device}")
    model_path = Path(config.model_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    dataset = DroneDetectionDataset(
        Path("data") / "splits" / f"{config.inference_split}.txt",
        project_root=Path.cwd(),
        config=DEFAULT_DETECTOR_CONFIG,
        limit=config.inference_limit,
        allow_test=config.allow_test_inference,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.num_workers, collate_fn=detection_collate)
    model = S8CFD(DEFAULT_DETECTOR_CONFIG)
    payload = load_checkpoint(model_path, model, map_location="cpu")
    model.to(device).eval()
    print(f"Loaded checkpoint: {model_path} | epoch={payload.get('epoch')} | parameters={count_trainable_parameters(model):,}")

    output_dir = Path("reports") / "predictions" / config.inference_split
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows = []
    gallery_files = []
    pred_boxes_by_image: list[torch.Tensor] = []
    pred_scores_by_image: list[torch.Tensor] = []
    gt_boxes_by_image: list[torch.Tensor] = []

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            image = batch["image"].to(device)
            predictions = model(image)
            detections = postprocess_predictions(
                predictions,
                DEFAULT_DETECTOR_CONFIG,
                confidence_threshold=config.confidence_threshold,
                nms_iou_threshold=config.nms_iou_threshold,
            )[0]
            gt_canvas_xywh = batch["boxes_canvas_xywh"][0]
            gt_canvas_xyxy = xywh_to_xyxy(gt_canvas_xywh)
            pred_boxes_by_image.append(detections.boxes_xyxy.detach().cpu())
            pred_scores_by_image.append(detections.scores.detach().cpu())
            gt_boxes_by_image.append(gt_canvas_xyxy)

            meta = batch["metadata"][0]
            image_path = Path(meta["image_path"])
            original_width = int(meta["original_width"])
            original_height = int(meta["original_height"])
            pred_pixel_xyxy = _canvas_xyxy_to_original_pixel_xyxy(
                detections.boxes_xyxy,
                original_width,
                original_height,
            )
            for det_index, (box, score) in enumerate(zip(pred_pixel_xyxy, detections.scores.detach().cpu())):
                prediction_rows.append({
                    "image_file": image_path.name,
                    "stem": image_path.stem,
                    "detection_index": det_index,
                    "score": float(score),
                    "x_min_px": float(box[0]),
                    "y_min_px": float(box[1]),
                    "x_max_px": float(box[2]),
                    "y_max_px": float(box[3]),
                })

            if config.save_predictions:
                rendered_name = f"{batch_index:04d}_{image_path.stem}.png"
                _render_prediction_image(
                    image_path,
                    batch["boxes_original_xywh"][0],
                    detections.boxes_xyxy,
                    detections.scores,
                    output_dir / rendered_name,
                )
                gallery_files.append(rendered_name)

    metrics = evaluate_single_class_detections(pred_boxes_by_image, pred_scores_by_image, gt_boxes_by_image)
    metrics_payload = {
        "split": config.inference_split,
        "image_count": len(dataset),
        "checkpoint": str(model_path),
        "confidence_threshold": config.confidence_threshold,
        "nms_iou_threshold": config.nms_iou_threshold,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "ap50": metrics.ap50,
        "ap75": metrics.ap75,
        "map50_95": metrics.map50_95,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["image_file", "stem", "detection_index", "score", "x_min_px", "y_min_px", "x_max_px", "y_max_px"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prediction_rows)
    if gallery_files:
        _write_html_gallery(output_dir, gallery_files)
    print(json.dumps(metrics_payload, indent=2))
    return metrics_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run S8-CFD inference on a frozen manifest split.")
    parser.add_argument("--env-file", default=".env", help="Path to .env file. Environment variables still take precedence.")
    parser.add_argument("--print-config", action="store_true", help="Print resolved configuration and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)
    try:
        config = AppConfig.from_env(load_env_file=False)
        if args.print_config:
            print(json.dumps(config.as_log_dict(), indent=2))
            return
        run_inference(config)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
