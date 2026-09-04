from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.backend.config.detector_config import DEFAULT_DETECTOR_CONFIG
from src.backend.config.settings import AppConfig, load_dotenv, select_device
from src.backend.data.loaders import build_detection_loader, summarize_loader_settings
from src.backend.data.targets import decode_target_map
from src.backend.models.boxes import xywh_to_xyxy
from src.backend.models.checkpoints import load_checkpoint, save_checkpoint
from src.backend.models.inference import postprocess_predictions
from src.backend.models.losses import S8CFDLoss
from src.backend.models.metrics import evaluate_at_confidence_threshold, evaluate_single_class_detections
from src.backend.models.s8_cfd import S8CFD, count_trainable_parameters
from src.backend.models.training_utils import finite_gradients, loss_output_to_float_dict
from src.backend.utils.reproducibility import seed_everything
from src.backend.utils.runs import ensure_run_directories, split_group_summary, write_json
from src.backend.utils.telemetry import (
    capture_cuda_telemetry,
    cuda_device_name,
    reset_cuda_peak_memory,
    synchronize_if_cuda,
)


def _initialize_size_bias_from_train(model: S8CFD, train_loader: DataLoader) -> None:
    widths = []
    heights = []
    dataset = train_loader.dataset
    for index in range(len(dataset)):
        item = dataset[index]
        widths.extend(item["boxes_canvas_xywh"][:, 2].tolist())
        heights.extend(item["boxes_canvas_xywh"][:, 3].tolist())
    if widths and heights:
        model.initialize_size_bias(float(torch.tensor(widths).median()), float(torch.tensor(heights).median()))


def _target_boxes_from_batch(batch: dict[str, Any], device: torch.device) -> list[torch.Tensor]:
    boxes = []
    for target, mask in zip(batch["target"], batch["positive_mask"]):
        decoded = decode_target_map(target, mask, DEFAULT_DETECTOR_CONFIG)
        boxes.append(xywh_to_xyxy(decoded).to(device))
    return boxes


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    non_blocking = device.type == "cuda"
    return (
        batch["image"].to(device, non_blocking=non_blocking),
        batch["target"].to(device, non_blocking=non_blocking),
        batch["positive_mask"].to(device, non_blocking=non_blocking),
    )


def _progress(iterable: Any, *, description: str, total: int, config: AppConfig) -> Any:
    return tqdm(iterable, desc=description, total=total, dynamic_ncols=True, leave=False, disable=not config.progress)


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    json_path = path.with_suffix(".json")
    json_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    if history:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(history[-1].keys()))
            writer.writeheader()
            writer.writerows(history)


def _run_metadata(config: AppConfig, device: torch.device, model: S8CFD | None = None) -> dict[str, Any]:
    return {
        **config.as_log_dict(),
        "detector_config": DEFAULT_DETECTOR_CONFIG.__dict__,
        "model_name": config.model_name,
        "input_resolution": [DEFAULT_DETECTOR_CONFIG.input_width, DEFAULT_DETECTOR_CONFIG.input_height],
        "stride": DEFAULT_DETECTOR_CONFIG.stride,
        "loss_config": {
            "lambda_obj": DEFAULT_DETECTOR_CONFIG.lambda_obj,
            "lambda_center": DEFAULT_DETECTOR_CONFIG.lambda_center,
            "lambda_size": DEFAULT_DETECTOR_CONFIG.lambda_size,
        },
        "optimizer": "AdamW",
        "selected_device": str(device),
        "cuda_gpu": cuda_device_name(device),
        "trainable_parameters": count_trainable_parameters(model) if model is not None else None,
        "split_grouping": split_group_summary(),
        "best_checkpoint_metric": "val_map50_95",
    }


def _evaluate(model: S8CFD, loader: DataLoader, device: torch.device, config: AppConfig, *, epoch: int) -> dict[str, float]:
    model.eval()
    pred_boxes_by_image: list[torch.Tensor] = []
    pred_scores_by_image: list[torch.Tensor] = []
    gt_boxes_by_image: list[torch.Tensor] = []
    losses = []
    criterion = S8CFDLoss(DEFAULT_DETECTOR_CONFIG)
    diagnostics = {
        "raw_predictions": 0,
        "valid_predictions": 0,
        "eval_score_filtered_predictions": 0,
        "nms_candidate_predictions": 0,
        "post_nms_predictions": 0,
    }
    val_images = 0
    start = time.time()
    with torch.no_grad():
        progress = _progress(loader, description=f"Epoch {epoch + 1}/{config.train_epochs} - val", total=len(loader), config=config)
        for batch in progress:
            images, targets, positive_mask = _move_batch_to_device(batch, device)
            predictions = model(images)
            loss = criterion(predictions, targets, positive_mask)
            loss_dict = loss_output_to_float_dict(loss)
            losses.append(loss_dict)
            detections = postprocess_predictions(
                predictions,
                DEFAULT_DETECTOR_CONFIG,
                confidence_threshold=config.eval_confidence_floor,
                nms_iou_threshold=config.nms_iou_threshold,
                top_k=config.eval_top_k,
                max_detections=config.max_detections,
            )
            gt_boxes = _target_boxes_from_batch(batch, device)
            for detection, gt in zip(detections, gt_boxes):
                pred_boxes_by_image.append(detection.boxes_xyxy.detach().cpu())
                pred_scores_by_image.append(detection.scores.detach().cpu())
                gt_boxes_by_image.append(gt.detach().cpu())
                diagnostics["raw_predictions"] += detection.raw_prediction_count
                diagnostics["valid_predictions"] += detection.valid_prediction_count
                diagnostics["eval_score_filtered_predictions"] += detection.score_filtered_count
                diagnostics["nms_candidate_predictions"] += detection.nms_candidate_count
                diagnostics["post_nms_predictions"] += int(detection.scores.numel())
            val_images += int(images.shape[0])
            progress.set_postfix(
                loss=f"{loss_dict['total']:.4f}",
                obj=f"{loss_dict['objectness']:.4f}",
                center=f"{loss_dict['center']:.4f}",
                size=f"{loss_dict['size']:.4f}",
            )
    synchronize_if_cuda(device)
    duration = time.time() - start
    ranked_metrics = evaluate_single_class_detections(pred_boxes_by_image, pred_scores_by_image, gt_boxes_by_image)
    operating_metrics = evaluate_at_confidence_threshold(
        pred_boxes_by_image,
        pred_scores_by_image,
        gt_boxes_by_image,
        confidence_threshold=config.confidence_threshold,
    )
    mean_losses = {
        f"val_{key}": float(sum(item[key] for item in losses) / max(len(losses), 1))
        for key in ["total", "objectness", "center", "size"]
    }
    return {
        **mean_losses,
        "val_precision": operating_metrics.precision,
        "val_recall": operating_metrics.recall,
        "val_f1": operating_metrics.f1,
        "val_true_positives": operating_metrics.true_positives,
        "val_false_positives": operating_metrics.false_positives,
        "val_false_negatives": operating_metrics.false_negatives,
        "val_ap50": ranked_metrics.ap50,
        "val_ap75": ranked_metrics.ap75,
        "val_map50_95": ranked_metrics.map50_95,
        "val_duration_sec": duration,
        "val_batches_per_sec": len(loader) / max(duration, 1e-8),
        "val_images_per_sec": val_images / max(duration, 1e-8),
        "val_raw_predictions": diagnostics["raw_predictions"],
        "val_valid_predictions": diagnostics["valid_predictions"],
        "val_eval_score_filtered_predictions": diagnostics["eval_score_filtered_predictions"],
        "val_nms_candidate_predictions": diagnostics["nms_candidate_predictions"],
        "val_post_nms_predictions": diagnostics["post_nms_predictions"],
        "val_post_nms_predictions_per_image": diagnostics["post_nms_predictions"] / max(val_images, 1),
    }


def train(config: AppConfig) -> dict[str, Any]:
    seed_everything(config.seed)
    device = select_device(config.device)
    run_dir = config.run_report_dir
    ensure_run_directories(run_dir)
    print("Trainer configuration:")
    print(json.dumps(config.as_log_dict(), indent=2))
    print(f"Selected device: {device}")
    if device.type == "cuda":
        print(f"CUDA GPU: {cuda_device_name(device)}")
    print("DataLoader settings:")
    print(json.dumps(summarize_loader_settings(config, device), indent=2))

    model_path = Path(config.model_path)
    if not config.retrain:
        if not model_path.exists() or not model_path.is_file():
            raise FileNotFoundError(f"RETRAIN=false but MODEL_PATH is not readable: {model_path}")
        with model_path.open("rb"):
            pass
        write_json(run_dir / "config.json", _run_metadata(config, device))
        print(f"RETRAIN=false; verified checkpoint and skipped training: {model_path}")
        return {"skipped": True, "model_path": str(model_path)}

    train_loader = build_detection_loader("train", config, device, shuffle=True, allow_test=False)
    val_loader = build_detection_loader("val", config, device, shuffle=False, allow_test=False)
    model = S8CFD(DEFAULT_DETECTOR_CONFIG).to(device)
    _initialize_size_bias_from_train(model, train_loader)
    criterion = S8CFDLoss(DEFAULT_DETECTOR_CONFIG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    write_json(run_dir / "config.json", _run_metadata(config, device, model))

    start_epoch = 0
    latest_path = model_path.parent / "latest.pt"
    if config.resume and latest_path.exists():
        payload = load_checkpoint(latest_path, model, map_location=device)
        if payload.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_epoch = int(payload.get("epoch", 0)) + 1
        print(f"Resumed from {latest_path} at epoch {start_epoch}")

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    wandb_run = None
    if config.wandb_enabled:
        try:
            import wandb

            wandb_run = wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name or config.run_name,
                tags=list(config.wandb_tags),
                config=_run_metadata(config, device, model),
            )
        except ImportError as exc:
            raise RuntimeError("WANDB_ENABLED=true but wandb is not installed. Install with .[tracking].") from exc

    history: list[dict[str, Any]] = []
    best_metric = float("-inf")
    best_epoch = None
    epochs_without_improvement = 0
    metric_name = "val_map50_95"
    checkpoint_config = _run_metadata(config, device, model)
    for epoch in range(start_epoch, config.train_epochs):
        reset_cuda_peak_memory(device)
        epoch_start = time.time()
        train_start = time.time()
        model.train()
        batch_losses = []
        train_images = 0
        progress = _progress(train_loader, description=f"Epoch {epoch + 1}/{config.train_epochs} - train", total=len(train_loader), config=config)
        for batch in progress:
            images, targets, positive_mask = _move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(images)
            loss = criterion(predictions, targets, positive_mask)
            if not torch.isfinite(loss.total):
                raise FloatingPointError("Non-finite training loss detected.")
            loss.total.backward()
            if not finite_gradients(model):
                raise FloatingPointError("Non-finite gradient detected.")
            optimizer.step()
            loss_dict = loss_output_to_float_dict(loss)
            batch_losses.append(loss_dict)
            train_images += int(images.shape[0])
            progress.set_postfix(
                loss=f"{loss_dict['total']:.4f}",
                obj=f"{loss_dict['objectness']:.4f}",
                center=f"{loss_dict['center']:.4f}",
                size=f"{loss_dict['size']:.4f}",
            )

        synchronize_if_cuda(device)
        train_duration = time.time() - train_start
        train_metrics = {
            f"train_{key}": float(sum(item[key] for item in batch_losses) / max(len(batch_losses), 1))
            for key in ["total", "objectness", "center", "size"]
        }
        val_metrics = _evaluate(model, val_loader, device, config, epoch=epoch)
        synchronize_if_cuda(device)
        cuda_telemetry = capture_cuda_telemetry(device)
        epoch_record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_duration_sec": time.time() - epoch_start,
            "train_duration_sec": train_duration,
            "train_batches_per_sec": len(train_loader) / max(train_duration, 1e-8),
            "train_images_per_sec": train_images / max(train_duration, 1e-8),
            "cuda_gpu": cuda_telemetry.device_name,
            "cuda_peak_allocated_mb": cuda_telemetry.peak_allocated_mb,
            "cuda_peak_reserved_mb": cuda_telemetry.peak_reserved_mb,
            **train_metrics,
            **val_metrics,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, indent=2))
        save_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metric_name=metric_name,
            metric_value=epoch_record[metric_name],
            config=checkpoint_config,
            metadata={"selected_metric": metric_name, "best_epoch": best_epoch, "checkpoint_kind": "latest"},
        )
        improved = epoch_record[metric_name] >= best_metric
        if improved:
            best_metric = epoch_record[metric_name]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                model_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metric_name=metric_name,
                metric_value=best_metric,
                config=checkpoint_config,
                metadata={"selected_metric": metric_name, "best_epoch": best_epoch, "checkpoint_kind": "best"},
            )
        else:
            epochs_without_improvement += 1
        if wandb_run is not None:
            wandb_run.log(epoch_record, step=epoch)
        if config.early_stopping_patience is not None and epochs_without_improvement >= config.early_stopping_patience:
            print(
                f"Early stopping after {epochs_without_improvement} epochs without "
                f"{metric_name} improvement. Best epoch: {best_epoch}."
            )
            break

    history_path = model_path.parent / "training_history.csv"
    _write_history(history_path, history)
    _write_history(run_dir / "metrics" / "training_history.csv", history)
    write_json(run_dir / "metrics" / "best_checkpoint.json", {
        "metric_name": metric_name,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "model_path": str(model_path),
    })
    if wandb_run is not None:
        wandb_run.finish()
    return {"skipped": False, "best_metric": best_metric, "best_epoch": best_epoch, "metric_name": metric_name, "model_path": str(model_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the custom S8-CFD drone detector.")
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
        train(config)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
