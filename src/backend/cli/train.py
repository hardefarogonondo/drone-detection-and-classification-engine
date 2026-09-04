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

from src.backend.config.detector_config import DEFAULT_DETECTOR_CONFIG
from src.backend.config.settings import AppConfig, load_dotenv, select_device
from src.backend.data.drone_dataset import DroneDetectionDataset, detection_collate
from src.backend.data.preprocessing import build_transform
from src.backend.data.targets import decode_target_map
from src.backend.models.boxes import xywh_to_xyxy
from src.backend.models.checkpoints import load_checkpoint, save_checkpoint
from src.backend.models.inference import postprocess_predictions
from src.backend.models.losses import S8CFDLoss
from src.backend.models.metrics import evaluate_single_class_detections
from src.backend.models.s8_cfd import S8CFD, count_trainable_parameters
from src.backend.models.training_utils import finite_gradients, loss_output_to_float_dict
from src.backend.utils.reproducibility import seed_everything


def _build_loader(split: str, config: AppConfig, *, shuffle: bool) -> DataLoader:
    if split == "test":
        raise ValueError("Trainer must not read the sealed test split.")
    limit = config.train_limit if split == "train" else config.val_limit
    dataset = DroneDetectionDataset(
        Path("data") / "splits" / f"{split}.txt",
        project_root=Path.cwd(),
        config=DEFAULT_DETECTOR_CONFIG,
        limit=limit,
        allow_test=False,
    )
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=detection_collate,
        generator=generator,
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


def _evaluate(model: S8CFD, loader: DataLoader, device: torch.device, config: AppConfig) -> dict[str, float]:
    model.eval()
    pred_boxes_by_image: list[torch.Tensor] = []
    pred_scores_by_image: list[torch.Tensor] = []
    gt_boxes_by_image: list[torch.Tensor] = []
    losses = []
    criterion = S8CFDLoss(DEFAULT_DETECTOR_CONFIG)
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            positive_mask = batch["positive_mask"].to(device)
            predictions = model(images)
            loss = criterion(predictions, targets, positive_mask)
            losses.append(loss_output_to_float_dict(loss))
            detections = postprocess_predictions(
                predictions,
                DEFAULT_DETECTOR_CONFIG,
                confidence_threshold=config.confidence_threshold,
                nms_iou_threshold=config.nms_iou_threshold,
            )
            gt_boxes = _target_boxes_from_batch(batch, device)
            for detection, gt in zip(detections, gt_boxes):
                pred_boxes_by_image.append(detection.boxes_xyxy.detach().cpu())
                pred_scores_by_image.append(detection.scores.detach().cpu())
                gt_boxes_by_image.append(gt.detach().cpu())
    metrics = evaluate_single_class_detections(pred_boxes_by_image, pred_scores_by_image, gt_boxes_by_image)
    mean_losses = {
        f"val_{key}": float(sum(item[key] for item in losses) / max(len(losses), 1))
        for key in ["total", "objectness", "center", "size"]
    }
    return {
        **mean_losses,
        "val_precision": metrics.precision,
        "val_recall": metrics.recall,
        "val_f1": metrics.f1,
        "val_ap50": metrics.ap50,
        "val_ap75": metrics.ap75,
        "val_map50_95": metrics.map50_95,
    }


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    json_path = path.with_suffix(".json")
    json_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    if history:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(history[-1].keys()))
            writer.writeheader()
            writer.writerows(history)


def train(config: AppConfig) -> dict[str, Any]:
    seed_everything(config.seed)
    print("Trainer configuration:")
    print(json.dumps(config.as_log_dict(), indent=2))

    model_path = Path(config.model_path)
    if not config.retrain:
        if not model_path.exists() or not model_path.is_file():
            raise FileNotFoundError(f"RETRAIN=false but MODEL_PATH is not readable: {model_path}")
        with model_path.open("rb"):
            pass
        print(f"RETRAIN=false; verified checkpoint and skipped training: {model_path}")
        return {"skipped": True, "model_path": str(model_path)}

    device = select_device(config.device)
    print(f"Selected device: {device}")

    train_loader = _build_loader("train", config, shuffle=True)
    val_loader = _build_loader("val", config, shuffle=False)
    model = S8CFD(DEFAULT_DETECTOR_CONFIG).to(device)
    _initialize_size_bias_from_train(model, train_loader)
    criterion = S8CFDLoss(DEFAULT_DETECTOR_CONFIG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
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

            wandb_run = wandb.init(project=config.wandb_project, config=config.as_log_dict())
            wandb_run.config.update({"trainable_parameters": count_trainable_parameters(model)})
        except ImportError as exc:
            raise RuntimeError("WANDB_ENABLED=true but wandb is not installed. Install with .[tracking].") from exc

    history: list[dict[str, Any]] = []
    best_metric = float("-inf")
    metric_name = "val_ap50"
    checkpoint_config = config.as_log_dict()
    for epoch in range(start_epoch, config.train_epochs):
        epoch_start = time.time()
        model.train()
        batch_losses = []
        for batch in train_loader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            positive_mask = batch["positive_mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(images)
            loss = criterion(predictions, targets, positive_mask)
            if not torch.isfinite(loss.total):
                raise FloatingPointError("Non-finite training loss detected.")
            loss.total.backward()
            if not finite_gradients(model):
                raise FloatingPointError("Non-finite gradient detected.")
            optimizer.step()
            batch_losses.append(loss_output_to_float_dict(loss))

        train_metrics = {
            f"train_{key}": float(sum(item[key] for item in batch_losses) / max(len(batch_losses), 1))
            for key in ["total", "objectness", "center", "size"]
        }
        val_metrics = _evaluate(model, val_loader, device, config)
        epoch_record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_duration_sec": time.time() - epoch_start,
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
        )
        if epoch_record[metric_name] >= best_metric:
            best_metric = epoch_record[metric_name]
            save_checkpoint(
                model_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metric_name=metric_name,
                metric_value=best_metric,
                config=checkpoint_config,
            )
        if wandb_run is not None:
            wandb_run.log(epoch_record, step=epoch)

    history_path = model_path.parent / "training_history.csv"
    _write_history(history_path, history)
    if wandb_run is not None:
        wandb_run.finish()
    return {"skipped": False, "best_metric": best_metric, "metric_name": metric_name, "model_path": str(model_path)}


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
