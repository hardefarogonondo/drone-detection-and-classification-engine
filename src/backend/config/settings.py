from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def load_dotenv(path: str | Path = ".env") -> None:
    """Small .env loader to avoid requiring python-dotenv at runtime."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be boolean-like, got {value!r}")


def _int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def _optional_int(name: str) -> int | None:
    value = os.getenv(name)
    return None if value is None or value == "" else int(value)


def _float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None or value == "" else float(value)


def _optional_float(name: str) -> float | None:
    value = os.getenv(name)
    return None if value is None or value == "" else float(value)


def _csv_list(name: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class AppConfig:
    run_name: str = "s8-cfd-dev"
    retrain: bool = True
    device: str = "auto"
    model_name: str = "s8_cfd"
    model_path: Path = Path("models/checkpoints/s8-cfd-dev/best.pt")
    train_epochs: int = 50
    batch_size: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    pin_memory: str = "auto"
    persistent_workers: str = "auto"
    prefetch_factor: int | None = 2
    train_limit: int | None = None
    val_limit: int | None = None
    resume: bool = False
    lr_scheduler: str = "none"
    early_stopping_patience: int | None = None
    inference_split: str = "val"
    inference_limit: int | None = 20
    confidence_threshold: float = 0.30
    eval_confidence_floor: float = 0.001
    nms_iou_threshold: float = 0.50
    eval_top_k: int = 500
    max_detections: int = 100
    save_predictions: bool = True
    threshold_analysis: bool = False
    threshold_min: float = 0.0
    threshold_max: float = 0.95
    threshold_step: float = 0.05
    progress: bool = True
    wandb_enabled: bool = False
    wandb_project: str = "drone-detection-and-classification-engine"
    wandb_run_name: str | None = None
    wandb_tags: tuple[str, ...] = ()
    allow_test_inference: bool = False
    seed: int = 42

    @classmethod
    def from_env(cls, *, load_env_file: bool = True) -> "AppConfig":
        if load_env_file:
            load_dotenv()
        run_name = os.getenv("RUN_NAME", cls.run_name).strip() or cls.run_name
        model_path_value = os.getenv("MODEL_PATH")
        model_path = (
            Path(model_path_value)
            if model_path_value
            else Path("models") / "checkpoints" / run_name / "best.pt"
        )
        config = cls(
            run_name=run_name,
            retrain=_bool("RETRAIN", cls.retrain),
            device=os.getenv("DEVICE", cls.device),
            model_name=os.getenv("MODEL_NAME", cls.model_name),
            model_path=model_path,
            train_epochs=_int("TRAIN_EPOCHS", cls.train_epochs),
            batch_size=_int("BATCH_SIZE", cls.batch_size),
            learning_rate=_float("LEARNING_RATE", cls.learning_rate),
            weight_decay=_float("WEIGHT_DECAY", cls.weight_decay),
            num_workers=_int("NUM_WORKERS", cls.num_workers),
            pin_memory=os.getenv("PIN_MEMORY", cls.pin_memory).strip().lower(),
            persistent_workers=os.getenv("PERSISTENT_WORKERS", cls.persistent_workers).strip().lower(),
            prefetch_factor=_optional_int("PREFETCH_FACTOR") if os.getenv("PREFETCH_FACTOR") is not None else cls.prefetch_factor,
            train_limit=_optional_int("TRAIN_LIMIT"),
            val_limit=_optional_int("VAL_LIMIT"),
            resume=_bool("RESUME", cls.resume),
            lr_scheduler=os.getenv("LR_SCHEDULER", cls.lr_scheduler),
            early_stopping_patience=_optional_int("EARLY_STOPPING_PATIENCE"),
            inference_split=os.getenv("INFERENCE_SPLIT", cls.inference_split),
            inference_limit=_optional_int("INFERENCE_LIMIT") if os.getenv("INFERENCE_LIMIT") is not None else cls.inference_limit,
            confidence_threshold=_float("CONFIDENCE_THRESHOLD", cls.confidence_threshold),
            eval_confidence_floor=_float("EVAL_CONFIDENCE_FLOOR", cls.eval_confidence_floor),
            nms_iou_threshold=_float("NMS_IOU_THRESHOLD", cls.nms_iou_threshold),
            eval_top_k=_int("EVAL_TOP_K", cls.eval_top_k),
            max_detections=_int("MAX_DETECTIONS", cls.max_detections),
            save_predictions=_bool("SAVE_PREDICTIONS", cls.save_predictions),
            threshold_analysis=_bool("THRESHOLD_ANALYSIS", cls.threshold_analysis),
            threshold_min=_float("THRESHOLD_MIN", cls.threshold_min),
            threshold_max=_float("THRESHOLD_MAX", cls.threshold_max),
            threshold_step=_float("THRESHOLD_STEP", cls.threshold_step),
            progress=_bool("PROGRESS", cls.progress),
            wandb_enabled=_bool("WANDB_ENABLED", cls.wandb_enabled),
            wandb_project=os.getenv("WANDB_PROJECT", cls.wandb_project),
            wandb_run_name=os.getenv("WANDB_RUN_NAME") or None,
            wandb_tags=_csv_list("WANDB_TAGS", cls.wandb_tags),
            allow_test_inference=_bool("ALLOW_TEST_INFERENCE", cls.allow_test_inference),
            seed=_int("SEED", cls.seed),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.model_name != "s8_cfd":
            raise ValueError("Only MODEL_NAME=s8_cfd is currently implemented.")
        if self.device not in {"auto", "cuda", "mps", "cpu"}:
            raise ValueError("DEVICE must be one of: auto, cuda, mps, cpu.")
        if self.train_epochs < 0:
            raise ValueError("TRAIN_EPOCHS must be non-negative.")
        if self.batch_size < 1:
            raise ValueError("BATCH_SIZE must be >= 1.")
        if self.learning_rate <= 0:
            raise ValueError("LEARNING_RATE must be positive.")
        if self.weight_decay < 0:
            raise ValueError("WEIGHT_DECAY must be >= 0.")
        if self.num_workers < 0:
            raise ValueError("NUM_WORKERS must be >= 0.")
        if self.pin_memory not in {"auto", "true", "false"}:
            raise ValueError("PIN_MEMORY must be one of: auto, true, false.")
        if self.persistent_workers not in {"auto", "true", "false"}:
            raise ValueError("PERSISTENT_WORKERS must be one of: auto, true, false.")
        if self.prefetch_factor is not None and self.prefetch_factor < 1:
            raise ValueError("PREFETCH_FACTOR must be empty or >= 1.")
        if self.train_limit is not None and self.train_limit < 1:
            raise ValueError("TRAIN_LIMIT must be empty or >= 1.")
        if self.val_limit is not None and self.val_limit < 1:
            raise ValueError("VAL_LIMIT must be empty or >= 1.")
        if self.lr_scheduler != "none":
            raise ValueError("Only LR_SCHEDULER=none is currently enabled for baseline reproducibility.")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError("EARLY_STOPPING_PATIENCE must be empty or >= 1.")
        if self.inference_limit is not None and self.inference_limit < 1:
            raise ValueError("INFERENCE_LIMIT must be empty or >= 1.")
        if self.inference_split not in {"train", "val", "test"}:
            raise ValueError("INFERENCE_SPLIT must be train, val, or test.")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("CONFIDENCE_THRESHOLD must be in [0, 1].")
        if not 0 <= self.eval_confidence_floor <= 1:
            raise ValueError("EVAL_CONFIDENCE_FLOOR must be in [0, 1].")
        if not 0 <= self.nms_iou_threshold <= 1:
            raise ValueError("NMS_IOU_THRESHOLD must be in [0, 1].")
        if self.eval_top_k < 1:
            raise ValueError("EVAL_TOP_K must be >= 1.")
        if self.max_detections < 1:
            raise ValueError("MAX_DETECTIONS must be >= 1.")
        if self.threshold_analysis and self.inference_split != "val":
            raise ValueError("THRESHOLD_ANALYSIS=true is restricted to INFERENCE_SPLIT=val.")
        if not 0 <= self.threshold_min <= 1:
            raise ValueError("THRESHOLD_MIN must be in [0, 1].")
        if not 0 <= self.threshold_max <= 1:
            raise ValueError("THRESHOLD_MAX must be in [0, 1].")
        if self.threshold_min > self.threshold_max:
            raise ValueError("THRESHOLD_MIN must be <= THRESHOLD_MAX.")
        if self.threshold_step <= 0:
            raise ValueError("THRESHOLD_STEP must be positive.")

    def as_log_dict(self) -> dict[str, Any]:
        out = self.__dict__.copy()
        out["model_path"] = str(self.model_path)
        out["wandb_tags"] = list(self.wandb_tags)
        return out

    @property
    def checkpoint_dir(self) -> Path:
        return self.model_path.parent

    @property
    def run_report_dir(self) -> Path:
        return Path("reports") / "runs" / self.run_name


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("DEVICE=cuda requested, but CUDA is not available.")
        return torch.device("cuda")
    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("DEVICE=mps requested, but Apple MPS is not available.")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {requested}")
