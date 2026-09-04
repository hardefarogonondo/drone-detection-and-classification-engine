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


@dataclass(frozen=True)
class AppConfig:
    retrain: bool = True
    device: str = "auto"
    model_name: str = "s8_cfd"
    model_path: Path = Path("models/checkpoints/best.pt")
    train_epochs: int = 50
    batch_size: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    train_limit: int | None = None
    val_limit: int | None = None
    resume: bool = False
    inference_split: str = "val"
    inference_limit: int | None = 20
    confidence_threshold: float = 0.30
    nms_iou_threshold: float = 0.50
    save_predictions: bool = True
    wandb_enabled: bool = False
    wandb_project: str = "drone-detection-and-classification-engine"
    allow_test_inference: bool = False
    seed: int = 42

    @classmethod
    def from_env(cls, *, load_env_file: bool = True) -> "AppConfig":
        if load_env_file:
            load_dotenv()
        config = cls(
            retrain=_bool("RETRAIN", cls.retrain),
            device=os.getenv("DEVICE", cls.device),
            model_name=os.getenv("MODEL_NAME", cls.model_name),
            model_path=Path(os.getenv("MODEL_PATH", str(cls.model_path))),
            train_epochs=_int("TRAIN_EPOCHS", cls.train_epochs),
            batch_size=_int("BATCH_SIZE", cls.batch_size),
            learning_rate=_float("LEARNING_RATE", cls.learning_rate),
            weight_decay=_float("WEIGHT_DECAY", cls.weight_decay),
            num_workers=_int("NUM_WORKERS", cls.num_workers),
            train_limit=_optional_int("TRAIN_LIMIT"),
            val_limit=_optional_int("VAL_LIMIT"),
            resume=_bool("RESUME", cls.resume),
            inference_split=os.getenv("INFERENCE_SPLIT", cls.inference_split),
            inference_limit=_optional_int("INFERENCE_LIMIT") if os.getenv("INFERENCE_LIMIT") is not None else cls.inference_limit,
            confidence_threshold=_float("CONFIDENCE_THRESHOLD", cls.confidence_threshold),
            nms_iou_threshold=_float("NMS_IOU_THRESHOLD", cls.nms_iou_threshold),
            save_predictions=_bool("SAVE_PREDICTIONS", cls.save_predictions),
            wandb_enabled=_bool("WANDB_ENABLED", cls.wandb_enabled),
            wandb_project=os.getenv("WANDB_PROJECT", cls.wandb_project),
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
        if self.train_limit is not None and self.train_limit < 1:
            raise ValueError("TRAIN_LIMIT must be empty or >= 1.")
        if self.val_limit is not None and self.val_limit < 1:
            raise ValueError("VAL_LIMIT must be empty or >= 1.")
        if self.inference_limit is not None and self.inference_limit < 1:
            raise ValueError("INFERENCE_LIMIT must be empty or >= 1.")
        if self.inference_split not in {"train", "val", "test"}:
            raise ValueError("INFERENCE_SPLIT must be train, val, or test.")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("CONFIDENCE_THRESHOLD must be in [0, 1].")
        if not 0 <= self.nms_iou_threshold <= 1:
            raise ValueError("NMS_IOU_THRESHOLD must be in [0, 1].")

    def as_log_dict(self) -> dict[str, Any]:
        out = self.__dict__.copy()
        out["model_path"] = str(self.model_path)
        return out


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
