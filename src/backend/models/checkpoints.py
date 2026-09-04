from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.backend.config.detector_config import DetectorConfig
from src.backend.models.s8_cfd import S8CFD


CHECKPOINT_SCHEMA_VERSION = 2


def normalize_checkpoint_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a backward-compatible checkpoint payload with metadata defaults."""
    if "model_state_dict" not in payload:
        raise KeyError("Checkpoint is missing required key: model_state_dict")
    payload.setdefault("optimizer_state_dict", None)
    payload.setdefault("config", {})
    payload.setdefault("checkpoint_schema_version", 1)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    metadata.setdefault("checkpoint_schema_version", payload.get("checkpoint_schema_version", 1))
    metadata.setdefault("epoch", payload.get("epoch"))
    metadata.setdefault("metric_name", payload.get("metric_name"))
    metadata.setdefault("metric_value", payload.get("metric_value"))
    payload["metadata"] = metadata
    return payload


def save_checkpoint(
    path: str | Path,
    *,
    model: S8CFD,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metric_name: str,
    metric_value: float,
    config: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged_metadata = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "epoch": epoch,
        "metric_name": metric_name,
        "metric_value": metric_value,
        **(metadata or {}),
    }
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "config": config,
        "metadata": merged_metadata,
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model: S8CFD, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=map_location)
    payload = normalize_checkpoint_payload(payload)
    model.load_state_dict(payload["model_state_dict"])
    return payload


def build_model_from_checkpoint(
    path: str | Path,
    config: DetectorConfig,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[S8CFD, dict[str, Any]]:
    model = S8CFD(config)
    payload = load_checkpoint(path, model, map_location=map_location)
    return model, payload
