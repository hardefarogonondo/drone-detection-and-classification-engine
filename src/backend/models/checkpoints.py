from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.backend.config.detector_config import DetectorConfig
from src.backend.models.s8_cfd import S8CFD


def save_checkpoint(
    path: str | Path,
    *,
    model: S8CFD,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metric_name: str,
    metric_value: float,
    config: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "config": config,
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model: S8CFD, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=map_location)
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
