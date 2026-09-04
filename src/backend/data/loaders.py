from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.backend.config.detector_config import DEFAULT_DETECTOR_CONFIG
from src.backend.config.settings import AppConfig
from src.backend.data.drone_dataset import DroneDetectionDataset, detection_collate


def _resolve_auto_bool(value: str, *, auto_value: bool) -> bool:
    if value == "auto":
        return auto_value
    return value == "true"


def data_loader_kwargs(config: AppConfig, device: torch.device, *, shuffle: bool) -> dict[str, Any]:
    pin_memory = _resolve_auto_bool(config.pin_memory, auto_value=device.type == "cuda")
    persistent_workers = _resolve_auto_bool(config.persistent_workers, auto_value=config.num_workers > 0)
    kwargs: dict[str, Any] = {
        "batch_size": config.batch_size,
        "shuffle": shuffle,
        "num_workers": config.num_workers,
        "collate_fn": detection_collate,
        "pin_memory": pin_memory,
    }
    if config.num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        if config.prefetch_factor is not None:
            kwargs["prefetch_factor"] = config.prefetch_factor
    return kwargs


def build_detection_loader(
    split: str,
    config: AppConfig,
    device: torch.device,
    *,
    shuffle: bool,
    allow_test: bool = False,
) -> DataLoader:
    if split == "test" and not allow_test:
        raise ValueError("The sealed test split must not be read during development.")
    limit = config.train_limit if split == "train" else config.val_limit if split == "val" else config.inference_limit
    dataset = DroneDetectionDataset(
        Path("data") / "splits" / f"{split}.txt",
        project_root=Path.cwd(),
        config=DEFAULT_DETECTOR_CONFIG,
        limit=limit,
        allow_test=allow_test,
    )
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(dataset, generator=generator, **data_loader_kwargs(config, device, shuffle=shuffle))


def summarize_loader_settings(config: AppConfig, device: torch.device) -> dict[str, Any]:
    kwargs = data_loader_kwargs(config, device, shuffle=False)
    return {
        "num_workers": kwargs["num_workers"],
        "pin_memory": kwargs["pin_memory"],
        "persistent_workers": kwargs.get("persistent_workers", False),
        "prefetch_factor": kwargs.get("prefetch_factor"),
    }
