from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CUDATelemetry:
    device_name: str | None
    peak_allocated_mb: float | None
    peak_reserved_mb: float | None


def cuda_device_name(device: torch.device) -> str | None:
    if device.type != "cuda":
        return None
    return torch.cuda.get_device_name(device)


def reset_cuda_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def capture_cuda_telemetry(device: torch.device) -> CUDATelemetry:
    if device.type != "cuda":
        return CUDATelemetry(None, None, None)
    mib = 1024 * 1024
    return CUDATelemetry(
        device_name=torch.cuda.get_device_name(device),
        peak_allocated_mb=torch.cuda.max_memory_allocated(device) / mib,
        peak_reserved_mb=torch.cuda.max_memory_reserved(device) / mib,
    )
