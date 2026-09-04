from __future__ import annotations

import torch

from src.backend.models.losses import DetectionLossOutput


def finite_gradients(model: torch.nn.Module) -> bool:
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            return False
    return True


def loss_output_to_float_dict(loss_output: DetectionLossOutput) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    for key, value in loss_output.__dict__.items():
        if isinstance(value, torch.Tensor):
            out[key] = float(value.detach().cpu().item())
        else:
            out[key] = value
    return out
