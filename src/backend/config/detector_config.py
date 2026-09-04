from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DetectorConfig:
    """Configuration for the first S8-CFD detector baseline."""

    source_width: int = 2560
    source_height: int = 1440
    input_width: int = 960
    content_height: int = 540
    input_height: int = 544
    stride: int = 8
    num_outputs: int = 5
    size_logit_min: float = -8.0
    size_logit_max: float = 0.0
    lambda_obj: float = 1.0
    lambda_center: float = 5.0
    lambda_size: float = 2.0

    @property
    def pad_top(self) -> float:
        return (self.input_height - self.content_height) / 2

    @property
    def grid_width(self) -> int:
        return self.input_width // self.stride

    @property
    def grid_height(self) -> int:
        return self.input_height // self.stride

    @property
    def image_shape(self) -> tuple[int, int]:
        return self.input_height, self.input_width

    @property
    def grid_shape(self) -> tuple[int, int]:
        return self.grid_height, self.grid_width


DEFAULT_DETECTOR_CONFIG = DetectorConfig()
