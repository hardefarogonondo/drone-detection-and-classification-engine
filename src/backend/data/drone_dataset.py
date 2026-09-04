from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.backend.config.detector_config import DetectorConfig, DEFAULT_DETECTOR_CONFIG
from src.backend.data.preprocessing import build_transform, preprocess_image
from src.backend.data.targets import encode_anchor_free_targets


def read_yolo_annotation(label_path: Path) -> torch.Tensor:
    boxes = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) != 5:
            raise ValueError(f"Malformed annotation line {label_path}:{line_number}")
        class_id, x, y, w, h = parts
        if int(class_id) != 0:
            raise ValueError(f"Unexpected class id {class_id} in {label_path}:{line_number}")
        boxes.append([float(x), float(y), float(w), float(h)])
    return torch.tensor(boxes, dtype=torch.float32)


class DroneDetectionDataset(Dataset):
    """Dataset backed by frozen train/validation manifests."""

    def __init__(
        self,
        manifest_file: str | Path,
        *,
        project_root: str | Path | None = None,
        config: DetectorConfig = DEFAULT_DETECTOR_CONFIG,
        limit: int | None = None,
        allow_test: bool = False,
    ) -> None:
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self.manifest_file = self.project_root / manifest_file
        if self.manifest_file.name == "test.txt" and not allow_test:
            raise ValueError("The sealed test split must not be read during development.")
        if not self.manifest_file.exists():
            raise FileNotFoundError(self.manifest_file)
        self.config = config
        self.image_paths = [
            self.project_root / line.strip()
            for line in self.manifest_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if limit is not None:
            self.image_paths = self.image_paths[:limit]
        if not self.image_paths:
            raise ValueError(f"No images listed in {self.manifest_file}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path = self.image_paths[index]
        label_path = image_path.with_suffix(".txt")
        with Image.open(image_path) as image:
            original_width, original_height = image.size
            image_tensor = preprocess_image(image, self.config)

        original_xywh = read_yolo_annotation(label_path)
        transform = build_transform(original_width, original_height, self.config)
        canvas_xywh = transform.yolo_xywh_to_canvas_xywh(original_xywh)
        encoded = encode_anchor_free_targets(canvas_xywh, self.config, raise_on_collision=True)
        metadata = {
            "image_path": str(image_path),
            "label_path": str(label_path),
            "stem": image_path.stem,
            "original_width": original_width,
            "original_height": original_height,
            "input_width": self.config.input_width,
            "input_height": self.config.input_height,
            "content_height": self.config.content_height,
            "pad_top": self.config.pad_top,
        }
        return {
            "image": image_tensor,
            "target": encoded.target,
            "positive_mask": encoded.positive_mask,
            "boxes_original_xywh": original_xywh,
            "boxes_canvas_xywh": canvas_xywh,
            "metadata": metadata,
        }


def detection_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "positive_mask": torch.stack([item["positive_mask"] for item in batch]),
        "boxes_original_xywh": [item["boxes_original_xywh"] for item in batch],
        "boxes_canvas_xywh": [item["boxes_canvas_xywh"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }
