from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def ensure_run_directories(run_dir: Path) -> None:
    for subdir in ["metrics", "figures", "predictions"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[-1].keys()))
        writer.writeheader()
        writer.writerows(rows)


def split_group_summary(manifest_path: Path = Path("data/splits/split_manifest.csv")) -> dict[str, Any]:
    if not manifest_path.exists():
        return {"available": False, "path": str(manifest_path)}
    rows: list[dict[str, str]] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows.extend(reader)
    split_counts: dict[str, int] = {}
    capture_pairs: dict[str, set[str]] = {}
    for row in rows:
        split = row.get("split", "")
        split_counts[split] = split_counts.get(split, 0) + 1
        pair = row.get("capture_pair") or row.get("capture_pair_id") or row.get("sequence_pair") or ""
        if pair:
            capture_pairs.setdefault(split, set()).add(pair)
    return {
        "available": True,
        "path": str(manifest_path),
        "image_counts": split_counts,
        "capture_pairs": {split: sorted(values) for split, values in capture_pairs.items()},
    }
