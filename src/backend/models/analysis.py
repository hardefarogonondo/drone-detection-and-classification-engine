from __future__ import annotations

import csv
from pathlib import Path

import torch

from src.backend.models.metrics import evaluate_at_confidence_threshold


def threshold_grid(start: float, stop: float, step: float) -> list[float]:
    values = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 6))
        current += step
    return values


def analyze_thresholds(
    pred_boxes_by_image: list[torch.Tensor],
    pred_scores_by_image: list[torch.Tensor],
    gt_boxes_by_image: list[torch.Tensor],
    *,
    thresholds: list[float],
) -> list[dict[str, float]]:
    rows = []
    image_count = max(len(pred_boxes_by_image), 1)
    for threshold in thresholds:
        metrics = evaluate_at_confidence_threshold(
            pred_boxes_by_image,
            pred_scores_by_image,
            gt_boxes_by_image,
            confidence_threshold=threshold,
        )
        rows.append({
            "threshold": threshold,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "true_positives": metrics.true_positives,
            "false_positives": metrics.false_positives,
            "false_negatives": metrics.false_negatives,
            "predictions": metrics.predictions,
            "predictions_per_image": metrics.predictions / image_count,
        })
    return rows


def write_threshold_analysis(
    output_dir: Path,
    rows: list[dict[str, float]],
) -> dict[str, float] | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "threshold_analysis.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["threshold"])
        writer.writeheader()
        writer.writerows(rows)
    if not rows:
        return None
    best = max(rows, key=lambda row: (row["f1"], row["precision"], row["recall"]))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return best

    thresholds = [row["threshold"] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, [row["precision"] for row in rows], label="precision")
    ax.plot(thresholds, [row["recall"] for row in rows], label="recall")
    ax.plot(thresholds, [row["f1"] for row in rows], label="F1")
    ax.set_xlabel("confidence threshold")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "threshold_metrics.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([row["recall"] for row in rows], [row["precision"] for row in rows], marker="o", markersize=3)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall_curve.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, [row["predictions_per_image"] for row in rows])
    ax.set_xlabel("confidence threshold")
    ax.set_ylabel("predictions / image")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "predictions_per_image_by_threshold.png", dpi=200)
    plt.close(fig)
    return best
