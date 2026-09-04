from __future__ import annotations

import torch

from src.backend.cli.train import _checkpoint_metadata, _is_improved, _resume_best_state
from src.backend.config.settings import AppConfig
from src.backend.models.checkpoints import CHECKPOINT_SCHEMA_VERSION, load_checkpoint, save_checkpoint
from src.backend.models.s8_cfd import S8CFD, count_trainable_parameters


def test_save_and_load_checkpoint_with_metadata(tmp_path) -> None:
    model = S8CFD()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    metadata = {
        "checkpoint_kind": "best",
        "run_name": "unit-test-run",
        "model_name": "s8_cfd",
        "parameter_count": count_trainable_parameters(model),
        "input_resolution": [960, 544],
        "stride": 8,
        "optimizer": {"name": "AdamW"},
        "seed": 42,
        "selected_metric": "val_map50_95",
        "current_metric_value": 0.42,
        "best_metric_value": 0.42,
        "best_epoch": 3,
    }
    path = tmp_path / "best.pt"

    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=3,
        metric_name="val_map50_95",
        metric_value=0.42,
        config={"run_name": "unit-test-run"},
        metadata=metadata,
    )
    loaded_model = S8CFD()
    payload = load_checkpoint(path, loaded_model, map_location="cpu")

    assert payload["checkpoint_schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert payload["epoch"] == 3
    assert payload["metric_name"] == "val_map50_95"
    assert payload["metric_value"] == 0.42
    assert payload["metadata"]["checkpoint_kind"] == "best"
    assert payload["metadata"]["best_metric_value"] == 0.42
    assert payload["metadata"]["best_epoch"] == 3
    assert payload["optimizer_state_dict"] is not None


def test_load_legacy_checkpoint_without_metadata(tmp_path) -> None:
    model = S8CFD()
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": None,
            "epoch": 5,
            "metric_name": "val_ap50",
            "metric_value": 0.7,
            "config": {"legacy": True},
        },
        path,
    )

    loaded_model = S8CFD()
    payload = load_checkpoint(path, loaded_model, map_location="cpu")

    assert payload["checkpoint_schema_version"] == 1
    assert payload["metadata"]["checkpoint_schema_version"] == 1
    assert payload["metadata"]["epoch"] == 5
    assert payload["metadata"]["metric_name"] == "val_ap50"
    assert payload["metadata"]["metric_value"] == 0.7


def test_best_and_latest_checkpoint_selection_metadata(tmp_path) -> None:
    config = AppConfig(run_name="unit-test-run", model_path=tmp_path / "best.pt")
    model = S8CFD()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_path = tmp_path / "best.pt"
    latest_path = tmp_path / "latest.pt"
    best_metric = float("-inf")
    best_epoch = None
    metric_name = "val_map50_95"

    for epoch, current_metric in [(0, 0.5), (1, 0.4)]:
        if _is_improved(current_metric, best_metric):
            best_metric = current_metric
            best_epoch = epoch
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metric_name=metric_name,
                metric_value=best_metric,
                config=config.as_log_dict(),
                metadata=_checkpoint_metadata(
                    config=config,
                    model=model,
                    optimizer=optimizer,
                    checkpoint_kind="best",
                    epoch=epoch,
                    metric_name=metric_name,
                    current_metric_value=current_metric,
                    best_metric_value=best_metric,
                    best_epoch=best_epoch,
                ),
            )
        save_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metric_name=metric_name,
            metric_value=current_metric,
            config=config.as_log_dict(),
            metadata=_checkpoint_metadata(
                config=config,
                model=model,
                optimizer=optimizer,
                checkpoint_kind="latest",
                epoch=epoch,
                metric_name=metric_name,
                current_metric_value=current_metric,
                best_metric_value=best_metric,
                best_epoch=best_epoch,
            ),
        )

    best_payload = load_checkpoint(best_path, S8CFD(), map_location="cpu")
    latest_payload = load_checkpoint(latest_path, S8CFD(), map_location="cpu")

    assert best_payload["epoch"] == 0
    assert best_payload["metadata"]["checkpoint_kind"] == "best"
    assert best_payload["metadata"]["best_metric_value"] == 0.5
    assert latest_payload["epoch"] == 1
    assert latest_payload["metric_value"] == 0.4
    assert latest_payload["metadata"]["checkpoint_kind"] == "latest"
    assert latest_payload["metadata"]["best_metric_value"] == 0.5
    assert latest_payload["metadata"]["best_epoch"] == 0


def test_resume_best_state_from_legacy_payload() -> None:
    best_metric, best_epoch = _resume_best_state(
        {
            "epoch": 2,
            "metric_name": "val_ap50",
            "metric_value": 0.33,
            "metadata": {},
        }
    )
    assert best_metric == 0.33
    assert best_epoch is None


def test_resume_best_state_from_latest_metadata() -> None:
    best_metric, best_epoch = _resume_best_state(
        {
            "epoch": 4,
            "metric_name": "val_map50_95",
            "metric_value": 0.31,
            "metadata": {
                "checkpoint_kind": "latest",
                "best_metric_value": 0.44,
                "best_epoch": 3,
            },
        }
    )
    assert best_metric == 0.44
    assert best_epoch == 3
