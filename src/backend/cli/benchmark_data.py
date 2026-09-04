from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from src.backend.config.settings import AppConfig, load_dotenv, select_device
from src.backend.data.loaders import build_detection_loader, summarize_loader_settings
from src.backend.utils.reproducibility import seed_everything
from src.backend.utils.runs import ensure_run_directories, write_json
from src.backend.utils.telemetry import cuda_device_name, synchronize_if_cuda


def benchmark_data_loading(
    config: AppConfig,
    *,
    split: str,
    batches: int,
    transfer_to_device: bool,
) -> dict[str, float | int | str | bool | None]:
    if split == "test":
        raise ValueError("DataLoader benchmark must not read the sealed test split.")
    seed_everything(config.seed)
    device = select_device(config.device)
    loader = build_detection_loader(split, config, device, shuffle=False, allow_test=False)
    max_batches = min(batches, len(loader))
    image_count = 0
    start = time.time()
    iterator = tqdm(loader, desc=f"Benchmark data - {split}", total=max_batches, dynamic_ncols=True, leave=False, disable=not config.progress)
    for batch_index, batch in enumerate(iterator):
        if batch_index >= max_batches:
            break
        if transfer_to_device:
            non_blocking = device.type == "cuda"
            batch["image"].to(device, non_blocking=non_blocking)
            batch["target"].to(device, non_blocking=non_blocking)
            batch["positive_mask"].to(device, non_blocking=non_blocking)
        image_count += int(batch["image"].shape[0])
    synchronize_if_cuda(device)
    elapsed = time.time() - start
    result = {
        "split": split,
        "batches": max_batches,
        "images": image_count,
        "elapsed_sec": elapsed,
        "batches_per_sec": max_batches / max(elapsed, 1e-8),
        "images_per_sec": image_count / max(elapsed, 1e-8),
        "transfer_to_device": transfer_to_device,
        "device": str(device),
        "cuda_gpu": cuda_device_name(device),
        **summarize_loader_settings(config, device),
    }
    run_dir = config.run_report_dir
    ensure_run_directories(run_dir)
    write_json(run_dir / "metrics" / f"dataloader_benchmark_{split}.json", result)
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark manifest-backed DataLoader throughput without training.")
    parser.add_argument("--env-file", default=".env", help="Path to .env file. Environment variables still take precedence.")
    parser.add_argument("--split", default="train", choices=["train", "val"], help="Manifest split to benchmark.")
    parser.add_argument("--batches", type=int, default=50, help="Maximum number of batches to iterate.")
    parser.add_argument("--no-transfer", action="store_true", help="Skip tensor transfer to the selected device.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)
    try:
        config = AppConfig.from_env(load_env_file=False)
        if args.batches < 1:
            raise ValueError("--batches must be >= 1.")
        benchmark_data_loading(
            config,
            split=args.split,
            batches=args.batches,
            transfer_to_device=not args.no_transfer,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
