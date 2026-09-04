# drone-detection-and-classification-engine

Custom from-scratch drone object detector for a take-home computer vision assessment.

The detector implementation intentionally avoids pretrained weights, YOLO/Ultralytics,
torchvision detector models, timm backbones, and external NMS utilities.

## Data

Dataset files are not committed to Git. The expected local layout is:

```text
data/obj_det_base/
data/splits/train.txt
data/splits/val.txt
data/splits/test.txt
data/splits/split_manifest.csv
```

The sealed test split must not be used for development decisions. By default, test
inference is blocked unless `ALLOW_TEST_INFERENCE=true` is set explicitly.

## Python And Dependency Policy

The project reference runtime is Python 3.11. The package metadata allows
Python 3.11 through 3.13:

```text
requires-python = ">=3.11,<3.14"
```

The local shell used during setup reported Python 3.14.6 from Anaconda, so use
`uv` with the checked-in `.python-version` rather than relying on whatever
interpreter is first on `PATH`.

PyTorch device selection is runtime-based:

```text
DEVICE=auto -> CUDA if available, then Apple MPS, then CPU
```

PyTorch resolution is platform-specific in `pyproject.toml`:

```text
macOS   -> native PyPI PyTorch wheel; MPS is used when available
Windows -> official PyTorch CUDA 12.6 wheel index
Linux   -> official PyTorch CPU wheel index
```

The Windows CUDA wheel includes the required CUDA runtime libraries, so a local
CUDA Toolkit install is not required. The Docker image runs on Linux and uses
the same locked dependency policy, which keeps the container CPU-only and avoids
pulling CUDA wheels into a slim runtime.

## Local Installation

```bash
uv sync --extra dev
uv run pytest -q
```

If you are inside an already-activated compatible conda environment:

```bash
uv sync --active --extra dev
```

Optional W&B support:

```bash
uv sync --extra dev --extra tracking
```

## Configuration

Copy `.env.example` to `.env` and adjust values as needed. Environment variables
override defaults.

Useful smoke-test values:

```bash
RUN_NAME=smoke-s8-cfd
TRAIN_EPOCHS=1
TRAIN_LIMIT=2
VAL_LIMIT=2
BATCH_SIZE=1
INFERENCE_LIMIT=2
```

Run-specific artifacts are written under:

```text
models/checkpoints/<run_name>/
reports/runs/<run_name>/
```

## Training

```bash
uv run drone-train --help
uv run drone-train
```

Training reads only:

```text
data/splits/train.txt
data/splits/val.txt
```

Checkpoints and history are written under:

```text
models/checkpoints/<run_name>/
```

`best.pt` is selected by validation `mAP@0.50:0.95` (`val_map50_95`).
Precision, recall, and F1 are reported at the configured operating threshold,
while AP is computed from a low evaluation candidate floor.

Training and validation use compact `tqdm` progress bars by default. Set
`PROGRESS=false` for quieter logs in headless jobs. CUDA runs also report GPU
name, epoch duration, throughput, and peak allocated/reserved memory from
PyTorch CUDA APIs.

The approved baseline keeps fixed learning rate training with
`LR_SCHEDULER=none`. `EARLY_STOPPING_PATIENCE` is available for future
experiments but is empty by default.

If `RETRAIN=false`, the trainer verifies `MODEL_PATH` exists and exits successfully
without training. This is useful for Docker Compose orchestration.

## DataLoader Benchmark

The default `NUM_WORKERS=0` is conservative. On Windows CUDA, benchmark a few
settings before a long run:

```powershell
$env:DEVICE="cuda"; $env:RUN_NAME="dataloader-workers-0"; $env:NUM_WORKERS="0"; uv run drone-benchmark-data --split train --batches 100
$env:DEVICE="cuda"; $env:RUN_NAME="dataloader-workers-2"; $env:NUM_WORKERS="2"; uv run drone-benchmark-data --split train --batches 100
$env:DEVICE="cuda"; $env:RUN_NAME="dataloader-workers-4"; $env:NUM_WORKERS="4"; uv run drone-benchmark-data --split train --batches 100
```

`PIN_MEMORY=auto` enables pinned host memory only for CUDA. `PERSISTENT_WORKERS=auto`
uses persistent workers when `NUM_WORKERS>0`. `PREFETCH_FACTOR=2` is used only
when worker processes are enabled.

## Validation Inference

```bash
uv run drone-infer --help
INFERENCE_SPLIT=val INFERENCE_LIMIT=20 uv run drone-infer
```

Outputs are written to:

```text
reports/runs/<run_name>/predictions/<split>/
```

including:

```text
metrics.json
predictions.csv
index.html
rendered prediction PNGs
```

Validation-only threshold analysis:

```bash
RUN_NAME=baseline-s8-cfd-960-s8 MODEL_PATH=models/checkpoints/baseline-s8-cfd-960-s8/best.pt INFERENCE_SPLIT=val THRESHOLD_ANALYSIS=true uv run drone-infer
```

This writes threshold metrics and figures under:

```text
reports/runs/<run_name>/metrics/
reports/runs/<run_name>/figures/
```

## Docker Compose

The Docker image does not bake in the dataset. `data/`, `models/`, and `reports/`
are mounted at runtime.

```bash
docker compose config
docker compose up --build
```

Typical smoke run:

```bash
RETRAIN=true TRAIN_EPOCHS=1 TRAIN_LIMIT=2 VAL_LIMIT=2 BATCH_SIZE=1 INFERENCE_LIMIT=2 docker compose up --build
```

Skip-training path when a checkpoint already exists:

```bash
RETRAIN=false MODEL_PATH=models/checkpoints/best.pt docker compose up --build
```

## Roadmap

See `docs/ROADMAP.md` for the current sequence of EDA, baseline training, and
evaluation gates.
