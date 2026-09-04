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

On macOS, install the normal locked environment and let PyTorch use MPS when it
is available. On Windows/Linux workstation environments, the normal PyTorch
package may use CUDA when the local driver/runtime supports it. The Docker image
is intentionally CPU-only so the container remains portable and does not pull
CUDA wheels into a slim runtime.

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
TRAIN_EPOCHS=1
TRAIN_LIMIT=2
VAL_LIMIT=2
BATCH_SIZE=1
INFERENCE_LIMIT=2
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
models/checkpoints/
```

If `RETRAIN=false`, the trainer verifies `MODEL_PATH` exists and exits successfully
without training. This is useful for Docker Compose orchestration.

## Validation Inference

```bash
uv run drone-infer --help
INFERENCE_SPLIT=val INFERENCE_LIMIT=20 uv run drone-infer
```

Outputs are written to:

```text
reports/predictions/<split>/
```

including:

```text
metrics.json
predictions.csv
index.html
rendered prediction PNGs
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
