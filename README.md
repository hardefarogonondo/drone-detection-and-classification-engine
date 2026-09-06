# S8-CFD: From-Scratch Drone Detection

Custom lightweight anchor-free drone detector implemented from scratch in
PyTorch for a small-drone detection research and engineering project.

The detector implementation intentionally avoids pretrained detector weights,
YOLO/Ultralytics, torchvision detector models, timm backbones, external detector
architectures, and external NMS utilities. Decoding, IoU, and NMS are implemented
inside this repository from primitive PyTorch tensor operations.

## Highlights

- 3,242,589 trainable parameters
- stride-8 small-object prediction on a 68x120 grid
- S8/S16/S32 context fusion with GroupNorm
- custom balanced BCE + Smooth L1 objective
- optional standard IoU localization-loss ablation
- from-scratch decoding, IoU, and NMS
- grouped leakage-aware train/validation/test split
- validation-only model selection and sealed test protocol
- W&B tracking support; logging is controlled by `WANDB_ENABLED`
- CUDA/MPS/CPU local runtime selection
- CPU-only Docker support for reproducibility smoke runs

## Final Results

The final model was selected by validation `mAP50-95`. The operating threshold
was selected only on validation:

```text
validation-selected threshold = 0.999
```

The model and threshold were frozen before the single sealed-test evaluation.
The final checkpoint is:

```text
models/weights/s8_cfd_final.pt
```

| Split | AP50 | AP75 | mAP50-95 | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.9376 | 0.4038 | 0.4792 | 0.9611 | 0.9082 | 0.9339 |
| Sealed test | 0.8811 | 0.3026 | 0.4115 | 0.9437 | 0.8584 | 0.8990 |

AP uses ranked candidates retained at `EVAL_CONFIDENCE_FLOOR=0.001`.
Precision, recall, and F1 use the operating threshold. The threshold was
selected only on validation. Test inference is blocked by default, and the final
test evaluation was run once after model selection was frozen.

## Model Summary

S8-CFD processes source images by resizing to 960x540 and vertically padding to
960x544. The detector predicts on a stride-8 68x120 grid. Each cell emits five
values:

```text
objectness, x offset, y offset, log width, log height
```

The network fuses S8/S16/S32 context features and uses GroupNorm. Inference does
not assume exactly two detections; predictions are decoded, score filtered, and
processed with the repository's own NMS implementation.

## Repository Layout

Important directories:

```text
src/                  model, data, metric, and CLI implementation
notebooks/            executed analysis notebooks
models/weights/       final model checkpoint
reports/experiments/  frozen validation experiment artifacts
reports/final/        single sealed-test evaluation artifacts
reports/figures/      publication-ready figures
data/splits/          grouped split manifests
```

Raw dataset files are intentionally not committed. Place or mount the dataset as:

```text
data/obj_det_base/
data/splits/train.txt
data/splits/val.txt
data/splits/test.txt
data/splits/split_manifest.csv
```

The split files contain manifest paths used by the CLI. The sealed test split is
guarded and must not be used for development decisions.

## Local Installation

Python 3.11 is the reference runtime. The project supports Python
`>=3.11,<3.14`.

```bash
uv sync --extra dev
uv run pytest -q
```

If you are inside an already activated compatible conda environment:

```bash
uv sync --active --extra dev
```

PyTorch resolution is platform-specific:

```text
macOS   -> native PyPI PyTorch wheel; MPS is used when available
Windows -> official PyTorch CUDA 12.6 wheel index
Linux   -> official PyTorch CPU wheel index
```

`DEVICE=auto` selects CUDA if available, then Apple MPS if available, then CPU.
The Windows CUDA wheel includes the required CUDA runtime libraries, so a local
CUDA Toolkit install is not required.

## CLI Entry Points

```bash
uv run drone-train --help
uv run drone-infer --help
uv run drone-benchmark-data --help
```

Run-specific artifacts are written under:

```text
models/checkpoints/<run_name>/
reports/runs/<run_name>/
```

## Final-Model Validation Inference Smoke

This smoke command uses the validation split, not the sealed test split.

```bash
RUN_NAME=final-inference-smoke \
MODEL_PATH=models/weights/s8_cfd_final.pt \
DEVICE=auto \
INFERENCE_SPLIT=val \
INFERENCE_LIMIT=2 \
CONFIDENCE_THRESHOLD=0.999 \
THRESHOLD_ANALYSIS=false \
ALLOW_TEST_INFERENCE=false \
uv run drone-infer
```

Outputs are written to:

```text
reports/runs/final-inference-smoke/
```

## Tiny Local Training Smoke

This is only a smoke test of the training path. It does not reproduce the full
baseline and should not overwrite frozen artifacts.

```bash
RUN_NAME=local-training-smoke \
DEVICE=auto \
RETRAIN=true \
TRAIN_EPOCHS=1 \
TRAIN_LIMIT=2 \
VAL_LIMIT=2 \
BATCH_SIZE=1 \
NUM_WORKERS=0 \
WANDB_ENABLED=false \
IOU_LOSS_WEIGHT=0.0 \
PROGRESS=false \
uv run drone-train
```

## Reference Baseline Configuration

```text
epochs       20
batch size   2
LR           0.001
weight decay 0.0001
IoU weight   0
scheduler    none
seed         42
```

The reference baseline training used Windows CUDA with `NUM_WORKERS=4`.

## IoU-Loss Ablation

The optional localization ablation is controlled by:

```text
IOU_LOSS_WEIGHT=1.0
```

Summary:

```text
baseline mAP50-95 = 0.47919
IoU mAP50-95      = 0.47840
```

The IoU `lambda=1` ablation did not improve the predetermined validation
selection metric, so the final selected model remains the baseline.

## Docker

Docker execution is CPU-only. The image does not bake in the dataset, model
weights, or reports. Compose mounts them at runtime:

```text
./data    -> /app/data
./models  -> /app/models
./reports -> /app/reports
```

Build the image:

```bash
docker compose build
```

Final-model validation inference smoke:

```bash
docker compose run --rm --no-deps \
  -e RUN_NAME=docker-final-inference-smoke \
  -e MODEL_PATH=models/weights/s8_cfd_final.pt \
  -e DEVICE=cpu \
  -e INFERENCE_SPLIT=val \
  -e INFERENCE_LIMIT=2 \
  -e CONFIDENCE_THRESHOLD=0.999 \
  -e THRESHOLD_ANALYSIS=false \
  -e ALLOW_TEST_INFERENCE=false \
  inferencer
```

Tiny Docker training smoke:

```bash
docker compose run --rm --no-deps \
  -e RUN_NAME=docker-training-smoke \
  -e DEVICE=cpu \
  -e RETRAIN=true \
  -e TRAIN_EPOCHS=1 \
  -e TRAIN_LIMIT=2 \
  -e VAL_LIMIT=2 \
  -e BATCH_SIZE=1 \
  -e NUM_WORKERS=0 \
  -e WANDB_ENABLED=false \
  -e IOU_LOSS_WEIGHT=0.0 \
  -e PROGRESS=false \
  trainer
```

`docker compose up` is not the primary reproducibility quickstart because the two
services are chained. Use the explicit `docker compose run --rm --no-deps ...`
commands above for targeted smoke checks.

## Notebooks

- `01_dataset_analysis.ipynb`: dataset reconnaissance, annotation schema, image
  dimensions, bounding boxes, and quality checks.
- `02_split_strategy.ipynb`: grouped leakage-aware split construction and split
  distribution validation.
- `03_detector_formulation.ipynb`: S8-CFD architecture, target representation,
  decoding, and objective formulation.
- `04_baseline_training_and_evaluation.ipynb`: frozen baseline training and
  validation analysis.
- `05_detector_experiments.ipynb`: baseline vs IoU-loss ablation comparison.
- `06_final_evaluation.ipynb`: final artifact-only sealed-test evaluation of
  the frozen baseline.

## Artifacts

```text
models/weights/s8_cfd_final.pt   final checkpoint
reports/experiments/             frozen validation experiment artifacts
reports/final/                   single sealed-test evaluation artifacts
reports/figures/                 generated report figures
```

W&B logging is controlled by `WANDB_ENABLED`. W&B project URL:
https://wandb.ai/hardefarogonondo-venturesea/drone-detection-and-classification-engine

## Future Work

The reported pipeline is frozen at the documented experimental scope. The items
below are future extensions rather than open work in the current release, and new
model-selection experiments were not continued after the sealed-test protocol was
frozen.

### Broader Controlled Experiments

Future controlled ablations could evaluate learning-rate scheduling,
GIoU/DIoU/CIoU, lower IoU-loss weights, 1280x720 inputs, stride-4 prediction,
S8-only versus S8/S16/S32 fusion, focal-loss comparison, and systematic
hyperparameter sweeps.

### Robustness

Multi-seed evaluation with mean/std reporting would better quantify run-to-run
variance for the baseline and any future ablations.

### Additional Evaluation

Further artifact-only analyses could include per-scene, weather, raw/augmented,
capture-pair, object-size, IoU/error-distribution, FP/FN taxonomy, and
latency/throughput breakdowns.

### Multi-GPU Training

DistributedDataParallel was intentionally not implemented because the available
training machine had a single NVIDIA GTX 1660 SUPER, so a multi-GPU
implementation could not be meaningfully validated. Multi-GPU support remains
outside the validated hardware scope for this release.

### Engineering Extensions

Useful engineering follow-ups include CI, Docker CI, ONNX/TorchScript export,
quantization, profiling, checksum/version manifests, configuration files,
serving/video inference, and experiment/release artifact automation.

## Paper

Final IEEE-format PDF: [paper/main.pdf](paper/main.pdf)
