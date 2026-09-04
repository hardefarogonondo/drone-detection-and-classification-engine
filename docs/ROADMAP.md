# Project Roadmap

This roadmap reflects the current state of the take-home assessment. The sealed
test split remains off-limits until final evaluation.

## Done

- Completed read-only dataset reconnaissance for `data/obj_det_base/`.
- Frozen a capture-pair-aware train/validation/test split in `data/splits/`.
- Formulated the first custom detector, S8-CFD, from primitive PyTorch modules.
- Implemented dataset loading, preprocessing, target encoding/decoding, model,
  loss, metrics, checkpoints, training CLI, and inference CLI.
- Verified shape, loss, backward-pass, NMS, metric, and coordinate-transform
  behavior with focused tests.
- Ran a tiny train-only overfit sanity check on eight training images.
- Ran validation-only inference smoke output behind the sealed-test guard.

## Current Guardrails

- Do not train a full baseline until the EDA/model-design review is accepted.
- Do not run test-set inference unless explicitly approved for final evaluation.
- Do not commit raw data, checkpoints, prediction renders, local `.env`, W&B
  runtime files, or OS/editor metadata.
- Keep `DEVICE=auto` behavior as CUDA, then Apple MPS, then CPU.

## Next EDA Steps

- Review `notebooks/01_dataset_analysis.ipynb`, `02_split_strategy.ipynb`, and
  `03_detector_formulation.ipynb` together before adding new notebooks.
- Add a focused augmentation/resizing EDA pass for small-object visibility and
  background diversity.
- Define train/validation evaluation tables before running long experiments.
- Decide whether S8-CFD needs multi-object collision handling beyond the current
  one-center-per-cell target assignment after seeing validation errors.

## Later Modeling Steps

- Run a short baseline training job on the frozen train split only.
- Evaluate on validation and inspect failure cases by object size, position,
  background, and confidence threshold.
- Tune the custom loss and post-processing thresholds against validation only.
- Reserve the sealed test split for one final reported evaluation.
