# Project Status and Future Scope

This document summarizes the frozen project state and future technical scope.
The final model, validation-based model selection, and single sealed-test
evaluation are complete. The sealed test split should remain represented only
through the reported final artifacts and should not be reused for development
decisions.

## Frozen Scope

- Completed dataset reconnaissance for `data/obj_det_base/`.
- Frozen a capture-pair-aware train/validation/test split in `data/splits/`.
- Implemented S8-CFD as a custom detector from primitive PyTorch modules.
- Implemented dataset loading, preprocessing, target encoding/decoding, model,
  loss, metrics, checkpoints, training CLI, and inference CLI.
- Verified shape, loss, backward-pass, NMS, metric, checkpoint, and
  coordinate-transform behavior with focused tests.
- Completed the reference baseline, IoU-loss ablation, final model selection, and
  sealed-test evaluation artifacts.

## Reproducibility Conventions

- Validation data, not the sealed test split, is used for any future development
  or model-selection decisions.
- Raw data, local `.env`, W&B runtime files, and OS/editor metadata remain
  outside version control.
- `DEVICE=auto` resolves to CUDA, then Apple MPS, then CPU.
- Docker execution remains CPU-only unless GPU container support is separately
  implemented and validated.

## Future Scope

- Broaden controlled experiments with learning-rate schedules, alternative
  localization losses, focal loss, higher resolution, stride-4 prediction,
  feature-fusion ablations, and systematic hyperparameter sweeps.
- Add multi-seed evaluation with mean/std reporting.
- Expand analysis by scene, weather, raw/augmented source, capture pair, object
  size, localization error, false-positive/false-negative taxonomy, and
  latency/throughput.
- Consider engineering extensions such as CI, Docker CI, export formats,
  profiling, checksum/version manifests, serving, video inference, and release
  artifact automation.
