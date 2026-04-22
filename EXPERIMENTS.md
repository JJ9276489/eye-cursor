# Experiments

This file records model lines that were tested and whether they are still worth carrying in the codebase. Use it as experiment history, not as the primary evaluation summary.

Current canonical result snapshot: [docs/results.md](docs/results.md).

Project overview: [docs/overview.md](docs/overview.md).

As of the documented local result snapshot, the practical main line is `spatial_geom 0.5x`: spatial eye-crop CNN plus head/face features plus engineered eye-geometry scalars. Older `frame_*` names are retained as historical benchmark labels.

Historical entries below are notes from earlier report paths and dataset snapshots. They are useful for context but are not always directly comparable to the current snapshot. Use [docs/results.md](docs/results.md) for the current compact result summary.

## Carried In Code

### `spatial_geom`
- type: spatial eye-crop CNN + head/face features + engineered eye-geometry scalars
- status: current main line
- reason: strongest completed local label-holdout result and focused strict session-holdout result for the newer architecture family
- documented local result:
  - `label_holdout`: `86.9px`
  - `session_holdout`: `134.9px`
  - `region_holdout`: `163.9px`

### `spatial`
- type: spatial eye-crop CNN + head/face features
- status: active candidate
- reason: isolates the benefit of preserving eye-crop spatial layout
- documented local result:
  - `label_holdout`: `92.7px`
  - `region_holdout`: `161.0px`

### `clifford`
- type: Clifford/geometric-algebra-inspired eye encoder + head/face features
- status: experimental
- reason: tests whether local geometric-product-style interactions improve eye-crop encoding over standard CNN and patch-transformer branches
- documented local result:
  - `label_holdout`: `92.8px`
  - `region_holdout`: `180.0px`

### `ridge`
- type: polynomial ridge over engineered eye/head features
- status: keep
- reason: cheap benchmark, interpretable, still competitive on easy splits

### `frame_wide_aug_long`
- type: wide eye-crop CNN + head/face features + concat fusion
- status: historical benchmark
- reason: useful comparison point, but no longer the main architecture line
- documented local historical result:
  - `region_holdout`: `97.5px`

### `frame_attention_matched_long`
- type: parameter-matched eye-crop CNN + head/face features + token attention fusion
- status: historical benchmark
- reason: useful comparison point for attention fusion, but not the current main architecture line
- documented local historical result:
  - `session_holdout`: `191.3px`

## Discarded

### `frame_base`
- type: smaller concat eye-crop model
- status: discard
- reason: dominated by the wider concat model

### `frame_base_aug`
- type: smaller concat eye-crop model with augmentation
- status: discard
- reason: dominated by the wider concat model

### `frame_wide_aug`
- type: intermediate training schedule for the wide concat model
- status: discard
- reason: replaced by `frame_wide_aug_long`

### `frame_attention_long`
- type: oversized attention fusion model
- status: discard
- reason: comparison was parameter-confounded; replaced by the matched attention model

### `hybrid_ridge_x_frame_wide_y`
- type: ridge for `x`, frame model for `y`
- status: discard
- reason: did not justify its extra complexity

### `temporal_matched_long`
- type: parameter-matched GRU over capture sequences
- status: discard
- reason: even after parameter matching, it was not competitive enough to justify its training cost
- parameter count: `201,138`
- result:
  - `label_holdout`: `225.6px`

## Notes

- Scaling reports are now consolidated under `reports/scaling/`.
  - canonical command: `.venv/bin/python -u scaling_experiments.py --device mps --output-dir reports/scaling/latest`
  - the old `scaling_studies.py` and `reports/scaling_studies/` path was removed to avoid duplicate experiment systems
- Current architecture candidates in the canonical scaler:
  - `concat`: current average-pooled CNN + head features baseline
  - `spatial`: CNN that preserves the final eye-crop spatial grid instead of global average pooling
  - `spatial_geom`: spatial CNN plus engineered eye-geometry scalars
  - `vit`: tiny patch transformer over eye crops
  - `attn`: current attention-fusion baseline
- Latest completed architecture scaling run (`label_holdout`):
  - `spatial_geom`: `86.9px`
  - `spatial`: `92.7px`
  - `concat`: `120.0-120.6px`
  - implication: spatial layout preservation is a major label-holdout win
- Focused strict spatial-geometry check:
  - `session_holdout`: `spatial_geom 0.5x` `134.9px`
  - implication: `spatial_geom 0.5x` is the current live/default model line
- Current finalist `region_holdout` gate:
  - `spatial 0.5x`: `161.0px`
  - `spatial_geom 0.5x`: `163.9px`
  - `concat 1.5x`: `207.1px`
  - implication: region holdout is worth keeping; spatial layout preservation transfers, while the extra geometry scalars are not a clear region-holdout win overall
- Region distribution ablation:
  - docs: [docs/data_distribution.md](docs/data_distribution.md)
  - `spatial_geom 0.5x` natural label-holdout mean across 3 seeds: `89.2px`
  - `spatial_geom 0.5x` region-balanced label-holdout mean across 3 seeds: `88.1px`
  - implication: region-balanced sampling is not clearly worth making the default; the effect is smaller than seed variance
- Prediction calibration diagnostic:
  - docs: [docs/prediction_calibration.md](docs/prediction_calibration.md)
  - raw eval capture MAE: `94.9px`
  - best global calibration eval capture MAE: `93.5px`
  - implication: edge predictions have inward bias, but a simple global output warp is not the main fix
- Earlier full strict comparisons established the current frame tradeoff:
  - `session_holdout`: `frame_attention_matched_long` `191.3px`, `frame_wide_aug_long` `215.1px`, `ridge` `271.4px`
  - `region_holdout`: `frame_wide_aug_long` `97.5px`, `frame_attention_matched_long` `123.6px`, `ridge` `148.0px`
- Latest active-model `label_holdout` comparison:
  - `ridge` `90.1px`
  - `frame_wide_aug_long` `92.4px`
  - `frame_attention_matched_long` `100.1px`
- Latest retrain + quick cross-session check on the larger `15`-session dataset (`6407` train / `1598` eval):
  - refreshed `concat` checkpoint trainer eval: `82.6px x / 71.2px y`
  - `session_holdout` `quick`: `frame_attention_matched_long` `220.0px`, `frame_wide_aug_long` `224.2px`, `ridge` `239.1px`
  - implication: `concat` remains the cleaner main line for easy/region behavior, but `attention` is still live on cross-session robustness and has not been ruled out
- Latest retrain on the `22`-session dataset (`8803` train / `2215` eval):
  - refreshed `concat` checkpoint trainer eval: `85.1px x / 77.9px y`
  - refreshed `attention` checkpoint trainer eval: `97.1px x / 88.9px y`
  - `label_holdout` `quick`: `frame_wide_aug_long` `131.1px`, `ridge` `144.7px`, `frame_attention_matched_long` `156.7px`
  - implication at that time: `concat` was still the practical line before the later `spatial`/`spatial_geom` scaling work superseded it
