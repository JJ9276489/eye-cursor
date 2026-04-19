# Results Snapshot

This is a public, non-private summary of the local experiments. The raw data, trained checkpoints, and full generated reports are intentionally not committed because they are user-specific and screen-specific.

Documented: `2026-04-19`

Tracked-code reference at transcription time: `c288691`

## Dataset

- Screen: `1440x900`
- Sessions: `22`
- Captures: `387`
- Frame samples: `11,018`
- Collection style: manual target placement, 1-second capture windows, aligned eye crops plus MediaPipe face/head features

## Metrics

- `capture_mae_distance_px`: mean Euclidean pixel error after aggregating frame predictions within each capture.
- `capture_mae_x_px` / `capture_mae_y_px`: absolute pixel error by axis after capture aggregation.
- `label_holdout`: train/eval split from collector labels. Useful for fast architecture filtering, but easier than real deployment.
- `session_holdout`: leave-session-out evaluation. This is the main robustness metric because it tests generalization across collection sessions.

## Current Main Result

`spatial_geom 0.5x` is the current main model line.

- Architecture: spatial eye-crop CNN preserving final feature grids, plus head/face features, plus engineered eye-geometry scalars.
- Parameters: `346,498`
- Strict metric: `session_holdout`
- Result: `134.9px` capture MAE distance
- Axis MAE: `84.4px x / 84.5px y`
- Average fold size: `369.4` train captures / `17.6` eval captures
- Source report: ignored local file `reports/scaling/session_holdout_spatial_geom/summary.json`

Command shape:

```bash
python scaling_experiments.py \
  --device mps \
  --output-dir reports/scaling/session_holdout_spatial_geom \
  --mode session_holdout \
  --models spatial_geom \
  --sweeps parameters \
  --param-multipliers 0.5 \
  --epochs 20
```

## Architecture Sweep

Latest completed dense architecture sweep used `label_holdout` for speed.

| Model | Best Capture MAE | Best Spec |
| --- | ---: | --- |
| `spatial_geom` | `86.9px` | `param_multiplier=0.50` |
| `spatial` | `92.7px` | `param_multiplier=0.50` |
| `concat` | `120.6px` | `param_multiplier=1.50` |
| `attn` | `128.5px` | `param_multiplier=1.25` |
| `vit` | `141.3px` | `param_multiplier=0.50` |

Interpretation: preserving spatial layout in the eye-crop CNN is the strongest observed architecture change so far. The attention and tiny patch-transformer lines were not competitive on this dataset.

## Live Checkpoint

The current local live checkpoint is trained as `spatial_geom 0.5x`.

- Train samples: `8,803`
- Eval samples: `2,215`
- Epochs: `20`
- Device: `mps`
- Random split eval: `61.5px x / 61.9px y`
- Source metadata: ignored local file `models/vision_gaze_spatial_geom.json`

Command:

```bash
python train_vision_model.py --model spatial_geom --param-multiplier 0.5 --epochs 20 --device mps
```

## Caveats

- These are single-machine, local-user results, not a public benchmark.
- Raw data and checkpoints are not committed, so outside reviewers cannot exactly reproduce the numbers from this repo alone.
- `label_holdout` can overstate practical quality because train and eval captures may be from similar sessions.
- `session_holdout` is stricter, but fold sizes are imbalanced because sessions have different amounts of captured data.
- This project does not currently claim SOTA webcam gaze tracking. The current goal is a usable personalized webcam-only cursor prototype.
