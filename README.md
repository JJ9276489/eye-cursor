# Eye Cursor

Webcam gaze-cursor research project using MediaPipe face tracking, manual target collection, aligned eye crops, and small personalized gaze models.

## Current State

Current compatible dataset:
- `22` sessions
- `387` captures
- `11,018` frame samples
- screen size: `1440x900`

Current model families:
- `ridge`: polynomial ridge baseline over engineered gaze/head features.
- `concat`: old CNN eye-crop baseline with global average pooling.
- `spatial`: CNN eye-crop model that preserves the final spatial feature grid.
- `spatial_geom`: spatial CNN plus engineered eye-geometry scalars.
- `attn`: previous token-attention fusion baseline.
- `vit`: tiny patch transformer comparison line.

Completed label-holdout scaling result:
- `spatial_geom`: best around `86.9px` capture MAE.
- `spatial`: best around `92.7px` capture MAE.
- `concat`: best around `120.0-120.6px` capture MAE.
- `attn` and `vit`: not competitive on label holdout.

Important caveat:
- Label holdout is useful for filtering architecture ideas, but `session_holdout` is the main robustness metric.
- A focused session-holdout run is currently comparing `concat`, `spatial`, and `spatial_geom`.

Historical model notes live in [EXPERIMENTS.md](/Users/jeraldyuan/dev/eye-cursor/EXPERIMENTS.md).

## Live App

Run the app:

```bash
cd /Users/jeraldyuan/dev/eye-cursor
.venv/bin/python main.py
```

Keys:
- `space`: record a 1-second capture at the current target.
- `m`: show or hide the collection screen.
- `g`: toggle gaze preview.
- `v`: switch active prediction model.
- `c`: clear the signal history plot.
- `q` or `Esc`: quit.

The gaze preview renders all loaded models as separate markers and highlights the active model. The runtime loads checkpoints in this priority order:
- `vision-spatial-geom`
- `vision-spatial`
- `vision-concat`
- `vision-attn`
- `ridge`

Missing checkpoints are skipped. The spatial models will appear in preview after their live checkpoints exist in `models/`.

## Training

Train the current best label-holdout architecture after MPS is free:

```bash
cd /Users/jeraldyuan/dev/eye-cursor
.venv/bin/python train_vision_model.py --model spatial_geom --param-multiplier 0.5 --epochs 20 --device mps
```

Other model examples:

```bash
.venv/bin/python train_vision_model.py --model spatial --param-multiplier 0.5 --epochs 20 --device mps
.venv/bin/python train_vision_model.py --model concat --epochs 56 --device mps
.venv/bin/python train_vision_model.py --model attn --epochs 56 --device mps
```

Live checkpoint outputs:
- `models/vision_gaze_spatial_geom.pt`
- `models/vision_gaze_spatial.pt`
- `models/vision_gaze_latest.pt`
- `models/vision_gaze_attention_matched.pt`

## Evaluation

Run canonical scaling sweeps:

```bash
cd /Users/jeraldyuan/dev/eye-cursor
.venv/bin/python scaling_experiments.py --mode label_holdout --device mps
```

Focused strict architecture check:

```bash
.venv/bin/python scaling_experiments.py \
  --device mps \
  --output-dir reports/scaling/session_holdout_top \
  --mode session_holdout \
  --models concat spatial spatial_geom \
  --sweeps parameters \
  --param-multipliers 0.5 1.5 \
  --epochs 20
```

Outputs:
- JSON summary: `reports/scaling/<run>/summary.json`
- text summary: `reports/scaling/<run>/summary.txt`
- flat points table: `reports/scaling/<run>/points.csv`
- plots: `reports/scaling/<run>/*.png`
- transform fit summary: `reports/scaling/<run>/transforms/fit_summary.txt`

Scaling runs cache fold/point results under `.cache/scaling`, so interrupted runs can resume without retraining completed points.

## File Map

Live app:
- [main.py](/Users/jeraldyuan/dev/eye-cursor/main.py): webcam loop, collection, prediction, preview routing.
- [runtime_models.py](/Users/jeraldyuan/dev/eye-cursor/runtime_models.py): live model loading, switching, and multi-model prediction.
- [ui.py](/Users/jeraldyuan/dev/eye-cursor/ui.py): debug overlay and gaze-preview rendering.
- [collection_window.py](/Users/jeraldyuan/dev/eye-cursor/collection_window.py): fullscreen target collection UI.

Collection and features:
- [collector.py](/Users/jeraldyuan/dev/eye-cursor/collector.py): session/capture/sample serialization.
- [camera.py](/Users/jeraldyuan/dev/eye-cursor/camera.py): webcam and screen helpers.
- [landmarker.py](/Users/jeraldyuan/dev/eye-cursor/landmarker.py): MediaPipe face landmarker setup.
- [features.py](/Users/jeraldyuan/dev/eye-cursor/features.py): engineered eye/head/face features.
- [eye_crops.py](/Users/jeraldyuan/dev/eye-cursor/eye_crops.py): aligned eye-crop extraction and saving.

Models and training:
- [mapper.py](/Users/jeraldyuan/dev/eye-cursor/mapper.py): ridge baseline.
- [vision_model.py](/Users/jeraldyuan/dev/eye-cursor/vision_model.py): neural gaze model architectures.
- [vision_runtime.py](/Users/jeraldyuan/dev/eye-cursor/vision_runtime.py): live checkpoint loading and inference.
- [vision_dataset.py](/Users/jeraldyuan/dev/eye-cursor/vision_dataset.py): dataset loading and sample tensors.
- [vision_training.py](/Users/jeraldyuan/dev/eye-cursor/vision_training.py): shared frame-model training loop.
- [train_vision_model.py](/Users/jeraldyuan/dev/eye-cursor/train_vision_model.py): trains one live checkpoint.
- [scaling_experiments.py](/Users/jeraldyuan/dev/eye-cursor/scaling_experiments.py): canonical architecture/data/epoch/parameter sweeps.

## Practical Direction

Near-term decision:
- Wait for strict `session_holdout` on `concat`, `spatial`, and `spatial_geom`.
- If `spatial_geom` holds up across sessions, make it the live default.
- Then train `spatial_geom 0.5x` as the preview/live checkpoint and continue collecting session-diverse data.

Data priority:
- more independent sessions
- varied lighting and posture
- full screen coverage
- avoid over-collecting many nearly identical captures in one sitting
