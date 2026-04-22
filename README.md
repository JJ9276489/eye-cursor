# Eye Cursor

[![Smoke](https://github.com/JJ9276489/eye-cursor/actions/workflows/smoke.yml/badge.svg)](https://github.com/JJ9276489/eye-cursor/actions/workflows/smoke.yml)

Webcam gaze-cursor research prototype using MediaPipe face tracking, aligned eye crops, and small personalized gaze models.

The current direction is personalized webcam-only gaze estimation: collect screen-targeted samples, train a local model, then preview the predicted gaze point in real time.

## Quick Live Preview

Requirements:
- Python 3.11+
- webcam access
- macOS is the most tested path so far

Setup:

```bash
git clone https://github.com/JJ9276489/eye-cursor.git
cd eye-cursor
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start the preview-focused app:

```bash
python live_preview.py
```

`live_preview.py` starts the gaze preview immediately and hides the fullscreen data-collection target. The webcam/debug window still opens because it shows face tracking, eye crops, model status, and key controls.

First launch downloads the MediaPipe face landmarker model into `models/`. Trained gaze checkpoints are not committed to the repo because they are local, screen-specific, and user-specific. Without a local checkpoint the preview will open, but it will show `No prediction available` until you collect data and train a model.

Useful keys:
- `g`: toggle gaze preview.
- `v`: switch active prediction model.
- `m`: show or hide the collection target screen.
- `c`: clear the signal history plot.
- `q` or `Esc`: quit.

## Documentation

Project docs:
- [docs/overview.md](docs/overview.md): entrypoints, code layout, local checks, and artifact policy.
- [docs/results.md](docs/results.md): latest local results snapshot and caveats.
- [docs/data_distribution.md](docs/data_distribution.md): screen-region data-distribution ablation.
- [EXPERIMENTS.md](EXPERIMENTS.md): historical model notes.

Run the no-webcam smoke test:

```bash
python smoke_test.py
```

CI runs a lightweight Python compile check on pushes and pull requests. The local smoke test is deeper because it instantiates the model families and requires the project dependencies.

## Collect And Train

Run the full collection app:

```bash
python main.py
```

Collection flow:
- Move the target with the mouse.
- Fix your gaze on the target.
- Press `space` to record a 1-second capture.
- Repeat across the screen and across varied sessions.

Train the current main model:

```bash
python train_vision_model.py --model spatial_geom --param-multiplier 0.5 --epochs 20 --device mps
```

On non-Apple hardware, replace `--device mps` with `--device cpu` or a supported PyTorch device.

Then run:

```bash
python live_preview.py
```

The live runtime loads checkpoints in this priority order:
- `vision-spatial-geom`: `models/vision_gaze_spatial_geom.pt`
- `vision-spatial`: `models/vision_gaze_spatial.pt`
- `vision-clifford`: `models/vision_gaze_clifford.pt`
- `vision-concat`: `models/vision_gaze_latest.pt`
- `vision-attn`: `models/vision_gaze_attention_matched.pt`
- `ridge`: latest compatible ridge mapper

Missing checkpoints are skipped.

## Current State

Current local compatible dataset:
- `22` sessions
- `399` captures
- `11,363` frame samples
- screen size: `1440x900`

Current model families:
- `ridge`: polynomial ridge baseline over engineered gaze/head features.
- `concat`: old CNN eye-crop baseline with global average pooling.
- `spatial`: CNN eye-crop model that preserves the final spatial feature grid.
- `spatial_geom`: spatial CNN plus engineered eye-geometry scalars.
- `clifford`: experimental Clifford/geometric-algebra-inspired spatial eye encoder.
- `attn`: previous token-attention fusion baseline.
- `vit`: tiny patch transformer comparison line.

Best current strict session-holdout result:
- `spatial_geom 0.5x`: about `134.9px` capture MAE.

Best current label-holdout architecture result:
- `spatial_geom`: about `86.9px` capture MAE.

Important caveat:
- Label holdout is useful for filtering architecture ideas.
- Session holdout is the main robustness metric because it tests generalization across recording sessions.

Historical model notes live in [EXPERIMENTS.md](EXPERIMENTS.md).

The current public result snapshot lives in [docs/results.md](docs/results.md).

## Evaluation

Run canonical scaling sweeps:

```bash
python scaling_experiments.py --mode label_holdout --device mps
```

Focused strict architecture check:

```bash
python scaling_experiments.py \
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

## Project Layout

The concise project layout is in [docs/overview.md](docs/overview.md).

Live app:
- [main.py](main.py): webcam loop, collection, prediction, preview routing.
- [live_preview.py](live_preview.py): one-command preview-focused launcher.
- [runtime_models.py](runtime_models.py): live model loading, switching, and multi-model prediction.
- [ui.py](ui.py): debug overlay and gaze-preview rendering.
- [collection_window.py](collection_window.py): fullscreen target collection UI.

Collection and features:
- [collector.py](collector.py): session/capture/sample serialization.
- [camera.py](camera.py): webcam and screen helpers.
- [landmarker.py](landmarker.py): MediaPipe face landmarker setup.
- [features.py](features.py): engineered eye/head/face features.
- [eye_crops.py](eye_crops.py): aligned eye-crop extraction and saving.

Models and training:
- [mapper.py](mapper.py): ridge baseline.
- [vision_model.py](vision_model.py): neural gaze model architectures.
- [vision_runtime.py](vision_runtime.py): live checkpoint loading and inference.
- [vision_dataset.py](vision_dataset.py): dataset loading and sample tensors.
- [vision_training.py](vision_training.py): shared frame-model training loop.
- [train_vision_model.py](train_vision_model.py): trains one live checkpoint.
- [scaling_experiments.py](scaling_experiments.py): canonical architecture/data/epoch/parameter sweeps.
- [data_distribution_ablation.py](data_distribution_ablation.py): natural-vs-region-balanced training diagnostic.
- [smoke_test.py](smoke_test.py): no-webcam import, compile, and model-forward smoke test.

Historical/support scripts:
- [benchmark_models.py](benchmark_models.py): older benchmark runner for ridge, wide-concat, and matched-attention lines.
- [eval_report.py](eval_report.py): ridge-only evaluation/report generator.
- [vision_eval_report.py](vision_eval_report.py): older frame-model evaluation CLI; fold helpers are still reused by the canonical scaler.

## Data Direction

Near-term priority:
- collect more independent sessions
- vary lighting and posture
- cover the full screen
- avoid over-collecting many nearly identical captures in one sitting

## License

MIT. See [LICENSE](LICENSE).
