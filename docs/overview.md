# Project Overview

This repo is organized around a webcam-only personalized gaze cursor prototype.

## Entry Points

- Live app: `main.py`
- Preview-focused launcher: `live_preview.py`
- Training entrypoint: `train_vision_model.py`
- Architecture/scaling evaluator: `scaling_experiments.py`
- Data-distribution diagnostic: `data_distribution_ablation.py`
- Current model family: `spatial_geom`
- Results snapshot: `docs/results.md`

## Code Layout

- `main.py`: webcam loop, collection, prediction, and preview routing.
- `collection_window.py`: fullscreen target collection UI.
- `runtime_models.py`: live model loading, model switching, and prediction dispatch.
- `ui.py`: debug overlay and fullscreen gaze-preview rendering.
- `features.py`: MediaPipe landmark feature extraction.
- `eye_crops.py`: aligned eye-crop extraction.
- `vision_model.py`: neural model architectures.
- `vision_dataset.py`: dataset loading and tensor construction.
- `vision_training.py`: shared training loop.
- `mapper.py`: polynomial ridge baseline.
- `data_distribution_ablation.py`: natural-vs-region-balanced training diagnostic.

## Historical And Support Scripts

- `benchmark_models.py`: older benchmark runner for the ridge, wide-concat, and matched-attention lines.
- `eval_report.py`: ridge-only evaluation/report generator.
- `vision_eval_report.py`: older frame-model evaluation CLI. Its fold-building helpers are still imported by `scaling_experiments.py`.

## Local Checks

Run the no-webcam smoke test:

```bash
python smoke_test.py
```

The smoke test does not open the webcam and does not require private data or trained checkpoints. It compiles tracked Python files, instantiates each model family, and runs a small CPU forward pass through the neural models.

Run the live preview:

```bash
python live_preview.py
```

Expected runtime behavior:
- A webcam/debug window opens.
- A fullscreen gaze-preview window opens.
- If no local trained checkpoint exists, the preview says no prediction is available.
- If a compatible checkpoint exists under `models/`, the preview renders predicted gaze markers.
- Press `v` to cycle loaded models.
- Press `q` or `Esc` to exit.

## Data And Artifacts

Raw sessions, model checkpoints, and generated reports are ignored by git because this is currently a personalized prototype. Treat `docs/results.md` as a documented local snapshot, not as a fully reproducible public benchmark.
