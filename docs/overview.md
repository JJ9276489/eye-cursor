# Project Overview

This repo is organized around a webcam-only personalized gaze cursor prototype. The core path is collection, local personalized training, and live preview. Most other scripts exist to evaluate or diagnose that path.

## Architecture

```text
webcam frame
  -> MediaPipe face landmarker
  -> face/head features
  -> aligned left/right eye crops
  -> shared eye encoder
  -> multimodal fusion with head and optional eye-geometry features
  -> normalized gaze point
  -> live preview rendering
```

The current main model family is `spatial_geom`: a compact spatial CNN eye encoder with head/face features and engineered eye-geometry scalars. It is selected because it is the strongest current local session-holdout line, not because it is a general pretrained gaze model.

## Current Entrypoints

- `main.py`: full webcam loop for collection, prediction, and preview routing.
- `live_preview.py`: preview-focused launcher; starts with the collection target hidden.
- `train_vision_model.py`: trains one local live vision checkpoint.
- `scaling_experiments.py`: canonical evaluation runner for label, session, and region holdout sweeps.
- `smoke_test.py`: no-webcam compile and model-forward smoke test.

## Core Runtime Files

- `collection_window.py`: fullscreen target collection UI.
- `runtime_models.py`: live model loading, model switching, and prediction dispatch.
- `ui.py`: debug overlay and fullscreen gaze-preview rendering.
- `camera.py`: webcam and screen helpers.
- `landmarker.py`: MediaPipe face landmarker setup.

## Data And Feature Files

- `collector.py`: session, capture, and sample serialization.
- `features.py`: MediaPipe landmark feature extraction.
- `eye_crops.py`: aligned eye-crop extraction.
- `vision_dataset.py`: dataset loading and tensor construction.

## Model And Training Files

- `vision_model.py`: neural model architectures.
- `vision_training.py`: shared neural training loop.
- `vision_runtime.py`: live checkpoint loading and inference.
- `mapper.py`: polynomial ridge baseline over engineered features.

## Evaluation And Diagnostics

- `scaling_experiments.py`: canonical architecture/data/epoch/parameter sweeps.
- `data_distribution_ablation.py`: natural-vs-region-balanced training diagnostic.
- `prediction_calibration_analysis.py`: output-calibration and edge-compression diagnostic.
- `docs/results.md`: current documented local result snapshot and metric definitions.
- `docs/data_distribution.md`: compact summary of the data-distribution diagnostic.
- `docs/prediction_calibration.md`: compact summary of the prediction-calibration diagnostic.

## Historical And Support Scripts

These files are kept because they document or support earlier comparisons. They are not the main path for a new reviewer.

- `benchmark_models.py`: older benchmark runner for ridge, wide-concat, and matched-attention lines.
- `eval_report.py`: ridge-only evaluation/report generator.
- `vision_eval_report.py`: older frame-model evaluation CLI. Its fold-building helpers are still imported by `scaling_experiments.py`.
- `EXPERIMENTS.md`: historical model notes, discarded lines, and context for earlier result names.

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

## Artifact Policy

Raw sessions, trained model checkpoints, generated reports, and scaling caches are ignored by git because this is currently a personalized prototype.

- Public and runnable: code, docs, smoke test, model definitions, collection/training/evaluation scripts.
- Local and private: `data/`, trained gaze checkpoints under `models/`, generated `reports/`, and `.cache/`.
- Documented but not exactly reproducible: numeric result snapshots in `docs/results.md`.

Use [docs/reviewer_guide.md](reviewer_guide.md) for an external review path and [docs/demo_plan.md](demo_plan.md) for a public demo plan.
