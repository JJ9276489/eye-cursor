# Expert Review Guide

This repo is organized around a webcam-only personalized gaze cursor prototype.

## Canonical Path

- Live app: `main.py`
- Preview-focused launcher: `live_preview.py`
- Current training entrypoint: `train_vision_model.py`
- Current architecture/scaling evaluator: `scaling_experiments.py`
- Current main model family: `spatial_geom`
- Current public results snapshot: `docs/results.md`

## Historical Or Support Code

- `benchmark_models.py`: older benchmark runner for the ridge, wide-concat, and matched-attention model lines. Kept for historical comparisons.
- `eval_report.py`: ridge-only evaluation/report generator. Kept because the ridge baseline is still useful.
- `vision_eval_report.py`: older frame-model evaluation CLI. Its fold-building helpers are still imported by `scaling_experiments.py`.

## Smoke Test

Run this before reviewing behavior:

```bash
python smoke_test.py
```

The smoke test does not open the webcam and does not require private data or trained checkpoints. It compiles tracked Python files, instantiates each model family, and runs a small CPU forward pass through the neural models.

## Runtime Test

Run:

```bash
python live_preview.py
```

Expected behavior:
- A webcam/debug window opens.
- A fullscreen gaze-preview window opens.
- If no local trained checkpoint exists, the preview says no prediction is available.
- If a compatible checkpoint exists under `models/`, the preview renders predicted gaze markers.
- Press `v` to cycle loaded models.
- Press `q` or `Esc` to exit.

## Reproducibility Boundary

The raw sessions, model checkpoints, and generated reports are ignored by git. That is intentional because this is currently a personalized prototype. Reviewers should treat `docs/results.md` as a documented snapshot, not as a fully reproducible public benchmark.
