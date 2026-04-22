# Reviewer Guide

This guide is for a technical reader who wants to inspect the project quickly and avoid confusing local personalized artifacts with public reproducible assets.

## Start Here

1. Read [README.md](../README.md) for the project boundary, quickstart, current model path, and reproducibility status.
2. Read [docs/results.md](results.md) for the current evaluation story and metric definitions.
3. Read [docs/overview.md](overview.md) for the file map and architecture sketch.
4. Use [EXPERIMENTS.md](../EXPERIMENTS.md) only after that, because it includes historical model names and older comparison context.

## Minimal Local Checks

Install dependencies from the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the no-webcam smoke test:

```bash
python smoke_test.py
```

This is the cleanest public check. It verifies imports, compiles tracked Python files, instantiates model families, and runs dummy CPU forward passes. It does not require private data or trained checkpoints.

Run the live preview:

```bash
python live_preview.py
```

Expected behavior without private artifacts: the preview opens and reports that no prediction is available. Real gaze prediction requires a locally trained checkpoint under `models/`.

## Main Evaluation Story

Use [docs/results.md](results.md) as the canonical public summary.

- Main robustness metric: `session_holdout` capture MAE.
- Current selected model line: `spatial_geom 0.5x`.
- Current documented main result: `134.9px` capture MAE on local session holdout.
- Internal filtering metric: `label_holdout`; useful for architecture sweeps but easier than deployment.
- Spatial extrapolation gate: `region_holdout`; useful for testing edge/corner generalization.

Do not interpret the reported metrics as a public benchmark. The raw personalized data, trained checkpoints, and generated reports are intentionally not committed.

## Core Files To Inspect

Runtime and collection:

- `main.py`
- `live_preview.py`
- `collection_window.py`
- `runtime_models.py`
- `ui.py`
- `camera.py`
- `landmarker.py`

Feature and data pipeline:

- `collector.py`
- `features.py`
- `eye_crops.py`
- `vision_dataset.py`

Models and training:

- `vision_model.py`
- `vision_training.py`
- `vision_runtime.py`
- `mapper.py`
- `train_vision_model.py`

Evaluation:

- `scaling_experiments.py`
- `data_distribution_ablation.py`
- `prediction_calibration_analysis.py`
- `smoke_test.py`

## Historical Or Support Material

- `benchmark_models.py`: older benchmark runner.
- `eval_report.py`: ridge-only report generator.
- `vision_eval_report.py`: older frame-model evaluation CLI; some fold helpers remain useful.
- `EXPERIMENTS.md`: model history, discarded lines, and older result labels.

These files are useful context, but they are not the shortest path to understanding the current system.

## Local Artifacts Omitted From Git

- `data/`: personalized collection sessions.
- `models/`: MediaPipe model download and trained local gaze checkpoints.
- `reports/`: generated evaluation outputs.
- `.cache/`: cached scaling fold results.
- `logs/`: local runtime logs if produced.

This omission means the repo is runnable, but the documented local results are not exactly reproducible from git alone.

## What To Look For Critically

- Whether `session_holdout` is the right robustness proxy for the intended use case.
- Whether collection sessions are diverse enough to support the claimed personalized behavior.
- Whether region/corner failures are data limitations, signal limitations, or model limitations.
- Whether the live preview behavior matches the documented metrics.
- Whether a public demo or public anonymized benchmark should be added before broader claims are made.
