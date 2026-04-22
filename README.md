# Eye Cursor

[![Smoke](https://github.com/JJ9276489/eye-cursor/actions/workflows/smoke.yml/badge.svg)](https://github.com/JJ9276489/eye-cursor/actions/workflows/smoke.yml)

Webcam-only personalized gaze-cursor prototype using MediaPipe face landmarks, aligned eye crops, and small local gaze models.

This project asks a narrow HMI question: how far can a commodity webcam, per-user calibration data, and lightweight personalized models get toward usable gaze-based cursor control?

It is not a polished assistive product, not a public benchmark, and not a claim of state-of-the-art webcam gaze tracking.

## Implemented Now

- Manual gaze-data collection against a fullscreen target.
- MediaPipe face landmark tracking and head/face feature extraction.
- Affine-aligned left/right eye crops at `96x64`.
- Local personalized training for several small model families.
- Live preview that renders predicted gaze points when a compatible local checkpoint exists.
- Evaluation scripts for label holdout, session holdout, region holdout, scaling sweeps, data-distribution checks, and output-calibration diagnostics.

Current model path:

```text
webcam frame
  -> MediaPipe face landmarks
  -> aligned left/right eye crops + head/face/eye-geometry features
  -> personalized spatial CNN gaze model
  -> normalized screen prediction
  -> live preview cursor overlay
```

The current main model family is `spatial_geom 0.5x`: a compact spatial eye-crop CNN that preserves the final eye feature grid, then fuses that visual representation with head/face features and engineered eye-geometry scalars.

## Quickstart

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

Run the no-webcam smoke test:

```bash
python smoke_test.py
```

The smoke test compiles tracked Python files, instantiates the neural model families, runs small CPU forward passes, and verifies checkpoint discovery metadata. It does not require private data, trained checkpoints, or webcam access.

Run the preview-focused app:

```bash
python live_preview.py
```

First launch downloads the MediaPipe face landmarker into `models/`. Trained gaze checkpoints are not committed because they are local, screen-specific, and user-specific. Without a local checkpoint, the preview still opens but reports that no prediction is available.

Useful runtime keys:

- `g`: toggle gaze preview.
- `v`: cycle loaded prediction models.
- `m`: show or hide the fullscreen collection target.
- `c`: clear the signal history plot.
- `q` or `Esc`: quit.

## Collect, Train, Evaluate

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

On non-Apple hardware, replace `--device mps` with `--device cpu` or another supported PyTorch device.

Run the live preview after training:

```bash
python live_preview.py
```

Run a focused label-holdout check:

```bash
python scaling_experiments.py \
  --device mps \
  --output-dir reports/scaling/label_holdout_spatial_geom \
  --mode label_holdout \
  --models spatial_geom \
  --sweeps parameters \
  --param-multipliers 0.5 \
  --epochs 20
```

Run the current strict session-holdout check for the main model:

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

Scaling runs write ignored local artifacts under `reports/scaling/` and cache completed points under `.cache/scaling/`.

## Current Reported Result

The current selected model line is `spatial_geom 0.5x`.

Main robustness result:

- Metric: `session_holdout` capture MAE distance.
- Reported local result: `134.9px`.
- Dataset snapshot: single user, `1440x900` screen, `22` sessions.
- Source: [docs/results.md](docs/results.md).

What this means:

- The model was evaluated by holding out entire collection sessions.
- This is stricter than a random or label-based split because it tests cross-session generalization.
- It is the main result to cite when discussing current robustness.

What this does not mean:

- It is not a public benchmark result.
- It is not externally reproducible from this repo alone because the raw personalized data and checkpoints are not committed.
- It does not establish usability for another person, another webcam, another screen, or assistive-tech deployment.
- It does not mean the model performs equally well at all screen regions; region holdout remains harder.

Internal model-filtering result:

- `label_holdout`: `spatial_geom 0.5x` reaches about `86.9px` capture MAE in the documented local sweep.
- This split is useful for comparing architecture ideas quickly, but it is easier than the deployment-like case because train and eval captures can be from similar sessions.

Stress-test result:

- `region_holdout`: `spatial_geom 0.5x` reports about `163.9px` capture MAE.
- This split holds out screen regions and tests spatial extrapolation. It is useful as a promotion gate, not as the default user-experience estimate.

## Reproducibility Status

| Path | Status |
| --- | --- |
| Install dependencies | Fully runnable from this repo. |
| `python smoke_test.py` | Fully runnable without webcam, data, or checkpoints. |
| `python live_preview.py` | Runnable with a webcam. Prediction requires a local trained checkpoint. |
| `python main.py` collection | Runnable, but produces private personalized data under ignored `data/`. |
| `python train_vision_model.py ...` | Runnable after local collection. Result depends on user, webcam, screen, lighting, and collection quality. |
| `python scaling_experiments.py ...` | Runnable after local collection. Metrics are local to that dataset. |
| Reported numbers in `docs/results.md` | Documented local snapshots, not exactly reproducible from the public repo. |

Ignored local artifacts:

- `data/`: raw personalized collection sessions.
- `models/`: MediaPipe model download plus trained local gaze checkpoints.
- `reports/`: generated experiment outputs.
- `.cache/`: cached scaling fold results.
- `logs/`: runtime logs if produced.

This boundary is intentional. The current data is biometric-ish, user-specific, and screen-specific.

## Main Limitations

- Personalized only: there is no cross-user model claim.
- Webcam-only signal is weak compared with dedicated eye trackers.
- Results are sensitive to camera position, screen geometry, lighting, glasses, head pose, and collection consistency.
- Edge and corner predictions show inward bias that simple output warping did not solve.
- Session-holdout folds are imbalanced because collection sessions contain different amounts of data.
- The live preview is a research/debug interface, not an accessibility-ready cursor replacement.
- No public dataset, public checkpoint, or independent benchmark is included.

## Documentation Map

- [docs/reviewer_guide.md](docs/reviewer_guide.md): fastest path for a technical reviewer.
- [docs/overview.md](docs/overview.md): architecture, entrypoints, core files, and support files.
- [docs/results.md](docs/results.md): current local result snapshot and evaluation definitions.
- [docs/data_distribution.md](docs/data_distribution.md): region-distribution ablation.
- [docs/prediction_calibration.md](docs/prediction_calibration.md): edge-compression and output-calibration diagnostic.
- [docs/demo_plan.md](docs/demo_plan.md): honest demo-video plan.
- [EXPERIMENTS.md](EXPERIMENTS.md): historical model notes and discarded lines.

## Core Entrypoints

- `main.py`: webcam loop, collection, prediction, and preview routing.
- `live_preview.py`: one-command preview-focused launcher.
- `train_vision_model.py`: trains one live vision checkpoint.
- `scaling_experiments.py`: canonical architecture/data/epoch/parameter sweeps.
- `smoke_test.py`: no-webcam compile and model-forward smoke test.

## License

MIT. See [LICENSE](LICENSE).
