# Demo Plan

This is a plan for recording an honest public demo of the current prototype. It is not a script for claiming product readiness.

## Goal

Show that the repo contains a working personalized webcam-only gaze-cursor prototype, while making the limits of the evidence explicit.

## Recording Setup

- Record the full screen and the webcam/debug window if possible.
- State the screen resolution and webcam setup.
- State whether a local trained checkpoint is being used.
- Keep the terminal visible when running commands.
- Avoid editing out failed predictions, edge drift, or setup friction.

## Suggested Sequence

1. Show the README sections for project boundary and reproducibility status.
2. Run the no-webcam smoke test:

```bash
python smoke_test.py
```

3. Launch the preview:

```bash
python live_preview.py
```

4. If no checkpoint is present, show the expected `No prediction available` state.
5. If a local checkpoint is present, show the model label and predicted gaze marker in the live preview.
6. Show the collection flow briefly:

```bash
python main.py
```

7. Move the target, fix gaze, press `space`, and record a few captures.
8. Show the exact training command for the current main model:

```bash
python train_vision_model.py --model spatial_geom --param-multiplier 0.5 --epochs 20 --device mps
```

9. Show the relevant section of [docs/results.md](results.md) and identify `session_holdout` as the main robustness metric.
10. Demonstrate one or two known failure modes, especially edge/corner drift or sensitivity to head pose/lighting.

## What To Show On Screen

- The live predicted gaze marker.
- The webcam/debug window with face and eye tracking.
- Model status and active checkpoint label.
- A simple target or cursor movement task.
- A short result-table view from `docs/results.md`.

## What To Say Explicitly

- This is personalized and trained locally.
- Raw data and trained checkpoints are not committed.
- Reported metrics are documented local snapshots, not public benchmark results.
- `label_holdout` is for fast model filtering.
- `session_holdout` is the main robustness result.
- `region_holdout` is a stricter spatial extrapolation stress test.

## What Not To Claim

- Do not claim this is ready for assistive use.
- Do not claim cross-user generalization.
- Do not claim state-of-the-art performance.
- Do not imply the documented local result is externally reproducible without the private data.
- Do not hide that edge and corner prediction remain difficult.

## Useful Follow-Up Artifacts

- A short unedited live-preview clip using the current local checkpoint.
- A short collection/training clip showing the workflow.
- A public anonymized or synthetic test fixture if exact external reproducibility becomes a goal.
- A small public benchmark protocol if collaborators need to compare across users or webcams.
