# Eye Cursor

Webcam-based gaze cursor research project.

Current stack:
- MediaPipe for face landmarks and head pose
- manual on-screen target collection with saved raw landmarks and eye crops
- one classical baseline plus two live neural contenders

## Current State

Dataset snapshot in [`data/sessions`](/Users/jeraldyuan/dev/eye-cursor/data/sessions):
- usable sessions: `6`
- train captures: `142`
- eval captures: `36`
- train samples: `4331`
- eval samples: `1098`

Active models:
- `ridge`: polynomial ridge over engineered features
- `vision-concat`: wide eye-crop CNN + head/face features + concat fusion
- `vision-attn`: parameter-matched eye-crop CNN + head/face features + token attention fusion

Current operating decision:
- live default: `vision-attn`
- reason: it has the best known `session_holdout`, which is the most important metric for actual reuse across runs

Best known strict results:
- `session_holdout`: `vision-attn` `191.3px`, `vision-concat` `215.1px`, `ridge` `271.4px`
- `region_holdout`: `vision-concat` `97.5px`, `vision-attn` `123.6px`, `ridge` `148.0px`
- latest `label_holdout`: `ridge` `90.1px`, `vision-concat` `92.4px`, `vision-attn` `100.1px`

Important implication:
- `vision-attn` is the current best choice if cross-session robustness matters most
- `vision-concat` is still the better screen-space interpolator
- ridge remains a useful benchmark and fallback, not the main path

The discarded temporal line was retested with a parameter-matched GRU and dropped again:
- `temporal_matched_long`: `201,138` params
- same scale as current frame contenders
- `label_holdout`: `225.6px`
- conclusion: not worth pursuing on the current data

Historical model notes live in [EXPERIMENTS.md](/Users/jeraldyuan/dev/eye-cursor/EXPERIMENTS.md).

## File Map

### Live app
- [`main.py`](/Users/jeraldyuan/dev/eye-cursor/main.py): app loop
- [`runtime_models.py`](/Users/jeraldyuan/dev/eye-cursor/runtime_models.py): live model loading, switching, and prediction routing
- [`ui.py`](/Users/jeraldyuan/dev/eye-cursor/ui.py): debug and preview rendering
- [`collection_window.py`](/Users/jeraldyuan/dev/eye-cursor/collection_window.py): fullscreen collection UI

### Collection and feature extraction
- [`collector.py`](/Users/jeraldyuan/dev/eye-cursor/collector.py): session/capture management and sample serialization
- [`camera.py`](/Users/jeraldyuan/dev/eye-cursor/camera.py): webcam and screen handling
- [`landmarker.py`](/Users/jeraldyuan/dev/eye-cursor/landmarker.py): MediaPipe setup
- [`features.py`](/Users/jeraldyuan/dev/eye-cursor/features.py): engineered gaze/head features
- [`eye_crops.py`](/Users/jeraldyuan/dev/eye-cursor/eye_crops.py): aligned eye crop extraction and saving

### Models
- [`mapper.py`](/Users/jeraldyuan/dev/eye-cursor/mapper.py): ridge baseline and fallback model
- [`vision_model.py`](/Users/jeraldyuan/dev/eye-cursor/vision_model.py): active frame vision architectures
- [`vision_runtime.py`](/Users/jeraldyuan/dev/eye-cursor/vision_runtime.py): live checkpoint loading for the frame vision models

### Training and evaluation
- [`train_vision_model.py`](/Users/jeraldyuan/dev/eye-cursor/train_vision_model.py): trains one of the two active live vision models
- [`eval_report.py`](/Users/jeraldyuan/dev/eye-cursor/eval_report.py): strict ridge diagnostics
- [`vision_eval_report.py`](/Users/jeraldyuan/dev/eye-cursor/vision_eval_report.py): strict frame vision diagnostics
- [`benchmark_models.py`](/Users/jeraldyuan/dev/eye-cursor/benchmark_models.py): direct comparison of the active models only
- [`scaling_experiments.py`](/Users/jeraldyuan/dev/eye-cursor/scaling_experiments.py): one-variable-at-a-time scaling sweeps for params, data volume, and epoch budget

## Recommended Workflow

Collect more data:

```bash
cd /Users/jeraldyuan/dev/eye-cursor
.venv/bin/python main.py
```

Keys:
- `space`: record 1-second capture
- `m`: show/hide collection screen
- `g`: toggle gaze preview
- `v`: switch active live model
- `q` or `Esc`: quit

Data priority:
1. more sessions, not more frames inside one session
2. modest variation in lighting, distance, and posture
3. maintain full-screen coverage

Train the current live-default model:

```bash
cd /Users/jeraldyuan/dev/eye-cursor
.venv/bin/python train_vision_model.py
```

Train the concat contender:

```bash
cd /Users/jeraldyuan/dev/eye-cursor
.venv/bin/python train_vision_model.py --model concat
```

Train the attention model explicitly:

```bash
cd /Users/jeraldyuan/dev/eye-cursor
.venv/bin/python train_vision_model.py --model attn
```

Benchmark the active models:

```bash
cd /Users/jeraldyuan/dev/eye-cursor
.venv/bin/python benchmark_models.py --mode label_holdout
.venv/bin/python benchmark_models.py --mode session_holdout
.venv/bin/python benchmark_models.py --mode region_holdout
```

Benchmark profiles:
- `standard`: default, uses early stopping to make strict holdouts more practical
- `full`: uses the full epoch budget for deliberate comparisons
- `quick`: lighter tracking run, especially for `session_holdout`

Benchmarks are cached by default. If data/code/settings have not changed, rerunning reuses the cached result instead of retraining. Use `--refresh` to force recomputation.

Examples:

```bash
cd /Users/jeraldyuan/dev/eye-cursor
.venv/bin/python benchmark_models.py --mode session_holdout --profile quick
.venv/bin/python benchmark_models.py --mode session_holdout --profile full
```

Run scaling sweeps:

```bash
cd /Users/jeraldyuan/dev/eye-cursor
.venv/bin/python scaling_experiments.py --mode label_holdout --models attn concat --device mps
```

This writes graphs and JSON summaries to [`reports/scaling/latest`](/Users/jeraldyuan/dev/eye-cursor/reports/scaling/latest). Point results are cached under `.cache/scaling`, so interrupted runs resume instead of retraining every point.

## Practical Read

The bottleneck is no longer “is there webcam signal at all?”

The real bottlenecks are:
- cross-session robustness
- screen-region generalization
- data diversity across sessions

So the next useful work is:
- keep collecting session-diverse data
- keep comparing `vision-attn` vs `vision-concat`
- treat `session_holdout` as the main truth metric

## Command Surface

There is now one training command and one comparison command:
- train: [`train_vision_model.py`](/Users/jeraldyuan/dev/eye-cursor/train_vision_model.py)
- compare: [`benchmark_models.py`](/Users/jeraldyuan/dev/eye-cursor/benchmark_models.py)

## Reports Layout

Generated reports now live in:
- active benchmarks: [`reports/benchmark/latest`](/Users/jeraldyuan/dev/eye-cursor/reports/benchmark/latest)
- scaling sweeps: [`reports/scaling/latest`](/Users/jeraldyuan/dev/eye-cursor/reports/scaling/latest)
- one-off analysis notes: [`reports/notes`](/Users/jeraldyuan/dev/eye-cursor/reports/notes)
