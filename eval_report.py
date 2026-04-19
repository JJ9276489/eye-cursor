from __future__ import annotations

"""Ridge-only evaluation/report generator.

The neural-model scaling path now lives in `scaling_experiments.py`, but this
module remains useful for the polynomial ridge baseline and its diagnostics.
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from constants import (
    DATASET_DIR,
    MAPPER_FEATURE_KEYS,
    POLYNOMIAL_DEGREE,
    RIDGE_ALPHA,
    ROOT_DIR,
    SPLIT_GRID_COLS,
    SPLIT_GRID_ROWS,
)
from mapper import (
    PolynomialRidgeGazeMapper,
    build_design_matrix,
    fit_ridge_regression,
    latest_session_dirs,
)


EVAL_MODES = ["all", "label_holdout", "session_holdout", "region_holdout"]
HEAD_MOTION_KEYS = [
    "head_yaw_deg",
    "head_pitch_deg",
    "head_roll_deg",
    "head_tx",
    "head_ty",
    "head_tz",
]
REGION_ROWS = ["top", "middle", "bottom"]
REGION_COLS = ["left", "center", "right"]


@dataclass(frozen=True)
class CaptureRecord:
    session_id: str
    capture_index: int
    source_split: str
    target_x: float
    target_y: float
    samples: list[dict]


@dataclass
class SampleEval:
    session_id: str
    capture_index: int
    source_split: str
    holdout_group: str
    target_x: float
    target_y: float
    pred_x: float
    pred_y: float
    error_x_px: float
    error_y_px: float
    error_distance_px: float


@dataclass
class CaptureEval:
    session_id: str
    capture_index: int
    source_split: str
    holdout_group: str
    target_x: float
    target_y: float
    sample_count: int
    mean_error_x_px: float
    mean_error_y_px: float
    mean_error_distance_px: float
    head_rotation_std_deg: float
    head_translation_std: float
    head_motion_score: float


@dataclass
class FoldDefinition:
    holdout_group: str
    train_captures: list[CaptureRecord]
    eval_captures: list[CaptureRecord]


@dataclass
class ModeResult:
    mode: str
    screen_size: tuple[int, int]
    all_captures: list[CaptureRecord]
    eval_captures: list[CaptureRecord]
    folds: list[FoldDefinition]
    sample_evals: list[SampleEval]
    capture_evals: list[CaptureEval]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate eval diagnostics for gaze mapping.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help="Directory containing session-* folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "reports" / "eval" / "latest",
        help="Directory to write report outputs.",
    )
    parser.add_argument(
        "--mode",
        choices=EVAL_MODES,
        default="all",
        help="Evaluation split strategy.",
    )
    return parser.parse_args()


def latest_screen_size(dataset_dir: Path) -> tuple[int, int]:
    for session_dir in latest_session_dirs(dataset_dir):
        session_path = session_dir / "session.json"
        captures_path = session_dir / "captures.jsonl"
        if not session_path.exists() or not captures_path.exists():
            continue
        if captures_path.stat().st_size == 0:
            continue
        session = json.loads(session_path.read_text())
        return int(session["screen_width"]), int(session["screen_height"])
    raise ValueError("No captured sessions found")


def iter_session_captures(
    dataset_dir: Path,
    screen_size_filter: tuple[int, int],
) -> list[CaptureRecord]:
    captures: list[CaptureRecord] = []
    for session_dir in sorted(dataset_dir.glob("session-*")):
        if not session_dir.is_dir():
            continue

        session_path = session_dir / "session.json"
        captures_path = session_dir / "captures.jsonl"
        if not session_path.exists() or not captures_path.exists():
            continue

        session = json.loads(session_path.read_text())
        session_size = (int(session["screen_width"]), int(session["screen_height"]))
        if session_size != screen_size_filter:
            continue

        lines = [line for line in captures_path.read_text().splitlines() if line.strip()]
        for capture_index, line in enumerate(lines, 1):
            capture = json.loads(line)
            captures.append(
                CaptureRecord(
                    session_id=session_dir.name,
                    capture_index=capture_index,
                    source_split=capture.get("split", "unknown"),
                    target_x=float(capture["target_x"]),
                    target_y=float(capture["target_y"]),
                    samples=list(capture.get("samples", [])),
                )
            )
    return captures


def compatible_samples(capture: CaptureRecord) -> list[dict]:
    return [
        sample
        for sample in capture.samples
        if all(key in sample for key in MAPPER_FEATURE_KEYS)
    ]


def region_bins(target_x: float, target_y: float) -> tuple[int, int]:
    x_bin = min(int(np.clip(target_x, 0.0, 0.999999) * 3.0), 2)
    y_bin = min(int(np.clip(target_y, 0.0, 0.999999) * 3.0), 2)
    return x_bin, y_bin


def region_label(target_x: float, target_y: float) -> str:
    x_bin, y_bin = region_bins(target_x, target_y)
    return f"{REGION_ROWS[y_bin]}-{REGION_COLS[x_bin]}"


def edge_class(target_x: float, target_y: float) -> str:
    x_bin, y_bin = region_bins(target_x, target_y)
    if x_bin == 1 and y_bin == 1:
        return "center"
    if x_bin != 1 and y_bin != 1:
        return "corner"
    return "edge"


def safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def train_mapper_from_rows(
    train_rows: list[dict],
    screen_size: tuple[int, int],
    label: str,
) -> PolynomialRidgeGazeMapper:
    if len(train_rows) < len(MAPPER_FEATURE_KEYS) + 1:
        raise ValueError("Not enough compatible training samples")

    train_matrix = np.array(
        [[float(sample[key]) for key in MAPPER_FEATURE_KEYS] for sample in train_rows],
        dtype=np.float64,
    )
    design, base_mean, base_scale, pair_indices = build_design_matrix(train_matrix)
    target_x = np.array([float(sample["target_x"]) for sample in train_rows], dtype=np.float64)
    target_y = np.array([float(sample["target_y"]) for sample in train_rows], dtype=np.float64)
    coef_x = fit_ridge_regression(design, target_x, RIDGE_ALPHA)
    coef_y = fit_ridge_regression(design, target_y, RIDGE_ALPHA)

    return PolynomialRidgeGazeMapper(
        dataset_label=label,
        feature_keys=list(MAPPER_FEATURE_KEYS),
        screen_size=screen_size,
        base_mean=base_mean,
        base_scale=base_scale,
        pair_indices=pair_indices,
        coef_x=coef_x,
        coef_y=coef_y,
        train_sample_count=len(train_rows),
        eval_sample_count=0,
        ridge_alpha=RIDGE_ALPHA,
        eval_mae_x_px=None,
        eval_mae_y_px=None,
    )


def build_folds(mode: str, captures: list[CaptureRecord]) -> list[FoldDefinition]:
    if mode == "label_holdout":
        train_captures = [capture for capture in captures if capture.source_split != "eval"]
        eval_captures = [capture for capture in captures if capture.source_split == "eval"]
        if not eval_captures:
            raise ValueError("No eval captures available for label_holdout mode")
        return [
            FoldDefinition(
                holdout_group="collector_eval",
                train_captures=train_captures,
                eval_captures=eval_captures,
            )
        ]

    if mode == "session_holdout":
        folds = []
        session_ids = sorted({capture.session_id for capture in captures})
        for session_id in session_ids:
            eval_captures = [capture for capture in captures if capture.session_id == session_id]
            train_captures = [capture for capture in captures if capture.session_id != session_id]
            if eval_captures and train_captures:
                folds.append(
                    FoldDefinition(
                        holdout_group=session_id,
                        train_captures=train_captures,
                        eval_captures=eval_captures,
                    )
                )
        if not folds:
            raise ValueError("Need at least two sessions for session_holdout mode")
        return folds

    if mode == "region_holdout":
        folds = []
        labels = [f"{row}-{column}" for row in REGION_ROWS for column in REGION_COLS]
        for label in labels:
            eval_captures = [
                capture
                for capture in captures
                if region_label(capture.target_x, capture.target_y) == label
            ]
            train_captures = [
                capture
                for capture in captures
                if region_label(capture.target_x, capture.target_y) != label
            ]
            if eval_captures and train_captures:
                folds.append(
                    FoldDefinition(
                        holdout_group=label,
                        train_captures=train_captures,
                        eval_captures=eval_captures,
                    )
                )
        if not folds:
            raise ValueError("No populated regions available for region_holdout mode")
        return folds

    raise ValueError(f"Unsupported mode: {mode}")


def compute_head_scales(captures: list[CaptureRecord]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for capture in captures:
        for sample in compatible_samples(capture):
            for key in HEAD_MOTION_KEYS:
                if key in sample:
                    values[key].append(float(sample[key]))

    scales: dict[str, float] = {}
    for key in HEAD_MOTION_KEYS:
        scale = float(np.std(values[key])) if values[key] else 0.0
        scales[key] = scale if scale >= 1e-6 else 1.0
    return scales


def evaluate_capture(
    capture: CaptureRecord,
    mapper: PolynomialRidgeGazeMapper,
    screen_size: tuple[int, int],
    holdout_group: str,
    head_scales: dict[str, float],
) -> tuple[list[SampleEval], CaptureEval] | tuple[list, None]:
    width, height = screen_size
    samples = compatible_samples(capture)
    if not samples:
        return [], None

    head_std_values: dict[str, float] = {}
    for key in HEAD_MOTION_KEYS:
        values = [float(sample.get(key, 0.0)) for sample in samples if key in sample]
        head_std_values[key] = float(np.std(values)) if values else 0.0

    rotation_std_deg = math.sqrt(
        head_std_values["head_yaw_deg"] ** 2
        + head_std_values["head_pitch_deg"] ** 2
        + head_std_values["head_roll_deg"] ** 2
    )
    translation_std = math.sqrt(
        head_std_values["head_tx"] ** 2
        + head_std_values["head_ty"] ** 2
        + head_std_values["head_tz"] ** 2
    )
    head_motion_score = float(
        np.mean(
            [
                head_std_values[key] / head_scales[key]
                for key in HEAD_MOTION_KEYS
            ]
        )
    )

    sample_evals: list[SampleEval] = []
    for sample in samples:
        pred_x, pred_y = mapper.predict_normalized(sample)
        target_x = float(sample["target_x"])
        target_y = float(sample["target_y"])
        error_x_px = abs(pred_x - target_x) * width
        error_y_px = abs(pred_y - target_y) * height
        error_distance_px = math.sqrt(error_x_px**2 + error_y_px**2)
        sample_evals.append(
            SampleEval(
                session_id=capture.session_id,
                capture_index=capture.capture_index,
                source_split=capture.source_split,
                holdout_group=holdout_group,
                target_x=target_x,
                target_y=target_y,
                pred_x=pred_x,
                pred_y=pred_y,
                error_x_px=error_x_px,
                error_y_px=error_y_px,
                error_distance_px=error_distance_px,
            )
        )

    capture_eval = CaptureEval(
        session_id=capture.session_id,
        capture_index=capture.capture_index,
        source_split=capture.source_split,
        holdout_group=holdout_group,
        target_x=capture.target_x,
        target_y=capture.target_y,
        sample_count=len(sample_evals),
        mean_error_x_px=safe_mean([sample.error_x_px for sample in sample_evals]),
        mean_error_y_px=safe_mean([sample.error_y_px for sample in sample_evals]),
        mean_error_distance_px=safe_mean(
            [sample.error_distance_px for sample in sample_evals]
        ),
        head_rotation_std_deg=rotation_std_deg,
        head_translation_std=translation_std,
        head_motion_score=head_motion_score,
    )
    return sample_evals, capture_eval


def execute_mode(
    mode: str,
    captures: list[CaptureRecord],
    screen_size: tuple[int, int],
) -> ModeResult:
    folds = build_folds(mode, captures)
    eval_captures_map = {
        (capture.session_id, capture.capture_index): capture
        for fold in folds
        for capture in fold.eval_captures
    }
    eval_captures = list(eval_captures_map.values())
    head_scales = compute_head_scales(eval_captures)

    sample_evals: list[SampleEval] = []
    capture_evals: list[CaptureEval] = []
    valid_folds: list[FoldDefinition] = []

    for fold in folds:
        train_rows = [
            sample
            for capture in fold.train_captures
            for sample in compatible_samples(capture)
        ]
        if len(train_rows) < len(MAPPER_FEATURE_KEYS) + 1:
            continue

        mapper = train_mapper_from_rows(
            train_rows,
            screen_size=screen_size,
            label=f"{mode}:{fold.holdout_group}",
        )
        valid_folds.append(fold)

        for capture in fold.eval_captures:
            fold_sample_evals, capture_eval = evaluate_capture(
                capture,
                mapper,
                screen_size,
                holdout_group=fold.holdout_group,
                head_scales=head_scales,
            )
            sample_evals.extend(fold_sample_evals)
            if capture_eval is not None:
                capture_evals.append(capture_eval)

    if not sample_evals:
        raise ValueError(f"No eval samples produced for mode {mode}")

    return ModeResult(
        mode=mode,
        screen_size=screen_size,
        all_captures=captures,
        eval_captures=eval_captures,
        folds=valid_folds,
        sample_evals=sample_evals,
        capture_evals=capture_evals,
    )


def summarize_regions(samples: list[SampleEval]) -> list[dict]:
    grouped: dict[str, list[SampleEval]] = defaultdict(list)
    for sample in samples:
        grouped[region_label(sample.target_x, sample.target_y)].append(sample)

    summaries = []
    for row_label in REGION_ROWS:
        for col_label in REGION_COLS:
            label = f"{row_label}-{col_label}"
            group = grouped.get(label, [])
            summaries.append(
                {
                    "region": label,
                    "sample_count": len(group),
                    "mae_x_px": safe_mean([sample.error_x_px for sample in group]),
                    "mae_y_px": safe_mean([sample.error_y_px for sample in group]),
                    "mae_distance_px": safe_mean(
                        [sample.error_distance_px for sample in group]
                    ),
                }
            )
    return summaries


def summarize_edge_classes(samples: list[SampleEval]) -> list[dict]:
    grouped: dict[str, list[SampleEval]] = defaultdict(list)
    for sample in samples:
        grouped[edge_class(sample.target_x, sample.target_y)].append(sample)

    summaries = []
    for label in ["center", "edge", "corner"]:
        group = grouped.get(label, [])
        summaries.append(
            {
                "class": label,
                "sample_count": len(group),
                "mae_distance_px": safe_mean(
                    [sample.error_distance_px for sample in group]
                ),
            }
        )
    return summaries


def summarize_sessions(samples: list[SampleEval]) -> list[dict]:
    grouped: dict[str, list[SampleEval]] = defaultdict(list)
    for sample in samples:
        grouped[sample.session_id].append(sample)

    summaries = []
    for session_id, group in sorted(grouped.items()):
        summaries.append(
            {
                "session_id": session_id,
                "sample_count": len(group),
                "mae_x_px": safe_mean([sample.error_x_px for sample in group]),
                "mae_y_px": safe_mean([sample.error_y_px for sample in group]),
                "mae_distance_px": safe_mean(
                    [sample.error_distance_px for sample in group]
                ),
            }
        )
    return summaries


def summarize_holdout_groups(samples: list[SampleEval]) -> list[dict]:
    grouped: dict[str, list[SampleEval]] = defaultdict(list)
    for sample in samples:
        grouped[sample.holdout_group].append(sample)

    summaries = []
    for holdout_group, group in sorted(grouped.items()):
        summaries.append(
            {
                "holdout_group": holdout_group,
                "sample_count": len(group),
                "mae_x_px": safe_mean([sample.error_x_px for sample in group]),
                "mae_y_px": safe_mean([sample.error_y_px for sample in group]),
                "mae_distance_px": safe_mean(
                    [sample.error_distance_px for sample in group]
                ),
            }
        )
    return summaries


def summarize_motion(captures: list[CaptureEval]) -> dict:
    if not captures:
        return {"correlation": float("nan"), "bands": []}

    scores = np.array([capture.head_motion_score for capture in captures], dtype=np.float64)
    errors = np.array([capture.mean_error_distance_px for capture in captures], dtype=np.float64)
    correlation = (
        float(np.corrcoef(scores, errors)[0, 1]) if len(captures) >= 2 else float("nan")
    )

    if len(captures) >= 3:
        low_cut, high_cut = np.quantile(scores, [1.0 / 3.0, 2.0 / 3.0])
    else:
        low_cut, high_cut = scores.min(), scores.max()

    bands = {"low": [], "medium": [], "high": []}
    for capture in captures:
        if capture.head_motion_score <= low_cut:
            bands["low"].append(capture)
        elif capture.head_motion_score <= high_cut:
            bands["medium"].append(capture)
        else:
            bands["high"].append(capture)

    band_summaries = []
    for label in ["low", "medium", "high"]:
        group = bands[label]
        band_summaries.append(
            {
                "band": label,
                "capture_count": len(group),
                "mean_motion_score": safe_mean([capture.head_motion_score for capture in group]),
                "mae_distance_px": safe_mean(
                    [capture.mean_error_distance_px for capture in group]
                ),
            }
        )

    return {
        "correlation": correlation,
        "bands": band_summaries,
    }


def summarize_worst_captures(captures: list[CaptureEval], limit: int = 10) -> list[dict]:
    ranked = sorted(captures, key=lambda capture: capture.mean_error_distance_px, reverse=True)
    output = []
    for capture in ranked[:limit]:
        output.append(
            {
                "session_id": capture.session_id,
                "capture_index": capture.capture_index,
                "source_split": capture.source_split,
                "holdout_group": capture.holdout_group,
                "target_x": capture.target_x,
                "target_y": capture.target_y,
                "sample_count": capture.sample_count,
                "mae_x_px": capture.mean_error_x_px,
                "mae_y_px": capture.mean_error_y_px,
                "mae_distance_px": capture.mean_error_distance_px,
                "head_motion_score": capture.head_motion_score,
                "head_rotation_std_deg": capture.head_rotation_std_deg,
                "head_translation_std": capture.head_translation_std,
            }
        )
    return output


def make_summary(result: ModeResult) -> dict:
    sample_evals = result.sample_evals
    capture_evals = result.capture_evals

    available_sample_count = sum(len(compatible_samples(capture)) for capture in result.all_captures)
    unique_eval_sample_count = sum(len(compatible_samples(capture)) for capture in result.eval_captures)
    mean_train_capture_count = safe_mean([len(fold.train_captures) for fold in result.folds])
    mean_train_sample_count = safe_mean(
        [
            sum(len(compatible_samples(capture)) for capture in fold.train_captures)
            for fold in result.folds
        ]
    )

    summary = {
        "dataset": {
            "mode": result.mode,
            "screen_size": {
                "width": result.screen_size[0],
                "height": result.screen_size[1],
            },
            "session_count": len({capture.session_id for capture in result.all_captures}),
            "available_capture_count": len(result.all_captures),
            "available_sample_count": available_sample_count,
            "unique_eval_capture_count": len(result.eval_captures),
            "unique_eval_sample_count": unique_eval_sample_count,
            "stored_eval_capture_count": sum(
                1 for capture in result.all_captures if capture.source_split == "eval"
            ),
            "fold_count": len(result.folds),
            "mean_train_capture_count_per_fold": mean_train_capture_count,
            "mean_train_sample_count_per_fold": mean_train_sample_count,
        },
        "model": {
            "type": "polynomial_ridge",
            "ridge_alpha": RIDGE_ALPHA,
            "feature_count": len(MAPPER_FEATURE_KEYS),
            "polynomial_degree": POLYNOMIAL_DEGREE,
        },
        "overall": {
            "mae_x_px": safe_mean([sample.error_x_px for sample in sample_evals]),
            "mae_y_px": safe_mean([sample.error_y_px for sample in sample_evals]),
            "mae_distance_px": safe_mean(
                [sample.error_distance_px for sample in sample_evals]
            ),
        },
        "holdout_groups": summarize_holdout_groups(sample_evals),
        "regions": summarize_regions(sample_evals),
        "edge_classes": summarize_edge_classes(sample_evals),
        "sessions": summarize_sessions(sample_evals),
        "motion": summarize_motion(capture_evals),
        "worst_captures": summarize_worst_captures(capture_evals),
    }
    return summary


def mode_title(mode: str) -> str:
    return mode.replace("_", " ").title()


def write_summary_text(summary: dict, output_path: Path) -> None:
    lines = []
    dataset = summary["dataset"]
    overall = summary["overall"]

    lines.append(f"Eval Report ({mode_title(dataset['mode'])})")
    lines.append("")
    lines.append(
        f"Screen: {dataset['screen_size']['width']}x{dataset['screen_size']['height']}"
    )
    lines.append(
        f"Available dataset: {dataset['available_capture_count']} captures, "
        f"{dataset['available_sample_count']} compatible samples across {dataset['session_count']} sessions"
    )
    lines.append(
        f"Eval coverage: {dataset['unique_eval_capture_count']} captures, "
        f"{dataset['unique_eval_sample_count']} samples across {dataset['fold_count']} folds"
    )
    lines.append(
        f"Average train set per fold: {dataset['mean_train_capture_count_per_fold']:.1f} captures, "
        f"{dataset['mean_train_sample_count_per_fold']:.1f} samples"
    )
    lines.append("")
    lines.append(
        f"Overall eval MAE: {overall['mae_x_px']:.1f}px x | "
        f"{overall['mae_y_px']:.1f}px y | {overall['mae_distance_px']:.1f}px distance"
    )
    lines.append("")

    group_ranked = sorted(
        [
            group
            for group in summary["holdout_groups"]
            if not math.isnan(group["mae_distance_px"])
        ],
        key=lambda group: group["mae_distance_px"],
        reverse=True,
    )
    lines.append("Holdout Groups")
    for group in group_ranked[:10]:
        lines.append(
            f"- {group['holdout_group']}: {group['mae_distance_px']:.1f}px "
            f"({group['sample_count']} eval samples)"
        )
    lines.append("")

    region_ranked = sorted(
        [
            region
            for region in summary["regions"]
            if not math.isnan(region["mae_distance_px"])
        ],
        key=lambda region: region["mae_distance_px"],
        reverse=True,
    )
    lines.append("Worst Regions")
    for region in region_ranked[:5]:
        lines.append(
            f"- {region['region']}: {region['mae_distance_px']:.1f}px "
            f"({region['sample_count']} eval samples)"
        )
    lines.append("")

    lines.append("Edge Classes")
    for entry in summary["edge_classes"]:
        value = entry["mae_distance_px"]
        value_text = "n/a" if math.isnan(value) else f"{value:.1f}px"
        lines.append(
            f"- {entry['class']}: {value_text} ({entry['sample_count']} eval samples)"
        )
    lines.append("")

    lines.append("Eval Sessions")
    for session in summary["sessions"]:
        lines.append(
            f"- {session['session_id']}: {session['mae_distance_px']:.1f}px "
            f"({session['sample_count']} eval samples)"
        )
    lines.append("")

    motion = summary["motion"]
    corr = motion["correlation"]
    corr_text = "n/a" if math.isnan(corr) else f"{corr:.2f}"
    lines.append(f"Head Motion vs Error Correlation: {corr_text}")
    for band in motion["bands"]:
        value = band["mae_distance_px"]
        value_text = "n/a" if math.isnan(value) else f"{value:.1f}px"
        lines.append(
            f"- {band['band']}: {value_text} ({band['capture_count']} eval captures)"
        )
    lines.append("")

    lines.append("Worst Eval Captures")
    for capture in summary["worst_captures"][:10]:
        lines.append(
            f"- {capture['session_id']} #{capture['capture_index']} "
            f"[{capture['holdout_group']}] "
            f"target=({capture['target_x']:.3f}, {capture['target_y']:.3f}) "
            f"error={capture['mae_distance_px']:.1f}px motion={capture['head_motion_score']:.2f}"
        )
    lines.append("")

    output_path.write_text("\n".join(lines))


def plot_actual_vs_pred(
    samples: list[SampleEval],
    screen_size: tuple[int, int],
    output_path: Path,
    title: str,
) -> None:
    width, height = screen_size
    fig, ax = plt.subplots(figsize=(9, 6))

    actual_x = np.array([sample.target_x * (width - 1) for sample in samples], dtype=np.float64)
    actual_y = np.array([sample.target_y * (height - 1) for sample in samples], dtype=np.float64)
    pred_x = np.array([sample.pred_x * (width - 1) for sample in samples], dtype=np.float64)
    pred_y = np.array([sample.pred_y * (height - 1) for sample in samples], dtype=np.float64)
    error = np.array([sample.error_distance_px for sample in samples], dtype=np.float64)

    ax.scatter(actual_x, actual_y, s=8, c="#a0a0a0", alpha=0.35, label="actual")
    scatter = ax.scatter(pred_x, pred_y, s=10, c=error, cmap="viridis", alpha=0.8, label="predicted")
    ax.set_xlim(0, width - 1)
    ax.set_ylim(height - 1, 0)
    ax.set_title(f"{title}: Actual vs Predicted")
    ax.set_xlabel("screen x (px)")
    ax.set_ylabel("screen y (px)")
    ax.legend(loc="upper right")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("error distance (px)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_error_heatmap(samples: list[SampleEval], output_path: Path, title: str) -> None:
    error_sum = np.zeros((SPLIT_GRID_ROWS, SPLIT_GRID_COLS), dtype=np.float64)
    count = np.zeros((SPLIT_GRID_ROWS, SPLIT_GRID_COLS), dtype=np.float64)

    for sample in samples:
        column = min(int(np.clip(sample.target_x, 0.0, 0.999999) * SPLIT_GRID_COLS), SPLIT_GRID_COLS - 1)
        row = min(int(np.clip(sample.target_y, 0.0, 0.999999) * SPLIT_GRID_ROWS), SPLIT_GRID_ROWS - 1)
        error_sum[row, column] += sample.error_distance_px
        count[row, column] += 1.0

    mean_error = np.divide(
        error_sum,
        count,
        out=np.full_like(error_sum, np.nan),
        where=count > 0,
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    image = ax.imshow(mean_error, cmap="magma", origin="upper")
    ax.set_title(f"{title}: Mean Error by Target Cell")
    ax.set_xlabel("x cell")
    ax.set_ylabel("y cell")
    ax.set_xticks(range(SPLIT_GRID_COLS))
    ax.set_yticks(range(SPLIT_GRID_ROWS))
    for row in range(SPLIT_GRID_ROWS):
        for column in range(SPLIT_GRID_COLS):
            if count[row, column] <= 0:
                continue
            ax.text(
                column,
                row,
                f"{mean_error[row, column]:.0f}\n({int(count[row, column])})",
                ha="center",
                va="center",
                color="white",
                fontsize=7,
            )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("mean error distance (px)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_region_bars(summary: dict, output_path: Path, title: str) -> None:
    labels = [region["region"] for region in summary["regions"]]
    values = [region["mae_distance_px"] for region in summary["regions"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values, color="#4e79a7")
    ax.set_title(f"{title}: Mean Error by 3x3 Region")
    ax.set_ylabel("mean error distance (px)")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_motion_vs_error(captures: list[CaptureEval], output_path: Path, title: str) -> None:
    if not captures:
        return

    x = np.array([capture.head_motion_score for capture in captures], dtype=np.float64)
    y = np.array([capture.mean_error_distance_px for capture in captures], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, color="#f28e2b", alpha=0.8)
    if len(captures) >= 2 and float(np.std(x)) > 1e-9:
        slope, intercept = np.polyfit(x, y, 1)
        line_x = np.linspace(np.min(x), np.max(x), 100)
        line_y = slope * line_x + intercept
        ax.plot(line_x, line_y, color="#2f2f2f", linewidth=1.5)
    ax.set_title(f"{title}: Capture Error vs Head Motion")
    ax.set_xlabel("normalized head motion score")
    ax.set_ylabel("mean error distance (px)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_mode_report(
    dataset_dir: Path,
    output_dir: Path,
    mode: str,
) -> Path:
    screen_size = latest_screen_size(dataset_dir)
    captures = iter_session_captures(dataset_dir, screen_size)
    result = execute_mode(mode, captures, screen_size)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = make_summary(result)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_summary_text(summary, output_dir / "summary.txt")

    title = mode_title(mode)
    plot_actual_vs_pred(result.sample_evals, screen_size, output_dir / "eval_actual_vs_pred.png", title)
    plot_error_heatmap(result.sample_evals, output_dir / "eval_error_heatmap.png", title)
    plot_region_bars(summary, output_dir / "eval_region_bars.png", title)
    plot_motion_vs_error(result.capture_evals, output_dir / "eval_motion_vs_error.png", title)

    return output_dir


def build_all_reports(dataset_dir: Path, output_dir: Path) -> Path:
    concrete_modes = [mode for mode in EVAL_MODES if mode != "all"]
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index = {}
    for mode in concrete_modes:
        mode_dir = output_dir / mode
        build_mode_report(dataset_dir, mode_dir, mode)
        summary = json.loads((mode_dir / "summary.json").read_text())
        index[mode] = {
            "overall": summary["overall"],
            "summary_path": str(mode_dir / "summary.txt"),
        }

    (output_dir / "index.json").write_text(json.dumps(index, indent=2))
    lines = ["Eval Report Index", ""]
    for mode in concrete_modes:
        overall = index[mode]["overall"]
        lines.append(
            f"- {mode}: {overall['mae_x_px']:.1f}px x | "
            f"{overall['mae_y_px']:.1f}px y | {overall['mae_distance_px']:.1f}px distance"
        )
    lines.append("")
    (output_dir / "index.txt").write_text("\n".join(lines))
    return output_dir


def main() -> None:
    args = parse_args()
    if args.mode == "all":
        output_dir = build_all_reports(args.dataset_dir, args.output_dir)
    else:
        output_dir = build_mode_report(args.dataset_dir, args.output_dir, args.mode)
    print(output_dir)


if __name__ == "__main__":
    main()
