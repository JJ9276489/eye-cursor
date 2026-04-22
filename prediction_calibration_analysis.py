from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from constants import (
    DATASET_DIR,
    ROOT_DIR,
    VISION_BATCH_SIZE,
    VISION_SPATIAL_GEOMETRY_MODEL_PATH,
)
from vision_dataset import (
    EyeCropDataset,
    VisionCapture,
    VisionSample,
    collect_vision_captures,
)
from vision_eval_report import REGION_COLS, REGION_ROWS, region_label
from vision_runtime import load_vision_predictor


REGION_LABELS = [f"{row}-{column}" for row in REGION_ROWS for column in REGION_COLS]


@dataclass(frozen=True)
class CapturePrediction:
    capture_id: str
    split: str
    region: str
    target_x: float
    target_y: float
    pred_x: float
    pred_y: float
    sample_count: int


@dataclass(frozen=True)
class AxisCalibrator:
    kind: str
    x_params: dict[str, float]
    y_params: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose edge compression and post-hoc calibration for NN gaze predictions."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--checkpoint", type=Path, default=VISION_SPATIAL_GEOMETRY_MODEL_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "reports" / "calibration" / "latest",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=VISION_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.5,
        help="Fraction of eval captures held out when simulating a calibration/test split.",
    )
    parser.add_argument(
        "--split-repeats",
        type=int,
        default=12,
        help="Number of random eval calibration/test splits to average.",
    )
    return parser.parse_args()


def split_captures(captures: list[VisionCapture]) -> tuple[list[VisionCapture], list[VisionCapture]]:
    train_captures = [capture for capture in captures if capture.split != "eval"]
    eval_captures = [capture for capture in captures if capture.split == "eval"]
    if not train_captures:
        raise ValueError("No train captures found")
    if not eval_captures:
        raise ValueError("No eval captures found")
    return train_captures, eval_captures


def flatten_samples(captures: list[VisionCapture]) -> list[VisionSample]:
    return [sample for capture in captures for sample in capture.samples]


def capture_id(capture: VisionCapture) -> str:
    return f"{capture.session_id}:{capture.capture_index}"


def predict_samples(
    samples: list[VisionSample],
    predictor,
    batch_size: int,
) -> np.ndarray:
    dataset = EyeCropDataset(
        samples,
        head_mean=predictor.head_mean,
        head_scale=predictor.head_scale,
        augment=False,
        extra_feature_keys=tuple(predictor.extra_feature_keys),
        extra_mean=predictor.extra_mean,
        extra_scale=predictor.extra_scale,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    outputs: list[np.ndarray] = []
    predictor.model.eval()
    with torch.no_grad():
        for batch in loader:
            left_eye = batch["left_eye"].to(predictor.device)
            right_eye = batch["right_eye"].to(predictor.device)
            head_features = batch["head_features"].to(predictor.device)
            extra_features = batch["extra_features"].to(predictor.device)
            prediction = predictor.model(left_eye, right_eye, head_features, extra_features)
            outputs.append(prediction.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def capture_predictions(
    captures: list[VisionCapture],
    predictor,
    batch_size: int,
) -> list[CapturePrediction]:
    samples = flatten_samples(captures)
    predictions = predict_samples(samples, predictor, batch_size)
    sample_predictions: dict[str, list[tuple[float, float]]] = {}
    sample_index = 0
    for capture in captures:
        key = capture_id(capture)
        rows: list[tuple[float, float]] = []
        for _ in capture.samples:
            rows.append((float(predictions[sample_index, 0]), float(predictions[sample_index, 1])))
            sample_index += 1
        sample_predictions[key] = rows

    capture_rows: list[CapturePrediction] = []
    for capture in captures:
        key = capture_id(capture)
        rows = sample_predictions[key]
        pred_x = float(np.mean([row[0] for row in rows]))
        pred_y = float(np.mean([row[1] for row in rows]))
        capture_rows.append(
            CapturePrediction(
                capture_id=key,
                split=capture.split,
                region=region_label(capture.target_x, capture.target_y),
                target_x=float(capture.target_x),
                target_y=float(capture.target_y),
                pred_x=pred_x,
                pred_y=pred_y,
                sample_count=len(rows),
            )
        )
    return capture_rows


def as_arrays(rows: list[CapturePrediction]) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray([[row.target_x, row.target_y] for row in rows], dtype=np.float64)
    preds = np.asarray([[row.pred_x, row.pred_y] for row in rows], dtype=np.float64)
    return targets, preds


def clamp01(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 1.0)


def safe_logit(values: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    clipped = np.clip(values, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def fit_linear(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    design = np.column_stack([pred, np.ones_like(pred)])
    slope, intercept = np.linalg.lstsq(design, target, rcond=None)[0]
    return {"slope": float(slope), "intercept": float(intercept)}


def fit_logit_affine(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    design = np.column_stack([safe_logit(pred), np.ones_like(pred)])
    slope, intercept = np.linalg.lstsq(design, safe_logit(target), rcond=None)[0]
    return {"slope": float(slope), "intercept": float(intercept)}


def centered_power(values: np.ndarray, gamma: float) -> np.ndarray:
    centered = np.clip(2.0 * values - 1.0, -1.0, 1.0)
    transformed = np.sign(centered) * (np.abs(centered) ** gamma)
    return clamp01((transformed + 1.0) / 2.0)


def fit_power(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    best_gamma = 1.0
    best_loss = float("inf")
    for gamma in np.linspace(0.35, 2.0, 166):
        corrected = centered_power(pred, float(gamma))
        loss = float(np.mean((corrected - target) ** 2))
        if loss < best_loss:
            best_gamma = float(gamma)
            best_loss = loss
    return {"gamma": best_gamma}


def fit_calibrator(kind: str, rows: list[CapturePrediction]) -> AxisCalibrator:
    targets, preds = as_arrays(rows)
    if kind == "identity":
        return AxisCalibrator(kind=kind, x_params={}, y_params={})
    if kind == "linear":
        return AxisCalibrator(
            kind=kind,
            x_params=fit_linear(preds[:, 0], targets[:, 0]),
            y_params=fit_linear(preds[:, 1], targets[:, 1]),
        )
    if kind == "logit_affine":
        return AxisCalibrator(
            kind=kind,
            x_params=fit_logit_affine(preds[:, 0], targets[:, 0]),
            y_params=fit_logit_affine(preds[:, 1], targets[:, 1]),
        )
    if kind == "centered_power":
        return AxisCalibrator(
            kind=kind,
            x_params=fit_power(preds[:, 0], targets[:, 0]),
            y_params=fit_power(preds[:, 1], targets[:, 1]),
        )
    raise ValueError(f"Unsupported calibrator kind: {kind}")


def apply_axis(values: np.ndarray, params: dict[str, float], kind: str) -> np.ndarray:
    if kind == "identity":
        return values
    if kind == "linear":
        return clamp01(params["slope"] * values + params["intercept"])
    if kind == "logit_affine":
        return sigmoid(params["slope"] * safe_logit(values) + params["intercept"])
    if kind == "centered_power":
        return centered_power(values, params["gamma"])
    raise ValueError(f"Unsupported calibrator kind: {kind}")


def apply_calibrator(preds: np.ndarray, calibrator: AxisCalibrator) -> np.ndarray:
    return np.column_stack(
        [
            apply_axis(preds[:, 0], calibrator.x_params, calibrator.kind),
            apply_axis(preds[:, 1], calibrator.y_params, calibrator.kind),
        ]
    )


def metric_summary(
    rows: list[CapturePrediction],
    calibrated_preds: np.ndarray,
    screen_size: tuple[int, int],
) -> dict[str, float]:
    targets, _ = as_arrays(rows)
    pixel_errors = np.abs(calibrated_preds - targets) * np.asarray(screen_size, dtype=np.float64)
    distances = np.sqrt(pixel_errors[:, 0] ** 2 + pixel_errors[:, 1] ** 2)
    return {
        "capture_count": len(rows),
        "capture_mae_distance_px": float(np.mean(distances)),
        "capture_mae_x_px": float(np.mean(pixel_errors[:, 0])),
        "capture_mae_y_px": float(np.mean(pixel_errors[:, 1])),
    }


def compression_summary(rows: list[CapturePrediction], screen_size: tuple[int, int]) -> dict:
    targets, preds = as_arrays(rows)
    x_fit = fit_linear(targets[:, 0], preds[:, 0])
    y_fit = fit_linear(targets[:, 1], preds[:, 1])

    low_x = targets[:, 0] <= 1.0 / 3.0
    high_x = targets[:, 0] >= 2.0 / 3.0
    low_y = targets[:, 1] <= 1.0 / 3.0
    high_y = targets[:, 1] >= 2.0 / 3.0
    width, height = screen_size

    def mean_mask(values: np.ndarray, mask: np.ndarray) -> float:
        return float(np.mean(values[mask])) if np.any(mask) else float("nan")

    return {
        "pred_vs_target_x": x_fit,
        "pred_vs_target_y": y_fit,
        "pred_range": {
            "x_min": float(np.min(preds[:, 0])),
            "x_max": float(np.max(preds[:, 0])),
            "y_min": float(np.min(preds[:, 1])),
            "y_max": float(np.max(preds[:, 1])),
        },
        "target_range": {
            "x_min": float(np.min(targets[:, 0])),
            "x_max": float(np.max(targets[:, 0])),
            "y_min": float(np.min(targets[:, 1])),
            "y_max": float(np.max(targets[:, 1])),
        },
        "inward_bias_px": {
            "left_x": mean_mask((preds[:, 0] - targets[:, 0]) * width, low_x),
            "right_x": mean_mask((targets[:, 0] - preds[:, 0]) * width, high_x),
            "top_y": mean_mask((preds[:, 1] - targets[:, 1]) * height, low_y),
            "bottom_y": mean_mask((targets[:, 1] - preds[:, 1]) * height, high_y),
        },
    }


def split_eval_for_calibration(
    eval_rows: list[CapturePrediction],
    seed: int,
    test_fraction: float,
) -> tuple[list[CapturePrediction], list[CapturePrediction]]:
    if not 0.1 <= test_fraction <= 0.9:
        raise ValueError("--test-fraction must be between 0.1 and 0.9")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(eval_rows))
    test_count = max(1, int(round(len(eval_rows) * test_fraction)))
    test_indices = set(int(index) for index in indices[:test_count])
    calibration = [row for index, row in enumerate(eval_rows) if index not in test_indices]
    test = [row for index, row in enumerate(eval_rows) if index in test_indices]
    return calibration, test


def evaluate_calibrators(
    fit_rows: list[CapturePrediction],
    eval_rows: list[CapturePrediction],
    screen_size: tuple[int, int],
) -> dict[str, dict]:
    _, raw_eval_preds = as_arrays(eval_rows)
    output: dict[str, dict] = {}
    for kind in ("identity", "linear", "logit_affine", "centered_power"):
        calibrator = fit_calibrator(kind, fit_rows)
        calibrated = apply_calibrator(raw_eval_preds, calibrator)
        output[kind] = {
            "params": {
                "x": calibrator.x_params,
                "y": calibrator.y_params,
            },
            "metrics": metric_summary(eval_rows, calibrated, screen_size),
        }
    return output


def repeated_eval_calibration_tests(
    eval_rows: list[CapturePrediction],
    screen_size: tuple[int, int],
    seed: int,
    test_fraction: float,
    repeats: int,
) -> dict[str, dict]:
    if repeats <= 0:
        raise ValueError("--split-repeats must be positive")

    metric_values: dict[str, dict[str, list[float]]] = {
        kind: {
            "capture_mae_distance_px": [],
            "capture_mae_x_px": [],
            "capture_mae_y_px": [],
        }
        for kind in ("identity", "linear", "logit_affine", "centered_power")
    }
    capture_counts: dict[str, list[int]] = {kind: [] for kind in metric_values}

    for offset in range(repeats):
        calibration_rows, test_rows = split_eval_for_calibration(
            eval_rows,
            seed=seed + offset,
            test_fraction=test_fraction,
        )
        split_results = evaluate_calibrators(calibration_rows, test_rows, screen_size)
        for kind, result in split_results.items():
            metrics = result["metrics"]
            capture_counts[kind].append(int(metrics["capture_count"]))
            for key in metric_values[kind]:
                metric_values[kind][key].append(float(metrics[key]))

    aggregate: dict[str, dict] = {}
    for kind, values_by_metric in metric_values.items():
        aggregate[kind] = {
            "test_capture_count_mean": float(np.mean(capture_counts[kind])),
            "metrics": {},
        }
        for metric_name, values in values_by_metric.items():
            aggregate[kind]["metrics"][metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "runs": values,
            }
    return aggregate


def region_metrics(
    rows: list[CapturePrediction],
    calibrated_preds: np.ndarray,
    screen_size: tuple[int, int],
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[int]] = {label: [] for label in REGION_LABELS}
    for index, row in enumerate(rows):
        grouped[row.region].append(index)
    summary: dict[str, dict[str, float]] = {}
    for label, indices in grouped.items():
        if not indices:
            summary[label] = {
                "capture_count": 0,
                "capture_mae_distance_px": float("nan"),
                "capture_mae_x_px": float("nan"),
                "capture_mae_y_px": float("nan"),
            }
            continue
        subset_rows = [rows[index] for index in indices]
        subset_preds = calibrated_preds[indices]
        summary[label] = metric_summary(subset_rows, subset_preds, screen_size)
    return summary


def write_region_csv(raw_regions: dict, calibrated_regions: dict, output_path: Path) -> None:
    columns = [
        "region",
        "capture_count",
        "raw_capture_mae_distance_px",
        "calibrated_capture_mae_distance_px",
        "delta_px",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for label in REGION_LABELS:
            raw = raw_regions[label]["capture_mae_distance_px"]
            calibrated = calibrated_regions[label]["capture_mae_distance_px"]
            writer.writerow(
                {
                    "region": label,
                    "capture_count": raw_regions[label]["capture_count"],
                    "raw_capture_mae_distance_px": raw,
                    "calibrated_capture_mae_distance_px": calibrated,
                    "delta_px": calibrated - raw,
                }
            )


def plot_axis_scatter(
    rows: list[CapturePrediction],
    calibrated_preds: np.ndarray,
    output_path: Path,
) -> None:
    targets, raw_preds = as_arrays(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for axis_index, axis_name in enumerate(("x", "y")):
        axis = axes[axis_index]
        axis.scatter(
            targets[:, axis_index],
            raw_preds[:, axis_index],
            s=18,
            alpha=0.65,
            label="raw",
            color="#2563eb",
        )
        axis.scatter(
            targets[:, axis_index],
            calibrated_preds[:, axis_index],
            s=18,
            alpha=0.65,
            label="calibrated",
            color="#ea580c",
        )
        axis.plot([0, 1], [0, 1], color="#111827", linewidth=1, linestyle="--")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel(f"target {axis_name}")
        axis.set_ylabel(f"predicted {axis_name}")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Prediction Calibration")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_region_bars(raw_regions: dict, calibrated_regions: dict, output_path: Path) -> None:
    labels = REGION_LABELS
    x = np.arange(len(labels))
    raw = [raw_regions[label]["capture_mae_distance_px"] for label in labels]
    calibrated = [calibrated_regions[label]["capture_mae_distance_px"] for label in labels]
    fig, axis = plt.subplots(figsize=(11, 5))
    width = 0.38
    axis.bar(x - width / 2, raw, width=width, label="raw", color="#2563eb")
    axis.bar(x + width / 2, calibrated, width=width, label="calibrated", color="#ea580c")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=35, ha="right")
    axis.set_ylabel("Capture MAE distance (px)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.suptitle("Calibration by Screen Region")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def format_params(params: dict[str, float]) -> str:
    if not params:
        return "-"
    return ", ".join(f"{key}={value:.3f}" for key, value in params.items())


def write_summary_md(summary: dict, output_path: Path) -> None:
    train_eval = summary["train_fit_eval"]
    split_eval = summary["eval_calibration_test"]
    best_name = summary["best_calibrator"]
    compression = summary["compression"]
    lines = [
        "# Prediction Calibration Analysis",
        "",
        f"Checkpoint: `{summary['checkpoint']}`",
        f"Dataset: `{summary['capture_count']}` captures / `{summary['sample_count']}` samples",
        f"Screen: `{summary['screen_width']}x{summary['screen_height']}`",
        "",
        "## Compression Diagnostic",
        "",
        (
            "The raw NN output is already bounded by a sigmoid, so this checks whether "
            "predictions are being pulled inward from the screen edges."
        ),
        "",
        "| Axis | Raw pred-vs-target slope | Intercept | Raw predicted range | Target range |",
        "| --- | ---: | ---: | --- | --- |",
        (
            f"| `x` | `{compression['pred_vs_target_x']['slope']:.3f}` | "
            f"`{compression['pred_vs_target_x']['intercept']:.3f}` | "
            f"`{compression['pred_range']['x_min']:.3f}-{compression['pred_range']['x_max']:.3f}` | "
            f"`{compression['target_range']['x_min']:.3f}-{compression['target_range']['x_max']:.3f}` |"
        ),
        (
            f"| `y` | `{compression['pred_vs_target_y']['slope']:.3f}` | "
            f"`{compression['pred_vs_target_y']['intercept']:.3f}` | "
            f"`{compression['pred_range']['y_min']:.3f}-{compression['pred_range']['y_max']:.3f}` | "
            f"`{compression['target_range']['y_min']:.3f}-{compression['target_range']['y_max']:.3f}` |"
        ),
        "",
        "Positive inward bias means predictions are pulled toward the center.",
        "",
        "| Edge group | Inward bias |",
        "| --- | ---: |",
        f"| left x | `{compression['inward_bias_px']['left_x']:+.1f}px` |",
        f"| right x | `{compression['inward_bias_px']['right_x']:+.1f}px` |",
        f"| top y | `{compression['inward_bias_px']['top_y']:+.1f}px` |",
        f"| bottom y | `{compression['inward_bias_px']['bottom_y']:+.1f}px` |",
        "",
        "## Post-Hoc Calibration",
        "",
        "Calibrators were fit on train captures, then evaluated on collector eval captures.",
        "",
        "| Calibrator | Capture MAE | X MAE | Y MAE | X params | Y params |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for name, result in train_eval.items():
        metrics = result["metrics"]
        lines.append(
            f"| `{name}` | `{metrics['capture_mae_distance_px']:.1f}px` | "
            f"`{metrics['capture_mae_x_px']:.1f}px` | "
            f"`{metrics['capture_mae_y_px']:.1f}px` | "
            f"`{format_params(result['params']['x'])}` | "
            f"`{format_params(result['params']['y'])}` |"
        )

    lines.extend(
        [
            "",
            (
                "A second check repeatedly fit calibration on half of the collector eval "
                "captures and tested on the other half. This is closer to a live "
                "post-training calibration workflow, but each split has fewer test captures."
            ),
            "",
            "| Calibrator | Test Capture MAE | X MAE | Y MAE |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, result in split_eval.items():
        metrics = result["metrics"]
        lines.append(
            f"| `{name}` | "
            f"`{metrics['capture_mae_distance_px']['mean']:.1f}px +/- "
            f"{metrics['capture_mae_distance_px']['std']:.1f}` | "
            f"`{metrics['capture_mae_x_px']['mean']:.1f}px +/- "
            f"{metrics['capture_mae_x_px']['std']:.1f}` | "
            f"`{metrics['capture_mae_y_px']['mean']:.1f}px +/- "
            f"{metrics['capture_mae_y_px']['std']:.1f}` |"
        )

    lines.extend(
        [
            "",
            f"Best train-fit eval calibrator: `{best_name}`.",
            "",
            "## Region Effect",
            "",
            "| Region | Captures | Raw MAE | Calibrated MAE | Delta |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for label in REGION_LABELS:
        raw = summary["raw_regions"][label]
        calibrated = summary["best_regions"][label]
        delta = calibrated["capture_mae_distance_px"] - raw["capture_mae_distance_px"]
        lines.append(
            f"| `{label}` | `{raw['capture_count']}` | "
            f"`{raw['capture_mae_distance_px']:.1f}px` | "
            f"`{calibrated['capture_mae_distance_px']:.1f}px` | "
            f"`{delta:+.1f}px` |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            summary["interpretation"],
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")


def interpretation_from_summary(summary: dict) -> str:
    compression = summary["compression"]
    x_slope = compression["pred_vs_target_x"]["slope"]
    y_slope = compression["pred_vs_target_y"]["slope"]
    raw = summary["train_fit_eval"]["identity"]["metrics"]["capture_mae_distance_px"]
    best = summary["train_fit_eval"][summary["best_calibrator"]]["metrics"]["capture_mae_distance_px"]
    improvement = raw - best

    compression_text = (
        "Raw predictions show substantial inward compression"
        if x_slope < 0.85 or y_slope < 0.85
        else "Raw predictions are not strongly compressed by slope alone"
    )
    calibration_text = (
        f"Post-hoc calibration improves the train-fit eval score by {improvement:.1f}px, "
        "so a calibration warp is likely worth testing in the live preview."
        if improvement >= 5.0
        else f"Post-hoc calibration improves the train-fit eval score by only {improvement:.1f}px, "
        "so calibration is not the main failure mode on this checkpoint."
    )
    return f"{compression_text}. {calibration_text}"


def main() -> None:
    args = parse_args()
    captures, screen_size = collect_vision_captures(args.dataset_dir)
    if screen_size is None or not captures:
        raise ValueError("No compatible captures found")
    predictor = load_vision_predictor(
        checkpoint_path=args.checkpoint,
        screen_size_filter=screen_size,
        requested_device=args.device,
    )
    if predictor is None:
        raise ValueError(f"No compatible checkpoint found at {args.checkpoint}")

    train_captures, eval_captures = split_captures(captures)
    print(f"device: {predictor.device}")
    print(
        f"dataset: {len(captures)} captures / {sum(len(c.samples) for c in captures)} samples | "
        f"screen {screen_size[0]}x{screen_size[1]}"
    )
    print(f"checkpoint: {args.checkpoint}")
    print("predicting train captures...")
    train_rows = capture_predictions(train_captures, predictor, args.batch_size)
    print("predicting eval captures...")
    eval_rows = capture_predictions(eval_captures, predictor, args.batch_size)

    train_fit_eval = evaluate_calibrators(train_rows, eval_rows, screen_size)
    eval_split_results = repeated_eval_calibration_tests(
        eval_rows=eval_rows,
        screen_size=screen_size,
        seed=args.seed,
        test_fraction=args.test_fraction,
        repeats=args.split_repeats,
    )

    raw_targets, raw_eval_preds = as_arrays(eval_rows)
    del raw_targets
    best_name = min(
        train_fit_eval,
        key=lambda name: train_fit_eval[name]["metrics"]["capture_mae_distance_px"],
    )
    best_calibrator = fit_calibrator(best_name, train_rows)
    best_eval_preds = apply_calibrator(raw_eval_preds, best_calibrator)

    raw_regions = region_metrics(eval_rows, raw_eval_preds, screen_size)
    best_regions = region_metrics(eval_rows, best_eval_preds, screen_size)

    summary = {
        "checkpoint": str(args.checkpoint),
        "screen_width": screen_size[0],
        "screen_height": screen_size[1],
        "capture_count": len(captures),
        "sample_count": sum(len(capture.samples) for capture in captures),
        "train_capture_count": len(train_captures),
        "eval_capture_count": len(eval_captures),
        "eval_test_fraction": args.test_fraction,
        "eval_split_repeats": args.split_repeats,
        "compression": compression_summary(eval_rows, screen_size),
        "train_fit_eval": train_fit_eval,
        "eval_calibration_test": eval_split_results,
        "best_calibrator": best_name,
        "raw_regions": raw_regions,
        "best_regions": best_regions,
    }
    summary["interpretation"] = interpretation_from_summary(summary)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_summary_md(summary, output_dir / "summary.md")
    write_region_csv(raw_regions, best_regions, output_dir / "region_metrics.csv")
    plot_axis_scatter(eval_rows, best_eval_preds, output_dir / "axis_calibration.svg")
    plot_region_bars(raw_regions, best_regions, output_dir / "region_calibration.svg")

    print((output_dir / "summary.md").read_text())


if __name__ == "__main__":
    main()
