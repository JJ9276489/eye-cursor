from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil

import numpy as np
import torch

from constants import (
    DATASET_DIR,
    ROOT_DIR,
    VISION_BATCH_SIZE,
    VISION_EVAL_EPOCHS,
    VISION_LEARNING_RATE,
    VISION_WEIGHT_DECAY,
)
from vision_dataset import (
    VisionCapture,
    collect_vision_captures,
    compute_head_normalization,
    load_grayscale_eye_crop,
)
from vision_training import choose_device, seed_everything, train_frame_model


EVAL_MODES = ["all", "label_holdout", "session_holdout", "region_holdout"]
REGION_ROWS = ["top", "middle", "bottom"]
REGION_COLS = ["left", "center", "right"]


@dataclass
class FoldDefinition:
    holdout_group: str
    train_captures: list[VisionCapture]
    eval_captures: list[VisionCapture]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate strict evaluation diagnostics for the frame vision gaze model.")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "reports" / "vision_eval" / "latest",
    )
    parser.add_argument("--mode", choices=EVAL_MODES, default="all")
    parser.add_argument("--epochs", type=int, default=VISION_EVAL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=VISION_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=VISION_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=VISION_WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def region_bins(target_x: float, target_y: float) -> tuple[int, int]:
    x_bin = min(int(np.clip(target_x, 0.0, 0.999999) * 3.0), 2)
    y_bin = min(int(np.clip(target_y, 0.0, 0.999999) * 3.0), 2)
    return x_bin, y_bin


def region_label(target_x: float, target_y: float) -> str:
    x_bin, y_bin = region_bins(target_x, target_y)
    return f"{REGION_ROWS[y_bin]}-{REGION_COLS[x_bin]}"


def build_folds(mode: str, captures: list[VisionCapture]) -> list[FoldDefinition]:
    if mode == "label_holdout":
        train_captures = [capture for capture in captures if capture.split != "eval"]
        eval_captures = [capture for capture in captures if capture.split == "eval"]
        if not eval_captures:
            raise ValueError("No eval captures available for label_holdout mode")
        return [FoldDefinition("collector_eval", train_captures, eval_captures)]

    if mode == "session_holdout":
        folds = []
        session_ids = sorted({capture.session_id for capture in captures})
        for session_id in session_ids:
            eval_captures = [capture for capture in captures if capture.session_id == session_id]
            train_captures = [capture for capture in captures if capture.session_id != session_id]
            if eval_captures and train_captures:
                folds.append(FoldDefinition(session_id, train_captures, eval_captures))
        if not folds:
            raise ValueError("Need at least two sessions for session_holdout mode")
        return folds

    if mode == "region_holdout":
        folds = []
        labels = [f"{row}-{column}" for row in REGION_ROWS for column in REGION_COLS]
        for label in labels:
            eval_captures = [capture for capture in captures if region_label(capture.target_x, capture.target_y) == label]
            train_captures = [capture for capture in captures if region_label(capture.target_x, capture.target_y) != label]
            if eval_captures and train_captures:
                folds.append(FoldDefinition(label, train_captures, eval_captures))
        if not folds:
            raise ValueError("No populated regions available for region_holdout mode")
        return folds

    raise ValueError(f"Unsupported mode: {mode}")


def predict_sample(
    model,
    sample,
    head_mean: np.ndarray,
    head_scale: np.ndarray,
    device: torch.device,
) -> tuple[float, float]:
    left_eye = load_grayscale_eye_crop(sample.left_path)
    right_eye = load_grayscale_eye_crop(sample.right_path)
    head_features = (sample.head_features - head_mean) / head_scale

    left_tensor = torch.from_numpy(left_eye).unsqueeze(0).unsqueeze(0).to(device)
    right_tensor = torch.from_numpy(right_eye).unsqueeze(0).unsqueeze(0).to(device)
    head_tensor = torch.from_numpy(head_features.astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(left_tensor, right_tensor, head_tensor)
    output = prediction.squeeze(0).detach().cpu().numpy()
    return float(output[0]), float(output[1])


def evaluate_fold(
    fold: FoldDefinition,
    screen_size: tuple[int, int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> dict:
    train_samples = [sample for capture in fold.train_captures for sample in capture.samples]
    eval_samples = [sample for capture in fold.eval_captures for sample in capture.samples]
    head_mean, head_scale = compute_head_normalization(train_samples)

    result = train_frame_model(
        train_samples=train_samples,
        eval_samples=eval_samples,
        screen_size=screen_size,
        head_mean=head_mean,
        head_scale=head_scale,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        verbose=False,
    )

    frame_errors_x: list[float] = []
    frame_errors_y: list[float] = []
    frame_errors_distance: list[float] = []
    capture_errors_x: list[float] = []
    capture_errors_y: list[float] = []
    capture_errors_distance: list[float] = []

    width, height = screen_size
    for capture in fold.eval_captures:
        capture_pred_x: list[float] = []
        capture_pred_y: list[float] = []
        for sample in capture.samples:
            pred_x, pred_y = predict_sample(
                result.model,
                sample,
                head_mean=head_mean,
                head_scale=head_scale,
                device=result.device,
            )
            err_x = abs(pred_x - sample.target_x) * width
            err_y = abs(pred_y - sample.target_y) * height
            frame_errors_x.append(err_x)
            frame_errors_y.append(err_y)
            frame_errors_distance.append(math.sqrt(err_x**2 + err_y**2))
            capture_pred_x.append(pred_x)
            capture_pred_y.append(pred_y)

        mean_pred_x = float(np.mean(capture_pred_x))
        mean_pred_y = float(np.mean(capture_pred_y))
        capture_err_x = abs(mean_pred_x - capture.target_x) * width
        capture_err_y = abs(mean_pred_y - capture.target_y) * height
        capture_errors_x.append(capture_err_x)
        capture_errors_y.append(capture_err_y)
        capture_errors_distance.append(math.sqrt(capture_err_x**2 + capture_err_y**2))

    return {
        "holdout_group": fold.holdout_group,
        "train_capture_count": len(fold.train_captures),
        "eval_capture_count": len(fold.eval_captures),
        "train_sample_count": len(train_samples),
        "eval_sample_count": len(eval_samples),
        "frame_mae_x_px": float(np.mean(frame_errors_x)),
        "frame_mae_y_px": float(np.mean(frame_errors_y)),
        "frame_mae_distance_px": float(np.mean(frame_errors_distance)),
        "capture_mae_x_px": float(np.mean(capture_errors_x)),
        "capture_mae_y_px": float(np.mean(capture_errors_y)),
        "capture_mae_distance_px": float(np.mean(capture_errors_distance)),
        "train_metrics": result.train_metrics,
        "in_fold_eval_metrics": result.eval_metrics,
    }


def summarize_mode(
    mode: str,
    captures: list[VisionCapture],
    screen_size: tuple[int, int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> dict:
    folds = build_folds(mode, captures)
    fold_summaries = [
        evaluate_fold(
            fold,
            screen_size=screen_size,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
        )
        for fold in folds
    ]

    return {
        "mode": mode,
        "screen_width": screen_size[0],
        "screen_height": screen_size[1],
        "fold_count": len(fold_summaries),
        "overall": {
            "frame_mae_x_px": float(np.mean([fold["frame_mae_x_px"] for fold in fold_summaries])),
            "frame_mae_y_px": float(np.mean([fold["frame_mae_y_px"] for fold in fold_summaries])),
            "frame_mae_distance_px": float(np.mean([fold["frame_mae_distance_px"] for fold in fold_summaries])),
            "capture_mae_x_px": float(np.mean([fold["capture_mae_x_px"] for fold in fold_summaries])),
            "capture_mae_y_px": float(np.mean([fold["capture_mae_y_px"] for fold in fold_summaries])),
            "capture_mae_distance_px": float(np.mean([fold["capture_mae_distance_px"] for fold in fold_summaries])),
        },
        "folds": fold_summaries,
    }


def write_summary_text(summary: dict, output_path: Path) -> None:
    overall = summary["overall"]
    lines = [
        f"Mode: {summary['mode']}",
        f"Screen: {summary['screen_width']}x{summary['screen_height']}",
        f"Folds: {summary['fold_count']}",
        "",
        "Overall",
        f"- frame mae: {overall['frame_mae_x_px']:.1f}px x | {overall['frame_mae_y_px']:.1f}px y | {overall['frame_mae_distance_px']:.1f}px distance",
        f"- capture mae: {overall['capture_mae_x_px']:.1f}px x | {overall['capture_mae_y_px']:.1f}px y | {overall['capture_mae_distance_px']:.1f}px distance",
        "",
        "Folds",
    ]
    for fold in summary["folds"]:
        lines.append(
            f"- {fold['holdout_group']}: "
            f"frame {fold['frame_mae_distance_px']:.1f}px | "
            f"capture {fold['capture_mae_distance_px']:.1f}px "
            f"({fold['eval_capture_count']} eval captures)"
        )
    output_path.write_text("\n".join(lines) + "\n")


def run_mode(args, mode: str, captures: list[VisionCapture], screen_size: tuple[int, int], output_root: Path) -> dict:
    summary = summarize_mode(
        mode=mode,
        captures=captures,
        screen_size=screen_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=choose_device(args.device),
    )
    mode_dir = output_root / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    (mode_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_summary_text(summary, mode_dir / "summary.txt")
    return {
        "mode": mode,
        "overall": summary["overall"],
        "summary_path": str(mode_dir / "summary.txt"),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    captures, screen_size = collect_vision_captures(args.dataset_dir)
    if screen_size is None or not captures:
        raise ValueError("No compatible eye-crop sessions found")

    modes = [mode for mode in EVAL_MODES if mode != "all"] if args.mode == "all" else [args.mode]
    output_dir = args.output_dir
    if args.mode == "all":
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        mode_dir = output_dir / args.mode
        if mode_dir.exists():
            shutil.rmtree(mode_dir)

    index = {
        "dataset_dir": str(args.dataset_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "screen_width": screen_size[0],
        "screen_height": screen_size[1],
        "modes": [],
    }

    for mode in modes:
        entry = run_mode(args, mode, captures, screen_size, output_dir)
        index["modes"].append(entry)
        overall = entry["overall"]
        print(
            f"{mode}: frame {overall['frame_mae_distance_px']:.1f}px | "
            f"capture {overall['capture_mae_distance_px']:.1f}px"
        )

    (output_dir / "index.json").write_text(json.dumps(index, indent=2))
    lines = ["Vision Eval Report", ""]
    for entry in index["modes"]:
        overall = entry["overall"]
        lines.append(
            f"- {entry['mode']}: frame {overall['frame_mae_distance_px']:.1f}px | "
            f"capture {overall['capture_mae_distance_px']:.1f}px | {entry['summary_path']}"
        )
    (output_dir / "index.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
