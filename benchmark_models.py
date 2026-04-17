from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil

import cv2
import numpy as np
import torch

from constants import DATASET_DIR, ROOT_DIR
from eval_report import (
    execute_mode as execute_ridge_mode,
    iter_session_captures,
    latest_screen_size,
    make_summary as make_ridge_summary,
)
from vision_dataset import collect_vision_captures, compute_head_normalization, sample_payload_features
from vision_eval_report import build_folds as build_vision_folds
from vision_model import (
    EyeCropModelConfig,
    best_frame_vision_config,
    matched_attention_frame_vision_config,
)
from vision_training import choose_device, seed_everything, train_frame_model


ACTIVE_CANDIDATES = [
    "ridge",
    "frame_wide_aug_long",
    "frame_attention_matched_long",
]
BENCHMARK_PROFILES = ["standard", "full", "quick"]
CACHE_DIR = ROOT_DIR / ".cache" / "benchmark"


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    kind: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    augment_train: bool = False
    model_config: EyeCropModelConfig | None = None


CANDIDATES: dict[str, CandidateSpec] = {
    "ridge": CandidateSpec(
        name="ridge",
        kind="ridge",
        epochs=0,
        batch_size=0,
        learning_rate=0.0,
        weight_decay=0.0,
    ),
    "frame_wide_aug_long": CandidateSpec(
        name="frame_wide_aug_long",
        kind="frame",
        epochs=28,
        batch_size=64,
        learning_rate=8e-4,
        weight_decay=1e-4,
        augment_train=True,
        model_config=best_frame_vision_config(),
    ),
    "frame_attention_matched_long": CandidateSpec(
        name="frame_attention_matched_long",
        kind="frame",
        epochs=28,
        batch_size=64,
        learning_rate=8e-4,
        weight_decay=1e-4,
        augment_train=True,
        model_config=matched_attention_frame_vision_config(),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the active gaze models across comparable holdouts.")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument(
        "--mode",
        choices=["label_holdout", "session_holdout", "region_holdout"],
        required=True,
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=ACTIVE_CANDIDATES,
        choices=ACTIVE_CANDIDATES,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "reports" / "benchmark" / "latest",
    )
    parser.add_argument(
        "--profile",
        choices=BENCHMARK_PROFILES,
        default="standard",
        help="standard uses early stopping, full uses the full epoch budget, quick is a lighter tracking run.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore cached results and recompute benchmarks.",
    )
    return parser.parse_args()


def resolve_training_profile(spec: CandidateSpec, profile: str) -> dict[str, int | float | None]:
    if profile == "full":
        return {
            "epochs": spec.epochs,
            "early_stopping_patience": None,
            "early_stopping_min_epochs": 0,
            "early_stopping_min_delta": 0.0,
        }
    if profile == "quick":
        return {
            "epochs": min(spec.epochs, 12),
            "early_stopping_patience": 3,
            "early_stopping_min_epochs": 6,
            "early_stopping_min_delta": 5e-5,
        }
    return {
        "epochs": spec.epochs,
        "early_stopping_patience": 4,
        "early_stopping_min_epochs": 8,
        "early_stopping_min_delta": 5e-5,
    }


def _hash_file(path: Path, digest: hashlib._Hash) -> None:
    digest.update(str(path.relative_to(ROOT_DIR)).encode("utf-8"))
    digest.update(b"\0")
    digest.update(path.read_bytes())
    digest.update(b"\0")


def dataset_fingerprint(dataset_dir: Path) -> str:
    digest = hashlib.sha256()
    digest.update(str(dataset_dir.resolve()).encode("utf-8"))
    digest.update(b"\0")
    session_dirs = sorted(path for path in dataset_dir.glob("session-*") if path.is_dir())
    for session_dir in session_dirs:
        for filename in ("session.json", "captures.jsonl"):
            path = session_dir / filename
            if path.exists():
                _hash_file(path, digest)
    return digest.hexdigest()


def code_fingerprint() -> str:
    digest = hashlib.sha256()
    paths = [
        ROOT_DIR / "benchmark_models.py",
        ROOT_DIR / "vision_training.py",
        ROOT_DIR / "vision_model.py",
        ROOT_DIR / "vision_dataset.py",
        ROOT_DIR / "mapper.py",
        ROOT_DIR / "eval_report.py",
    ]
    for path in paths:
        _hash_file(path, digest)
    return digest.hexdigest()


def cache_key(
    spec: CandidateSpec,
    mode: str,
    profile: str,
    dataset_dir: Path,
) -> str:
    training_profile = resolve_training_profile(spec, profile) if spec.kind == "frame" else None
    payload = {
        "candidate": spec.name,
        "kind": spec.kind,
        "mode": mode,
        "profile": profile,
        "dataset_dir": str(dataset_dir.resolve()),
        "dataset_fingerprint": dataset_fingerprint(dataset_dir),
        "code_fingerprint": code_fingerprint(),
        "epochs": spec.epochs,
        "batch_size": spec.batch_size,
        "learning_rate": spec.learning_rate,
        "weight_decay": spec.weight_decay,
        "augment_train": spec.augment_train,
        "training_profile": training_profile,
        "model_config": spec.model_config.to_dict() if spec.model_config else None,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def cache_path_for_key(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def load_cached_result(key: str) -> dict | None:
    path = cache_path_for_key(key)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    payload["cache"] = {"hit": True, "key": key}
    return payload


def save_cached_result(key: str, result: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = dict(result)
    payload["cache"] = {"hit": False, "key": key}
    cache_path_for_key(key).write_text(json.dumps(payload, indent=2))


def predict_frame_sample(
    model,
    sample,
    head_mean,
    head_scale,
    device: torch.device,
    extra_mean=None,
    extra_scale=None,
) -> tuple[float, float]:
    left = cv2.imread(str(sample.left_path), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(sample.right_path), cv2.IMREAD_GRAYSCALE)
    if left is None or right is None:
        raise RuntimeError("Failed to read eye crop during benchmark")

    left = ((left.astype(np.float32) / 255.0) - 0.5) / 0.5
    right = ((right.astype(np.float32) / 255.0) - 0.5) / 0.5
    head = (sample.head_features - head_mean) / head_scale

    left_tensor = torch.from_numpy(left).unsqueeze(0).unsqueeze(0).to(device)
    right_tensor = torch.from_numpy(right).unsqueeze(0).unsqueeze(0).to(device)
    head_tensor = torch.from_numpy(head.astype(np.float32)).unsqueeze(0).to(device)
    extra_keys = tuple(model.config.extra_feature_keys)
    if extra_keys:
        if extra_mean is None or extra_scale is None:
            raise ValueError("extra_mean and extra_scale are required for extra-feature models")
        extra = (sample_payload_features(sample, extra_keys) - extra_mean) / extra_scale
        extra_tensor = torch.from_numpy(extra.astype(np.float32)).unsqueeze(0).to(device)
    else:
        extra_tensor = None

    with torch.no_grad():
        prediction = model(left_tensor, right_tensor, head_tensor, extra_tensor)
    pred = prediction.squeeze(0).detach().cpu().numpy()
    return float(pred[0]), float(pred[1])


def evaluate_frame_candidate(
    spec: CandidateSpec,
    dataset_dir: Path,
    mode: str,
    profile: str,
    seed: int,
    device: torch.device,
) -> dict:
    captures, screen_size = collect_vision_captures(dataset_dir)
    if screen_size is None or not captures:
        raise ValueError("No compatible eye-crop sessions found for frame benchmark")

    folds = build_vision_folds(mode, captures)
    width, height = screen_size
    fold_results = []
    training_profile = resolve_training_profile(spec, profile)
    for fold_index, fold in enumerate(folds):
        train_samples = [sample for capture in fold.train_captures for sample in capture.samples]
        eval_captures = fold.eval_captures
        eval_samples = [sample for capture in eval_captures for sample in capture.samples]

        head_mean, head_scale = compute_head_normalization(train_samples)
        seed_everything(seed + fold_index)
        result = train_frame_model(
            train_samples=train_samples,
            eval_samples=eval_samples,
            screen_size=screen_size,
            head_mean=head_mean,
            head_scale=head_scale,
            batch_size=spec.batch_size,
            epochs=int(training_profile["epochs"]),
            learning_rate=spec.learning_rate,
            weight_decay=spec.weight_decay,
            device=device,
            model_config=spec.model_config,
            augment_train=spec.augment_train,
            early_stopping_patience=training_profile["early_stopping_patience"],
            early_stopping_min_epochs=int(training_profile["early_stopping_min_epochs"]),
            early_stopping_min_delta=float(training_profile["early_stopping_min_delta"]),
            verbose=False,
        )

        frame_err_x = []
        frame_err_y = []
        frame_err_d = []
        capture_err_x = []
        capture_err_y = []
        capture_err_d = []
        for capture in eval_captures:
            preds = []
            for sample in capture.samples:
                pred_x, pred_y = predict_frame_sample(
                    result.model,
                    sample,
                    head_mean=head_mean,
                    head_scale=head_scale,
                    device=result.device,
                    extra_mean=result.extra_mean,
                    extra_scale=result.extra_scale,
                )
                err_x = abs(pred_x - sample.target_x) * width
                err_y = abs(pred_y - sample.target_y) * height
                frame_err_x.append(err_x)
                frame_err_y.append(err_y)
                frame_err_d.append(math.sqrt(err_x**2 + err_y**2))
                preds.append((pred_x, pred_y))

            mean_x = float(np.mean([item[0] for item in preds]))
            mean_y = float(np.mean([item[1] for item in preds]))
            capture_err_x.append(abs(mean_x - capture.target_x) * width)
            capture_err_y.append(abs(mean_y - capture.target_y) * height)
            capture_err_d.append(math.sqrt(capture_err_x[-1] ** 2 + capture_err_y[-1] ** 2))

        fold_results.append(
            {
                "holdout_group": fold.holdout_group,
                "frame_mae_x_px": float(np.mean(frame_err_x)),
                "frame_mae_y_px": float(np.mean(frame_err_y)),
                "frame_mae_distance_px": float(np.mean(frame_err_d)),
                "capture_mae_x_px": float(np.mean(capture_err_x)),
                "capture_mae_y_px": float(np.mean(capture_err_y)),
                "capture_mae_distance_px": float(np.mean(capture_err_d)),
                "train_samples": len(train_samples),
                "eval_samples": len(eval_samples),
                "train_metrics": result.train_metrics,
                "eval_metrics": result.eval_metrics,
                "best_epoch": result.best_epoch,
                "epochs_trained": result.epochs_trained,
            }
        )

    return {
        "candidate": spec.name,
        "kind": spec.kind,
        "mode": mode,
        "profile": profile,
        "screen_width": screen_size[0],
        "screen_height": screen_size[1],
        "fold_count": len(fold_results),
        "overall": {
            "frame_mae_x_px": float(np.mean([fold["frame_mae_x_px"] for fold in fold_results])),
            "frame_mae_y_px": float(np.mean([fold["frame_mae_y_px"] for fold in fold_results])),
            "frame_mae_distance_px": float(np.mean([fold["frame_mae_distance_px"] for fold in fold_results])),
            "capture_mae_x_px": float(np.mean([fold["capture_mae_x_px"] for fold in fold_results])),
            "capture_mae_y_px": float(np.mean([fold["capture_mae_y_px"] for fold in fold_results])),
            "capture_mae_distance_px": float(np.mean([fold["capture_mae_distance_px"] for fold in fold_results])),
        },
        "folds": fold_results,
        "config": {
            "epochs": spec.epochs,
            "batch_size": spec.batch_size,
            "learning_rate": spec.learning_rate,
            "weight_decay": spec.weight_decay,
            "augment_train": spec.augment_train,
            "training_profile": training_profile,
            "model_config": spec.model_config.to_dict() if spec.model_config else None,
        },
    }


def evaluate_ridge_candidate(dataset_dir: Path, mode: str) -> dict:
    screen_size = latest_screen_size(dataset_dir)
    captures = iter_session_captures(dataset_dir, screen_size)
    result = execute_ridge_mode(mode, captures, screen_size)
    summary = make_ridge_summary(result)
    grouped_predictions: dict[tuple[str, int], list[tuple[float, float, float, float]]] = {}
    for sample in result.sample_evals:
        key = (sample.session_id, sample.capture_index)
        grouped_predictions.setdefault(key, []).append(
            (sample.pred_x, sample.pred_y, sample.target_x, sample.target_y)
        )

    width, height = screen_size
    capture_mae_x_values = []
    capture_mae_y_values = []
    capture_mae_d_values = []
    for predictions in grouped_predictions.values():
        mean_pred_x = float(np.mean([item[0] for item in predictions]))
        mean_pred_y = float(np.mean([item[1] for item in predictions]))
        target_x = float(predictions[0][2])
        target_y = float(predictions[0][3])
        err_x = abs(mean_pred_x - target_x) * width
        err_y = abs(mean_pred_y - target_y) * height
        capture_mae_x_values.append(err_x)
        capture_mae_y_values.append(err_y)
        capture_mae_d_values.append(math.sqrt(err_x**2 + err_y**2))

    return {
        "candidate": "ridge",
        "kind": "ridge",
        "mode": mode,
        "profile": "n/a",
        "screen_width": screen_size[0],
        "screen_height": screen_size[1],
        "fold_count": len(summary["holdout_groups"]),
        "overall": {
            "frame_mae_x_px": summary["overall"]["mae_x_px"],
            "frame_mae_y_px": summary["overall"]["mae_y_px"],
            "frame_mae_distance_px": summary["overall"]["mae_distance_px"],
            "capture_mae_x_px": float(np.mean(capture_mae_x_values)),
            "capture_mae_y_px": float(np.mean(capture_mae_y_values)),
            "capture_mae_distance_px": float(np.mean(capture_mae_d_values)),
        },
        "folds": summary["holdout_groups"],
        "config": {"type": "polynomial_ridge"},
    }


def write_summary(results: list[dict], output_dir: Path) -> None:
    ranked = sorted(results, key=lambda item: item["overall"]["capture_mae_distance_px"])
    summary = {
        "ranked_candidates": [
            {
                "candidate": item["candidate"],
                "kind": item["kind"],
                "capture_mae_distance_px": item["overall"]["capture_mae_distance_px"],
                "capture_mae_x_px": item["overall"]["capture_mae_x_px"],
                "capture_mae_y_px": item["overall"]["capture_mae_y_px"],
            }
            for item in ranked
        ]
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = ["Model Benchmark", ""]
    for index, item in enumerate(ranked, start=1):
        overall = item["overall"]
        lines.append(
            f"{index}. {item['candidate']}: "
            f"{overall['capture_mae_distance_px']:.1f}px distance | "
            f"{overall['capture_mae_x_px']:.1f}px x | "
            f"{overall['capture_mae_y_px']:.1f}px y"
        )
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir / args.mode
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    results = []
    for name in args.candidates:
        spec = CANDIDATES[name]
        key = cache_key(spec, args.mode, args.profile, args.dataset_dir)
        result = None if args.refresh else load_cached_result(key)
        if result is not None:
            print(f"benchmarking {name} on {args.mode} [{args.profile}] (cached)")
        else:
            print(f"benchmarking {name} on {args.mode} [{args.profile}]")
            if spec.kind == "ridge":
                result = evaluate_ridge_candidate(args.dataset_dir, args.mode)
            else:
                result = evaluate_frame_candidate(
                    spec,
                    dataset_dir=args.dataset_dir,
                    mode=args.mode,
                    profile=args.profile,
                    seed=args.seed,
                    device=device,
                )
            save_cached_result(key, result)

        if spec.kind == "ridge" and result.get("cache", {}).get("key") is None:
            result["cache"] = {"hit": False, "key": key}
        elif result.get("cache", {}).get("key") is None:
            result["cache"] = {"hit": False, "key": key}
        results.append(result)
        (output_dir / f"{name}.json").write_text(json.dumps(result, indent=2))
        overall = result["overall"]
        print(
            f"{name}: {overall['capture_mae_distance_px']:.1f}px distance | "
            f"{overall['capture_mae_x_px']:.1f}px x | "
            f"{overall['capture_mae_y_px']:.1f}px y"
        )

    write_summary(results, output_dir)


if __name__ == "__main__":
    main()
