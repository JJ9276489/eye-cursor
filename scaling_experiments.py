from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch

from constants import (
    DATASET_DIR,
    ROOT_DIR,
    VISION_BATCH_SIZE,
    VISION_EPOCHS,
    VISION_LEARNING_RATE,
    VISION_WEIGHT_DECAY,
)
from vision_dataset import (
    VisionCapture,
    VisionSample,
    collect_vision_captures,
    compute_head_normalization,
    sample_payload_features,
)
from vision_eval_report import build_folds
from vision_model import (
    EyeCropModelConfig,
    EyeCropRegressor,
    best_frame_vision_config,
    clifford_frame_vision_config,
    matched_attention_frame_vision_config,
    spatial_frame_vision_config,
    spatial_geometry_frame_vision_config,
    tiny_patch_transformer_frame_vision_config,
)
from vision_training import choose_device, seed_everything, train_frame_model


MODEL_CHOICES = ["concat", "spatial", "spatial_geom", "clifford", "vit", "attn"]
DEFAULT_MODELS = ["concat", "spatial", "spatial_geom", "vit", "attn"]
DEFAULT_PARAM_MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5]
DEFAULT_DATA_FRACTIONS = [0.15, 0.25, 0.4, 0.6, 0.8, 1.0]
DEFAULT_EPOCH_BUDGETS = [4, 8, 12, 20, 28, 40, 56]
DEFAULT_SWEEPS = ["parameters", "data", "epochs"]
CACHE_DIR = ROOT_DIR / ".cache" / "scaling"
PLOT_STYLE = {
    "attn": {"label": "Attention", "color": "#3b82f6"},
    "concat": {"label": "Concat", "color": "#16a34a"},
    "clifford": {"label": "Clifford-Inspired", "color": "#0891b2"},
    "spatial": {"label": "Spatial CNN", "color": "#ea580c"},
    "spatial_geom": {"label": "Spatial CNN + Geometry", "color": "#dc2626"},
    "vit": {"label": "Tiny Patch Transformer", "color": "#7c3aed"},
}


@dataclass(frozen=True)
class PlotTransform:
    key: str
    title: str
    log_x: bool
    log_y: bool


@dataclass(frozen=True)
class SweepSpec:
    name: str
    model_key: str
    model_config: EyeCropModelConfig
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    augment_train: bool
    metadata: dict


PLOT_TRANSFORMS = [
    PlotTransform("linear", "Linear", log_x=False, log_y=False),
    PlotTransform("semilogx", "Semilog X", log_x=True, log_y=False),
    PlotTransform("semilogy", "Semilog Y", log_x=False, log_y=True),
    PlotTransform("loglog", "Log Log", log_x=True, log_y=True),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one-variable-at-a-time scaling sweeps for the frame vision gaze models."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "reports" / "scaling" / "latest",
    )
    parser.add_argument(
        "--mode",
        choices=["label_holdout", "session_holdout", "region_holdout"],
        default="label_holdout",
        help="Evaluation holdout mode. label_holdout is the practical default for dense sweeps.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_CHOICES,
        default=DEFAULT_MODELS,
        help="Model families to sweep.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch-size", type=int, default=VISION_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=VISION_EPOCHS)
    parser.add_argument("--lr", type=float, default=VISION_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=VISION_WEIGHT_DECAY)
    parser.add_argument(
        "--sweeps",
        nargs="+",
        choices=DEFAULT_SWEEPS,
        default=DEFAULT_SWEEPS,
        help="Which scaling sweeps to run.",
    )
    parser.add_argument(
        "--param-multipliers",
        type=float,
        nargs="+",
        default=DEFAULT_PARAM_MULTIPLIERS,
        help="Width multipliers for the parameter-count sweep.",
    )
    parser.add_argument(
        "--data-fractions",
        type=float,
        nargs="+",
        default=DEFAULT_DATA_FRACTIONS,
        help="Train-capture fractions for the data sweep.",
    )
    parser.add_argument(
        "--epoch-budgets",
        type=int,
        nargs="+",
        default=DEFAULT_EPOCH_BUDGETS,
        help="Epoch budgets for the epoch sweep.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore cached point results and recompute them.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Stop a fold after this many epochs without eval-loss improvement.",
    )
    parser.add_argument(
        "--early-stopping-min-epochs",
        type=int,
        default=0,
        help="Minimum epochs to train before early stopping can trigger.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum eval-loss improvement needed to reset early stopping.",
    )
    return parser.parse_args()


def dataset_fingerprint(dataset_dir: Path) -> str:
    digest = hashlib.sha256()
    digest.update(str(dataset_dir.resolve()).encode("utf-8"))
    digest.update(b"\0")
    for session_dir in sorted(path for path in dataset_dir.glob("session-*") if path.is_dir()):
        for filename in ("session.json", "captures.jsonl"):
            path = session_dir / filename
            if path.exists():
                digest.update(str(path.relative_to(ROOT_DIR)).encode("utf-8"))
                digest.update(b"\0")
                digest.update(path.read_bytes())
                digest.update(b"\0")
    return digest.hexdigest()


def code_fingerprint() -> str:
    digest = hashlib.sha256()
    paths = [
        ROOT_DIR / "scaling_experiments.py",
        ROOT_DIR / "constants.py",
        ROOT_DIR / "vision_model.py",
        ROOT_DIR / "vision_training.py",
        ROOT_DIR / "vision_dataset.py",
        ROOT_DIR / "vision_eval_report.py",
    ]
    for path in paths:
        digest.update(str(path.relative_to(ROOT_DIR)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def model_label(model_key: str) -> str:
    return PLOT_STYLE[model_key]["label"]


def base_config_for_model(model_key: str) -> EyeCropModelConfig:
    if model_key == "attn":
        return matched_attention_frame_vision_config()
    if model_key == "spatial":
        return spatial_frame_vision_config()
    if model_key == "spatial_geom":
        return spatial_geometry_frame_vision_config()
    if model_key == "clifford":
        return clifford_frame_vision_config()
    if model_key == "vit":
        return tiny_patch_transformer_frame_vision_config()
    return best_frame_vision_config()


def scaled_dim(value: int, multiplier: float, minimum: int = 8, multiple: int = 8) -> int:
    scaled = max(minimum, int(round(value * multiplier)))
    return max(multiple, int(round(scaled / multiple) * multiple))


def scale_config(base: EyeCropModelConfig, multiplier: float) -> EyeCropModelConfig:
    token_dim = scaled_dim(base.token_dim, multiplier, minimum=16, multiple=4)
    patch_heads = min(base.patch_heads, token_dim)
    while token_dim % patch_heads != 0 and patch_heads > 1:
        patch_heads -= 1
    return EyeCropModelConfig(
        encoder_channels=tuple(scaled_dim(value, multiplier, minimum=8, multiple=8) for value in base.encoder_channels),
        encoder_type=base.encoder_type,
        encoder_pooling=base.encoder_pooling,
        eye_coord_channels=base.eye_coord_channels,
        head_hidden_dims=tuple(scaled_dim(value, multiplier, minimum=16, multiple=8) for value in base.head_hidden_dims),
        extra_feature_keys=base.extra_feature_keys,
        extra_hidden_dims=tuple(
            scaled_dim(value, multiplier, minimum=16, multiple=8) for value in base.extra_hidden_dims
        ),
        regressor_hidden_dims=tuple(
            scaled_dim(value, multiplier, minimum=16, multiple=8) for value in base.regressor_hidden_dims
        ),
        dropout=base.dropout,
        fusion_mode=base.fusion_mode,
        token_dim=token_dim,
        attention_heads=base.attention_heads,
        attention_layers=base.attention_layers,
        attention_dropout=base.attention_dropout,
        patch_size=base.patch_size,
        patch_layers=base.patch_layers,
        patch_heads=patch_heads,
        patch_dropout=base.patch_dropout,
        clifford_blades=base.clifford_blades,
        clifford_kernel_size=base.clifford_kernel_size,
    )


def model_param_count(config: EyeCropModelConfig) -> int:
    model = EyeCropRegressor(
        extra_feature_dim=len(config.extra_feature_keys),
        config=config,
    )
    return sum(parameter.numel() for parameter in model.parameters())


def load_eye(path: Path) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read eye crop: {path}")
    image = image.astype(np.float32) / 255.0
    return ((image - 0.5) / 0.5).astype(np.float32)


def predict_sample(
    model: EyeCropRegressor,
    sample: VisionSample,
    head_mean: np.ndarray,
    head_scale: np.ndarray,
    device: torch.device,
    extra_mean: np.ndarray | None = None,
    extra_scale: np.ndarray | None = None,
) -> tuple[float, float]:
    left_eye = load_eye(sample.left_path)
    right_eye = load_eye(sample.right_path)
    head = (sample.head_features - head_mean) / head_scale

    left_tensor = torch.from_numpy(left_eye).unsqueeze(0).unsqueeze(0).to(device)
    right_tensor = torch.from_numpy(right_eye).unsqueeze(0).unsqueeze(0).to(device)
    head_tensor = torch.from_numpy(head.astype(np.float32)).unsqueeze(0).to(device)
    extra_keys = tuple(model.config.extra_feature_keys)
    if extra_keys:
        if extra_mean is None or extra_scale is None:
            raise ValueError("extra_mean and extra_scale are required for extra-feature models")
        extra_features = (sample_payload_features(sample, extra_keys) - extra_mean) / extra_scale
        extra_tensor = torch.from_numpy(extra_features.astype(np.float32)).unsqueeze(0).to(device)
    else:
        extra_tensor = None
    with torch.no_grad():
        prediction = model(left_tensor, right_tensor, head_tensor, extra_tensor)
    output = prediction.squeeze(0).detach().cpu().numpy()
    return float(output[0]), float(output[1])


def stratified_capture_subset(
    captures: list[VisionCapture],
    fraction: float,
    seed: int,
) -> list[VisionCapture]:
    if fraction >= 1.0:
        return list(captures)
    if fraction <= 0.0:
        raise ValueError("fraction must be > 0")

    groups: dict[str, list[VisionCapture]] = defaultdict(list)
    for capture in captures:
        groups[capture.session_id].append(capture)

    rng = np.random.default_rng(seed)
    target_total = max(1, int(round(len(captures) * fraction)))
    allocations: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    allocated = 0
    for session_id, items in groups.items():
        ideal = len(items) * fraction
        count = min(len(items), int(math.floor(ideal)))
        if count == 0 and target_total >= len(groups):
            count = 1
        allocations[session_id] = count
        allocated += count
        remainders.append((ideal - math.floor(ideal), session_id))

    while allocated < target_total:
        progressed = False
        for _, session_id in sorted(remainders, reverse=True):
            if allocations[session_id] < len(groups[session_id]):
                allocations[session_id] += 1
                allocated += 1
                progressed = True
                if allocated >= target_total:
                    break
        if not progressed:
            break

    selected: list[VisionCapture] = []
    for session_id, items in groups.items():
        indices = rng.permutation(len(items))
        count = min(len(items), allocations[session_id])
        chosen = [items[index] for index in indices[:count]]
        selected.extend(chosen)

    selected.sort(key=lambda capture: (capture.session_id, capture.capture_index))
    return selected[:target_total] if len(selected) > target_total else selected


def capture_metric_summary(
    model: EyeCropRegressor,
    eval_captures: list[VisionCapture],
    head_mean: np.ndarray,
    head_scale: np.ndarray,
    screen_size: tuple[int, int],
    device: torch.device,
    extra_mean: np.ndarray | None = None,
    extra_scale: np.ndarray | None = None,
) -> dict[str, float]:
    width, height = screen_size
    frame_err_x: list[float] = []
    frame_err_y: list[float] = []
    frame_err_d: list[float] = []
    capture_err_x: list[float] = []
    capture_err_y: list[float] = []
    capture_err_d: list[float] = []

    for capture in eval_captures:
        preds: list[tuple[float, float]] = []
        for sample in capture.samples:
            pred_x, pred_y = predict_sample(
                model=model,
                sample=sample,
                head_mean=head_mean,
                head_scale=head_scale,
                device=device,
                extra_mean=extra_mean,
                extra_scale=extra_scale,
            )
            err_x = abs(pred_x - sample.target_x) * width
            err_y = abs(pred_y - sample.target_y) * height
            frame_err_x.append(err_x)
            frame_err_y.append(err_y)
            frame_err_d.append(math.sqrt(err_x**2 + err_y**2))
            preds.append((pred_x, pred_y))

        mean_x = float(np.mean([item[0] for item in preds]))
        mean_y = float(np.mean([item[1] for item in preds]))
        err_x = abs(mean_x - capture.target_x) * width
        err_y = abs(mean_y - capture.target_y) * height
        capture_err_x.append(err_x)
        capture_err_y.append(err_y)
        capture_err_d.append(math.sqrt(err_x**2 + err_y**2))

    return {
        "frame_mae_x_px": float(np.mean(frame_err_x)),
        "frame_mae_y_px": float(np.mean(frame_err_y)),
        "frame_mae_distance_px": float(np.mean(frame_err_d)),
        "capture_mae_x_px": float(np.mean(capture_err_x)),
        "capture_mae_y_px": float(np.mean(capture_err_y)),
        "capture_mae_distance_px": float(np.mean(capture_err_d)),
    }


def point_cache_key(
    args: argparse.Namespace,
    sweep_name: str,
    spec: SweepSpec,
    fold_group: str,
    train_capture_ids: list[str],
) -> str:
    payload = {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "dataset_fingerprint": dataset_fingerprint(args.dataset_dir),
        "code_fingerprint": code_fingerprint(),
        "mode": args.mode,
        "sweep": sweep_name,
        "fold_group": fold_group,
        "spec": {
            "name": spec.name,
            "model_key": spec.model_key,
            "model_config": spec.model_config.to_dict(),
            "batch_size": spec.batch_size,
            "epochs": spec.epochs,
            "learning_rate": spec.learning_rate,
            "weight_decay": spec.weight_decay,
            "augment_train": spec.augment_train,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_epochs": args.early_stopping_min_epochs,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "metadata": spec.metadata,
        },
        "train_capture_ids": train_capture_ids,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def cache_path_for_key(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def load_cached_point(key: str) -> dict | None:
    path = cache_path_for_key(key)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_cached_point(key: str, result: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path_for_key(key).write_text(json.dumps(result, indent=2))


def evaluate_spec_on_fold(
    args: argparse.Namespace,
    spec: SweepSpec,
    train_captures: list[VisionCapture],
    eval_captures: list[VisionCapture],
    screen_size: tuple[int, int],
    seed: int,
    device: torch.device,
) -> dict:
    train_samples = [sample for capture in train_captures for sample in capture.samples]
    eval_samples = [sample for capture in eval_captures for sample in capture.samples]
    head_mean, head_scale = compute_head_normalization(train_samples)

    result = train_frame_model(
        train_samples=train_samples,
        eval_samples=eval_samples,
        screen_size=screen_size,
        head_mean=head_mean,
        head_scale=head_scale,
        batch_size=spec.batch_size,
        epochs=spec.epochs,
        learning_rate=spec.learning_rate,
        weight_decay=spec.weight_decay,
        device=device,
        model_config=spec.model_config,
        augment_train=spec.augment_train,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_epochs=args.early_stopping_min_epochs,
        early_stopping_min_delta=args.early_stopping_min_delta,
        verbose=False,
    )
    metrics = capture_metric_summary(
        model=result.model,
        eval_captures=eval_captures,
        head_mean=head_mean,
        head_scale=head_scale,
        screen_size=screen_size,
        device=result.device,
        extra_mean=result.extra_mean,
        extra_scale=result.extra_scale,
    )
    metrics["best_epoch"] = result.best_epoch
    metrics["epochs_trained"] = result.epochs_trained
    metrics["train_sample_count"] = len(train_samples)
    metrics["eval_sample_count"] = len(eval_samples)
    metrics["train_capture_count"] = len(train_captures)
    metrics["eval_capture_count"] = len(eval_captures)
    metrics["seed"] = seed
    return metrics


def evaluate_sweep_point(
    args: argparse.Namespace,
    sweep_name: str,
    spec: SweepSpec,
    captures: list[VisionCapture],
    screen_size: tuple[int, int],
    device: torch.device,
) -> dict:
    folds = build_folds(args.mode, captures)
    fold_results = []
    total_folds = len(folds)
    for fold_index, fold in enumerate(folds):
        train_captures = fold.train_captures
        if sweep_name == "data":
            train_captures = stratified_capture_subset(
                fold.train_captures,
                fraction=float(spec.metadata["data_fraction"]),
                seed=args.seed + fold_index,
            )
        train_capture_ids = [f"{capture.session_id}:{capture.capture_index}" for capture in train_captures]
        cache_key = point_cache_key(
            args=args,
            sweep_name=sweep_name,
            spec=spec,
            fold_group=fold.holdout_group,
            train_capture_ids=train_capture_ids,
        )
        cached = None if args.refresh else load_cached_point(cache_key)
        if cached is not None:
            print(
                f"    fold {fold_index + 1}/{total_folds} {fold.holdout_group}: cached",
                flush=True,
            )
            fold_results.append(cached)
            continue

        print(
            f"    fold {fold_index + 1}/{total_folds} {fold.holdout_group}: "
            f"train {len(train_captures)} caps / eval {len(fold.eval_captures)} caps",
            flush=True,
        )
        seed_everything(args.seed + fold_index)
        result = evaluate_spec_on_fold(
            args=args,
            spec=spec,
            train_captures=train_captures,
            eval_captures=fold.eval_captures,
            screen_size=screen_size,
            seed=args.seed + fold_index,
            device=device,
        )
        result["holdout_group"] = fold.holdout_group
        save_cached_point(cache_key, result)
        print(
            f"    fold {fold_index + 1}/{total_folds} {fold.holdout_group}: "
            f"{result['capture_mae_distance_px']:.1f}px capture MAE "
            f"(best epoch {result['best_epoch']})",
            flush=True,
        )
        fold_results.append(result)

    return {
        "sweep": sweep_name,
        "model": spec.model_key,
        "spec_name": spec.name,
        "metadata": spec.metadata,
        "model_config": spec.model_config.to_dict(),
        "parameter_count": model_param_count(spec.model_config),
        "overall": {
            "frame_mae_x_px": float(np.mean([fold["frame_mae_x_px"] for fold in fold_results])),
            "frame_mae_y_px": float(np.mean([fold["frame_mae_y_px"] for fold in fold_results])),
            "frame_mae_distance_px": float(np.mean([fold["frame_mae_distance_px"] for fold in fold_results])),
            "capture_mae_x_px": float(np.mean([fold["capture_mae_x_px"] for fold in fold_results])),
            "capture_mae_y_px": float(np.mean([fold["capture_mae_y_px"] for fold in fold_results])),
            "capture_mae_distance_px": float(np.mean([fold["capture_mae_distance_px"] for fold in fold_results])),
            "train_sample_count": float(np.mean([fold["train_sample_count"] for fold in fold_results])),
            "eval_sample_count": float(np.mean([fold["eval_sample_count"] for fold in fold_results])),
            "train_capture_count": float(np.mean([fold["train_capture_count"] for fold in fold_results])),
            "eval_capture_count": float(np.mean([fold["eval_capture_count"] for fold in fold_results])),
        },
        "folds": fold_results,
    }


def build_specs_for_model(args: argparse.Namespace, model_key: str) -> dict[str, list[SweepSpec]]:
    base = base_config_for_model(model_key)
    param_specs = []
    for multiplier in args.param_multipliers:
        config = scale_config(base, float(multiplier))
        param_specs.append(
            SweepSpec(
                name=f"{model_key}_param_{multiplier:.2f}",
                model_key=model_key,
                model_config=config,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                augment_train=True,
                metadata={"param_multiplier": float(multiplier)},
            )
        )

    data_specs = []
    for fraction in args.data_fractions:
        data_specs.append(
            SweepSpec(
                name=f"{model_key}_data_{fraction:.2f}",
                model_key=model_key,
                model_config=base,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                augment_train=True,
                metadata={"data_fraction": float(fraction)},
            )
        )

    epoch_specs = []
    for budget in args.epoch_budgets:
        epoch_specs.append(
            SweepSpec(
                name=f"{model_key}_epoch_{budget}",
                model_key=model_key,
                model_config=base,
                batch_size=args.batch_size,
                epochs=int(budget),
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                augment_train=True,
                metadata={"epoch_budget": int(budget)},
            )
        )

    return {"parameters": param_specs, "data": data_specs, "epochs": epoch_specs}


def write_summary_text(results: dict, output_path: Path) -> None:
    lines = [
        f"Mode: {results['mode']}",
        f"Screen: {results['screen_width']}x{results['screen_height']}",
        f"Models: {', '.join(results['models'])}",
        "",
    ]
    for sweep_name in results["sweep_order"]:
        lines.append(sweep_name.capitalize())
        for model_key in results["models"]:
            series = results["sweeps"].get(sweep_name, {}).get(model_key, [])
            if not series:
                continue
            best = min(series, key=lambda item: item["overall"]["capture_mae_distance_px"])
            lines.append(
                f"- {model_label(model_key)} best: "
                f"{best['overall']['capture_mae_distance_px']:.1f}px capture MAE | "
                f"spec={best['spec_name']} | metadata={best['metadata']}"
            )
        lines.append("")
    output_path.write_text("\n".join(lines).rstrip() + "\n")


def write_points_csv(results: dict, output_path: Path) -> None:
    columns = [
        "mode",
        "sweep",
        "model",
        "spec_name",
        "parameter_count",
        "param_multiplier",
        "data_fraction",
        "epoch_budget",
        "train_sample_count",
        "eval_sample_count",
        "capture_mae_distance_px",
        "capture_mae_x_px",
        "capture_mae_y_px",
        "frame_mae_distance_px",
        "frame_mae_x_px",
        "frame_mae_y_px",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for sweep_name in results["sweep_order"]:
            for model_key in results["models"]:
                for item in results["sweeps"].get(sweep_name, {}).get(model_key, []):
                    metadata = item["metadata"]
                    overall = item["overall"]
                    writer.writerow(
                        {
                            "mode": results["mode"],
                            "sweep": sweep_name,
                            "model": model_key,
                            "spec_name": item["spec_name"],
                            "parameter_count": item["parameter_count"],
                            "param_multiplier": metadata.get("param_multiplier"),
                            "data_fraction": metadata.get("data_fraction"),
                            "epoch_budget": metadata.get("epoch_budget"),
                            "train_sample_count": overall["train_sample_count"],
                            "eval_sample_count": overall["eval_sample_count"],
                            "capture_mae_distance_px": overall["capture_mae_distance_px"],
                            "capture_mae_x_px": overall["capture_mae_x_px"],
                            "capture_mae_y_px": overall["capture_mae_y_px"],
                            "frame_mae_distance_px": overall["frame_mae_distance_px"],
                            "frame_mae_x_px": overall["frame_mae_x_px"],
                            "frame_mae_y_px": overall["frame_mae_y_px"],
                        }
                    )


def sweep_xy(item: dict, sweep_name: str) -> tuple[float, float, str]:
    if sweep_name == "parameters":
        return (
            float(item["parameter_count"]),
            float(item["overall"]["capture_mae_distance_px"]),
            "Parameter Count",
        )
    if sweep_name == "data":
        return (
            float(item["overall"]["train_sample_count"]),
            float(item["overall"]["capture_mae_distance_px"]),
            "Train Samples",
        )
    return (
        float(item["metadata"]["epoch_budget"]),
        float(item["overall"]["capture_mae_distance_px"]),
        "Epoch Budget",
    )


def series_xy(series: list[dict], sweep_name: str) -> tuple[list[float], list[float], str]:
    points = [sweep_xy(item, sweep_name) for item in series]
    points.sort(key=lambda item: item[0])
    x_values = [item[0] for item in points]
    y_values = [item[1] for item in points]
    x_label = points[0][2] if points else ""
    return x_values, y_values, x_label


def transformed_for_fit(
    x_values: list[float],
    y_values: list[float],
    transform: PlotTransform,
) -> tuple[np.ndarray, np.ndarray]:
    x_array = np.asarray(x_values, dtype=np.float64)
    y_array = np.asarray(y_values, dtype=np.float64)
    mask = np.isfinite(x_array) & np.isfinite(y_array)
    if transform.log_x:
        mask &= x_array > 0
    if transform.log_y:
        mask &= y_array > 0
    x_array = x_array[mask]
    y_array = y_array[mask]
    if transform.log_x:
        x_array = np.log10(x_array)
    if transform.log_y:
        y_array = np.log10(y_array)
    return x_array, y_array


def fit_r2(x_values: np.ndarray, y_values: np.ndarray) -> float | None:
    if len(x_values) < 3:
        return None
    if float(np.ptp(x_values)) == 0.0 or float(np.ptp(y_values)) == 0.0:
        return None
    slope, intercept = np.polyfit(x_values, y_values, deg=1)
    predicted = slope * x_values + intercept
    ss_res = float(np.sum((y_values - predicted) ** 2))
    ss_tot = float(np.sum((y_values - np.mean(y_values)) ** 2))
    if ss_tot <= 0.0:
        return None
    return 1.0 - ss_res / ss_tot


def plot_sweep(
    results: dict,
    sweep_name: str,
    output_path: Path,
    transform: PlotTransform = PLOT_TRANSFORMS[0],
) -> None:
    plt.figure(figsize=(8, 5))
    for model_key in results["models"]:
        series = results["sweeps"].get(sweep_name, {}).get(model_key, [])
        if not series:
            continue
        x_values, y_values, x_label = series_xy(series, sweep_name)
        style = PLOT_STYLE[model_key]
        plt.plot(x_values, y_values, marker="o", label=style["label"], color=style["color"])

    plt.title(f"{sweep_name.capitalize()} Sweep ({results['mode']}, {transform.title})")
    plt.xlabel(x_label)
    plt.ylabel("Capture MAE Distance (px)")
    if transform.log_x:
        plt.xscale("log")
    if transform.log_y:
        plt.yscale("log")
    plt.grid(alpha=0.25, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_combined(
    results: dict,
    output_path: Path,
    transform: PlotTransform = PLOT_TRANSFORMS[0],
) -> None:
    sweep_order = results["sweep_order"]
    fig, axes = plt.subplots(1, len(sweep_order), figsize=(5 * len(sweep_order), 4.5), squeeze=False)
    for axis, sweep_name in zip(axes[0], sweep_order, strict=True):
        for model_key in results["models"]:
            series = results["sweeps"].get(sweep_name, {}).get(model_key, [])
            if not series:
                continue
            x_values, y_values, x_label = series_xy(series, sweep_name)
            style = PLOT_STYLE[model_key]
            axis.plot(x_values, y_values, marker="o", label=style["label"], color=style["color"])
            axis.set_xlabel(x_label)
            if transform.log_x:
                axis.set_xscale("log")
            if transform.log_y:
                axis.set_yscale("log")
        axis.set_title(sweep_name.capitalize())
        axis.set_ylabel("Capture MAE (px)")
        axis.grid(alpha=0.25, which="both")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(results["models"]))
    fig.suptitle(f"Scaling Experiments ({results['mode']}, {transform.title})")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_transform_fit_summary(results: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    lines = ["Scaling Transform Fit Summary", ""]

    for sweep_name in results["sweep_order"]:
        for model_key in results["models"]:
            series = results["sweeps"].get(sweep_name, {}).get(model_key, [])
            if len(series) < 2:
                continue
            x_values, y_values, _ = series_xy(series, sweep_name)
            scored_rows = []
            for transform in PLOT_TRANSFORMS:
                fit_x, fit_y = transformed_for_fit(x_values, y_values, transform)
                r2 = fit_r2(fit_x, fit_y)
                row = {
                    "sweep": sweep_name,
                    "model": model_key,
                    "transform": transform.key,
                    "points": len(fit_x),
                    "r2": "" if r2 is None else f"{r2:.6f}",
                }
                rows.append(row)
                if r2 is not None:
                    scored_rows.append(row)
            if scored_rows:
                best = max(scored_rows, key=lambda row: float(row["r2"]))
                lines.append(
                    f"{sweep_name} / {model_label(model_key)}: "
                    f"best={best['transform']} r2={best['r2']}"
                )
            else:
                lines.append(f"{sweep_name} / {model_label(model_key)}: not enough points for R2")

    with (output_dir / "fit_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sweep", "model", "transform", "points", "r2"])
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "fit_summary.txt").write_text("\n".join(lines).rstrip() + "\n")


def write_plots(results: dict, output_dir: Path) -> None:
    for transform in PLOT_TRANSFORMS:
        suffix = "" if transform.key == "linear" else f"_{transform.key}"
        for sweep_name in results["sweep_order"]:
            filename_prefix = "params" if sweep_name == "parameters" else sweep_name
            plot_sweep(
                results,
                sweep_name,
                output_dir / f"{filename_prefix}_vs_capture_mae{suffix}.png",
                transform=transform,
            )
        plot_combined(
            results,
            output_dir / f"combined_scaling{suffix}.png",
            transform=transform,
        )
    write_transform_fit_summary(results, output_dir / "transforms")


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    seed_everything(args.seed)

    captures, screen_size = collect_vision_captures(args.dataset_dir)
    if screen_size is None or not captures:
        raise ValueError("No compatible eye-crop sessions found")

    output_dir = args.output_dir
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"device: {device} | mode: {args.mode} | screen: {screen_size[0]}x{screen_size[1]} | "
        f"captures: {len(captures)}"
    )

    results = {
        "mode": args.mode,
        "dataset_dir": str(args.dataset_dir),
        "dataset_fingerprint": dataset_fingerprint(args.dataset_dir),
        "code_fingerprint": code_fingerprint(),
        "screen_width": screen_size[0],
        "screen_height": screen_size[1],
        "models": list(args.models),
        "sweep_order": list(args.sweeps),
        "sweeps": {sweep_name: {} for sweep_name in args.sweeps},
    }

    for model_key in args.models:
        spec_groups = build_specs_for_model(args, model_key)
        for sweep_name, specs in spec_groups.items():
            if sweep_name not in args.sweeps:
                continue
            sweep_results = []
            print(f"[{model_key}] {sweep_name} sweep: {len(specs)} points")
            for index, spec in enumerate(specs, start=1):
                print(
                    f"  {index}/{len(specs)} | {spec.name} | metadata={spec.metadata}",
                    flush=True,
                )
                sweep_results.append(
                    evaluate_sweep_point(
                        args=args,
                        sweep_name=sweep_name if sweep_name != "parameters" else "parameters",
                        spec=spec,
                        captures=captures,
                        screen_size=screen_size,
                        device=device,
                    )
                )
                results["sweeps"][sweep_name][model_key] = sweep_results
                (output_dir / "summary.json").write_text(json.dumps(results, indent=2))
            results["sweeps"][sweep_name][model_key] = sweep_results

    (output_dir / "summary.json").write_text(json.dumps(results, indent=2))
    write_summary_text(results, output_dir / "summary.txt")
    write_points_csv(results, output_dir / "points.csv")
    write_plots(results, output_dir)
    print(f"saved summary: {output_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
