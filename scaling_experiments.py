from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
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
from vision_dataset import VisionCapture, VisionSample, collect_vision_captures, compute_head_normalization
from vision_eval_report import build_folds
from vision_model import (
    EyeCropModelConfig,
    EyeCropRegressor,
    best_frame_vision_config,
    matched_attention_frame_vision_config,
)
from vision_training import choose_device, seed_everything, train_frame_model


DEFAULT_MODELS = ["attn", "concat"]
DEFAULT_PARAM_MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5]
DEFAULT_DATA_FRACTIONS = [0.15, 0.25, 0.4, 0.6, 0.8, 1.0]
DEFAULT_EPOCH_BUDGETS = [4, 8, 12, 20, 28, 40, 56]
CACHE_DIR = ROOT_DIR / ".cache" / "scaling"
PLOT_STYLE = {
    "attn": {"label": "Attention", "color": "#3b82f6"},
    "concat": {"label": "Concat", "color": "#16a34a"},
}


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
        choices=DEFAULT_MODELS,
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
    return best_frame_vision_config()


def scaled_dim(value: int, multiplier: float, minimum: int = 8, multiple: int = 8) -> int:
    scaled = max(minimum, int(round(value * multiplier)))
    return max(multiple, int(round(scaled / multiple) * multiple))


def scale_config(base: EyeCropModelConfig, multiplier: float) -> EyeCropModelConfig:
    token_dim = scaled_dim(base.token_dim, multiplier, minimum=16, multiple=4)
    return EyeCropModelConfig(
        encoder_channels=tuple(scaled_dim(value, multiplier, minimum=8, multiple=8) for value in base.encoder_channels),
        head_hidden_dims=tuple(scaled_dim(value, multiplier, minimum=16, multiple=8) for value in base.head_hidden_dims),
        regressor_hidden_dims=tuple(
            scaled_dim(value, multiplier, minimum=16, multiple=8) for value in base.regressor_hidden_dims
        ),
        dropout=base.dropout,
        fusion_mode=base.fusion_mode,
        token_dim=token_dim,
        attention_heads=base.attention_heads,
        attention_layers=base.attention_layers,
        attention_dropout=base.attention_dropout,
    )


def model_param_count(config: EyeCropModelConfig) -> int:
    model = EyeCropRegressor(config=config)
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
) -> tuple[float, float]:
    left_eye = load_eye(sample.left_path)
    right_eye = load_eye(sample.right_path)
    head = (sample.head_features - head_mean) / head_scale

    left_tensor = torch.from_numpy(left_eye).unsqueeze(0).unsqueeze(0).to(device)
    right_tensor = torch.from_numpy(right_eye).unsqueeze(0).unsqueeze(0).to(device)
    head_tensor = torch.from_numpy(head.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(left_tensor, right_tensor, head_tensor)
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
        early_stopping_patience=None,
        verbose=False,
    )
    metrics = capture_metric_summary(
        model=result.model,
        eval_captures=eval_captures,
        head_mean=head_mean,
        head_scale=head_scale,
        screen_size=screen_size,
        device=result.device,
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
            fold_results.append(cached)
            continue

        seed_everything(args.seed + fold_index)
        result = evaluate_spec_on_fold(
            spec=spec,
            train_captures=train_captures,
            eval_captures=fold.eval_captures,
            screen_size=screen_size,
            seed=args.seed + fold_index,
            device=device,
        )
        result["holdout_group"] = fold.holdout_group
        save_cached_point(cache_key, result)
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
    for sweep_name in ("parameters", "data", "epochs"):
        lines.append(sweep_name.capitalize())
        for model_key in results["models"]:
            series = results["sweeps"][sweep_name][model_key]
            best = min(series, key=lambda item: item["overall"]["capture_mae_distance_px"])
            lines.append(
                f"- {model_label(model_key)} best: "
                f"{best['overall']['capture_mae_distance_px']:.1f}px capture MAE | "
                f"spec={best['spec_name']} | metadata={best['metadata']}"
            )
        lines.append("")
    output_path.write_text("\n".join(lines).rstrip() + "\n")


def plot_sweep(results: dict, sweep_name: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for model_key in results["models"]:
        series = results["sweeps"][sweep_name][model_key]
        if sweep_name == "parameters":
            x_values = [item["parameter_count"] for item in series]
            x_label = "Parameter Count"
        elif sweep_name == "data":
            x_values = [item["overall"]["train_sample_count"] for item in series]
            x_label = "Train Samples"
        else:
            x_values = [item["metadata"]["epoch_budget"] for item in series]
            x_label = "Epoch Budget"
        y_values = [item["overall"]["capture_mae_distance_px"] for item in series]
        style = PLOT_STYLE[model_key]
        plt.plot(x_values, y_values, marker="o", label=style["label"], color=style["color"])

    plt.title(f"{sweep_name.capitalize()} Sweep ({results['mode']})")
    plt.xlabel(x_label)
    plt.ylabel("Capture MAE Distance (px)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_combined(results: dict, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    sweep_order = ["parameters", "data", "epochs"]
    for axis, sweep_name in zip(axes, sweep_order, strict=True):
        for model_key in results["models"]:
            series = results["sweeps"][sweep_name][model_key]
            if sweep_name == "parameters":
                x_values = [item["parameter_count"] for item in series]
                x_label = "Params"
            elif sweep_name == "data":
                x_values = [item["overall"]["train_sample_count"] for item in series]
                x_label = "Train Samples"
            else:
                x_values = [item["metadata"]["epoch_budget"] for item in series]
                x_label = "Epochs"
            y_values = [item["overall"]["capture_mae_distance_px"] for item in series]
            style = PLOT_STYLE[model_key]
            axis.plot(x_values, y_values, marker="o", label=style["label"], color=style["color"])
            axis.set_xlabel(x_label)
        axis.set_title(sweep_name.capitalize())
        axis.set_ylabel("Capture MAE (px)")
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(results["models"]))
    fig.suptitle(f"Scaling Experiments ({results['mode']})")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


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
        "sweeps": {
            "parameters": {},
            "data": {},
            "epochs": {},
        },
    }

    for model_key in args.models:
        spec_groups = build_specs_for_model(args, model_key)
        for sweep_name, specs in spec_groups.items():
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
    write_summary_text(results, output_dir / "summary.txt")
    plot_sweep(results, "parameters", output_dir / "params_vs_capture_mae.png")
    plot_sweep(results, "data", output_dir / "data_vs_capture_mae.png")
    plot_sweep(results, "epochs", output_dir / "epochs_vs_capture_mae.png")
    plot_combined(results, output_dir / "combined_scaling.png")
    print(f"saved summary: {output_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
