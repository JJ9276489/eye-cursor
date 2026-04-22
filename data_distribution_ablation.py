from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path

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
from scaling_experiments import (
    base_config_for_model,
    model_param_count,
    predict_sample,
    scale_config,
)
from vision_dataset import (
    VisionCapture,
    VisionSample,
    collect_vision_captures,
    compute_head_normalization,
)
from vision_eval_report import REGION_COLS, REGION_ROWS, region_label
from vision_training import choose_device, seed_everything, train_frame_model


REGION_LABELS = [f"{row}-{column}" for row in REGION_ROWS for column in REGION_COLS]


@dataclass(frozen=True)
class TrainVariant:
    name: str
    train_samples: list[VisionSample]
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether screen-region data distribution affects frame-vision gaze training."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "reports" / "distribution" / "latest",
    )
    parser.add_argument(
        "--model",
        choices=["concat", "spatial", "spatial_geom", "clifford", "vit", "attn"],
        default="spatial_geom",
    )
    parser.add_argument("--param-multiplier", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=VISION_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=VISION_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=VISION_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=VISION_WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable eye-crop augmentation during training.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=4,
        help="Stop after this many epochs without eval-loss improvement.",
    )
    parser.add_argument(
        "--early-stopping-min-epochs",
        type=int,
        default=8,
        help="Minimum epochs before early stopping can trigger.",
    )
    return parser.parse_args()


def capture_split(captures: list[VisionCapture]) -> tuple[list[VisionCapture], list[VisionCapture]]:
    train_captures = [capture for capture in captures if capture.split != "eval"]
    eval_captures = [capture for capture in captures if capture.split == "eval"]
    if not train_captures:
        raise ValueError("No train captures found")
    if not eval_captures:
        raise ValueError("No eval captures found")
    return train_captures, eval_captures


def flatten_samples(captures: list[VisionCapture]) -> list[VisionSample]:
    return [sample for capture in captures for sample in capture.samples]


def grouped_samples_by_region(samples: list[VisionSample]) -> dict[str, list[VisionSample]]:
    groups: dict[str, list[VisionSample]] = defaultdict(list)
    for sample in samples:
        groups[region_label(sample.target_x, sample.target_y)].append(sample)
    return dict(groups)


def grouped_captures_by_region(captures: list[VisionCapture]) -> dict[str, list[VisionCapture]]:
    groups: dict[str, list[VisionCapture]] = defaultdict(list)
    for capture in captures:
        groups[region_label(capture.target_x, capture.target_y)].append(capture)
    return dict(groups)


def balanced_region_samples(
    samples: list[VisionSample],
    seed: int,
    target_total: int | None = None,
) -> list[VisionSample]:
    groups = grouped_samples_by_region(samples)
    populated_regions = [label for label in REGION_LABELS if groups.get(label)]
    if not populated_regions:
        raise ValueError("Cannot balance an empty sample set")

    total = len(samples) if target_total is None else target_total
    base_count = total // len(populated_regions)
    remainder = total % len(populated_regions)
    rng = np.random.default_rng(seed)

    balanced: list[VisionSample] = []
    for index, label in enumerate(populated_regions):
        group = groups[label]
        count = base_count + (1 if index < remainder else 0)
        indices = rng.choice(
            len(group),
            size=count,
            replace=count > len(group),
        )
        balanced.extend(group[int(item)] for item in indices)

    shuffled = rng.permutation(len(balanced))
    return [balanced[int(index)] for index in shuffled]


def count_regions(captures: list[VisionCapture], samples: list[VisionSample]) -> dict[str, dict[str, int]]:
    capture_groups = grouped_captures_by_region(captures)
    sample_groups = grouped_samples_by_region(samples)
    return {
        label: {
            "capture_count": len(capture_groups.get(label, [])),
            "sample_count": len(sample_groups.get(label, [])),
        }
        for label in REGION_LABELS
    }


def mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def evaluate_regions(
    model,
    eval_captures: list[VisionCapture],
    head_mean: np.ndarray,
    head_scale: np.ndarray,
    screen_size: tuple[int, int],
    device: torch.device,
    extra_mean: np.ndarray | None,
    extra_scale: np.ndarray | None,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    width, height = screen_size
    region_values: dict[str, dict[str, list[float] | int]] = {
        label: {
            "frame_x": [],
            "frame_y": [],
            "frame_d": [],
            "capture_x": [],
            "capture_y": [],
            "capture_d": [],
            "capture_count": 0,
            "sample_count": 0,
        }
        for label in REGION_LABELS
    }

    for capture in eval_captures:
        label = region_label(capture.target_x, capture.target_y)
        values = region_values[label]
        values["capture_count"] = int(values["capture_count"]) + 1
        values["sample_count"] = int(values["sample_count"]) + len(capture.samples)

        predictions: list[tuple[float, float]] = []
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
            values["frame_x"].append(err_x)
            values["frame_y"].append(err_y)
            values["frame_d"].append(math.sqrt(err_x**2 + err_y**2))
            predictions.append((pred_x, pred_y))

        mean_x = float(np.mean([item[0] for item in predictions]))
        mean_y = float(np.mean([item[1] for item in predictions]))
        err_x = abs(mean_x - capture.target_x) * width
        err_y = abs(mean_y - capture.target_y) * height
        values["capture_x"].append(err_x)
        values["capture_y"].append(err_y)
        values["capture_d"].append(math.sqrt(err_x**2 + err_y**2))

    region_summary: dict[str, dict[str, float]] = {}
    for label, values in region_values.items():
        region_summary[label] = {
            "eval_capture_count": int(values["capture_count"]),
            "eval_sample_count": int(values["sample_count"]),
            "frame_mae_x_px": mean_or_nan(values["frame_x"]),
            "frame_mae_y_px": mean_or_nan(values["frame_y"]),
            "frame_mae_distance_px": mean_or_nan(values["frame_d"]),
            "capture_mae_x_px": mean_or_nan(values["capture_x"]),
            "capture_mae_y_px": mean_or_nan(values["capture_y"]),
            "capture_mae_distance_px": mean_or_nan(values["capture_d"]),
        }

    populated = [
        summary
        for summary in region_summary.values()
        if summary["eval_capture_count"] > 0
    ]
    overall = {
        "frame_mae_x_px": mean_or_nan([item["frame_mae_x_px"] for item in populated]),
        "frame_mae_y_px": mean_or_nan([item["frame_mae_y_px"] for item in populated]),
        "frame_mae_distance_px": mean_or_nan(
            [item["frame_mae_distance_px"] for item in populated]
        ),
        "capture_mae_x_px": mean_or_nan([item["capture_mae_x_px"] for item in populated]),
        "capture_mae_y_px": mean_or_nan([item["capture_mae_y_px"] for item in populated]),
        "capture_mae_distance_px": mean_or_nan(
            [item["capture_mae_distance_px"] for item in populated]
        ),
    }
    return region_summary, overall


def pearson(values_x: list[float], values_y: list[float]) -> float | None:
    x = np.asarray(values_x, dtype=np.float64)
    y = np.asarray(values_y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    output = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and values[order[end + 1]] == values[order[index]]:
            end += 1
        rank = (index + end) / 2.0 + 1.0
        for item in range(index, end + 1):
            output[order[item]] = rank
        index = end + 1
    return output


def spearman(values_x: list[float], values_y: list[float]) -> float | None:
    return pearson(ranks(values_x), ranks(values_y))


def distribution_correlations(
    train_counts: dict[str, dict[str, int]],
    region_metrics: dict[str, dict[str, float]],
) -> dict[str, float | None]:
    labels = [
        label
        for label in REGION_LABELS
        if region_metrics[label]["eval_capture_count"] > 0
        and math.isfinite(region_metrics[label]["capture_mae_distance_px"])
    ]
    sample_counts = [float(train_counts[label]["sample_count"]) for label in labels]
    capture_counts = [float(train_counts[label]["capture_count"]) for label in labels]
    errors = [float(region_metrics[label]["capture_mae_distance_px"]) for label in labels]
    return {
        "train_samples_vs_capture_mae_pearson": pearson(sample_counts, errors),
        "train_samples_vs_capture_mae_spearman": spearman(sample_counts, errors),
        "train_captures_vs_capture_mae_pearson": pearson(capture_counts, errors),
        "train_captures_vs_capture_mae_spearman": spearman(capture_counts, errors),
    }


def run_variant(
    args: argparse.Namespace,
    variant: TrainVariant,
    eval_captures: list[VisionCapture],
    screen_size: tuple[int, int],
    device: torch.device,
    model_config,
    seed_offset: int,
) -> dict:
    seed_everything(args.seed + seed_offset)
    head_mean, head_scale = compute_head_normalization(variant.train_samples)
    result = train_frame_model(
        train_samples=variant.train_samples,
        eval_samples=flatten_samples(eval_captures),
        screen_size=screen_size,
        head_mean=head_mean,
        head_scale=head_scale,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        model_config=model_config,
        augment_train=not args.no_augment,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_epochs=args.early_stopping_min_epochs,
        verbose=True,
    )
    region_metrics, overall = evaluate_regions(
        model=result.model,
        eval_captures=eval_captures,
        head_mean=head_mean,
        head_scale=head_scale,
        screen_size=screen_size,
        device=result.device,
        extra_mean=result.extra_mean,
        extra_scale=result.extra_scale,
    )
    return {
        "name": variant.name,
        "description": variant.description,
        "train_sample_count": len(variant.train_samples),
        "best_epoch": result.best_epoch,
        "epochs_trained": result.epochs_trained,
        "train_metrics": result.train_metrics,
        "eval_metrics": result.eval_metrics,
        "overall": overall,
        "regions": region_metrics,
    }


def write_region_csv(results: dict, output_path: Path) -> None:
    columns = [
        "variant",
        "region",
        "train_captures",
        "train_samples",
        "eval_captures",
        "eval_samples",
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
        for variant in results["variants"]:
            for label in REGION_LABELS:
                train_counts = results["train_region_counts"][variant["name"]][label]
                region = variant["regions"][label]
                writer.writerow(
                    {
                        "variant": variant["name"],
                        "region": label,
                        "train_captures": train_counts["capture_count"],
                        "train_samples": train_counts["sample_count"],
                        "eval_captures": region["eval_capture_count"],
                        "eval_samples": region["eval_sample_count"],
                        "capture_mae_distance_px": region["capture_mae_distance_px"],
                        "capture_mae_x_px": region["capture_mae_x_px"],
                        "capture_mae_y_px": region["capture_mae_y_px"],
                        "frame_mae_distance_px": region["frame_mae_distance_px"],
                        "frame_mae_x_px": region["frame_mae_x_px"],
                        "frame_mae_y_px": region["frame_mae_y_px"],
                    }
                )


def write_summary_md(results: dict, output_path: Path) -> None:
    lines = [
        "# Region Distribution Ablation",
        "",
        f"Dataset: `{results['capture_count']}` captures / `{results['sample_count']}` samples",
        f"Screen: `{results['screen_width']}x{results['screen_height']}`",
        f"Model: `{results['model']} {results['param_multiplier']}x`",
        f"Parameters: `{results['parameter_count']}`",
        "",
        "## Hypothesis",
        "",
        (
            "If the training distribution is a bottleneck, reweighting training so each "
            "screen region contributes equally should reduce per-region error imbalance "
            "or improve underrepresented regions on the label-holdout eval split."
        ),
        "",
        "## Overall Results",
        "",
        "| Training variant | Capture MAE | X MAE | Y MAE | Best epoch | Epochs trained |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in results["variants"]:
        overall = variant["overall"]
        lines.append(
            f"| `{variant['name']}` | "
            f"`{overall['capture_mae_distance_px']:.1f}px` | "
            f"`{overall['capture_mae_x_px']:.1f}px` | "
            f"`{overall['capture_mae_y_px']:.1f}px` | "
            f"`{variant['best_epoch']}` | "
            f"`{variant['epochs_trained']}` |"
        )

    lines.extend(
        [
            "",
            "## Region Results",
            "",
            "| Region | Natural train samples | Balanced train samples | Eval captures | Natural MAE | Balanced MAE | Delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    natural = next(item for item in results["variants"] if item["name"] == "natural")
    balanced = next(item for item in results["variants"] if item["name"] == "region_balanced")
    for label in REGION_LABELS:
        natural_region = natural["regions"][label]
        balanced_region = balanced["regions"][label]
        natural_error = natural_region["capture_mae_distance_px"]
        balanced_error = balanced_region["capture_mae_distance_px"]
        delta = balanced_error - natural_error
        lines.append(
            f"| `{label}` | "
            f"`{results['train_region_counts']['natural'][label]['sample_count']}` | "
            f"`{results['train_region_counts']['region_balanced'][label]['sample_count']}` | "
            f"`{natural_region['eval_capture_count']}` | "
            f"`{natural_error:.1f}px` | "
            f"`{balanced_error:.1f}px` | "
            f"`{delta:+.1f}px` |"
        )

    lines.extend(
        [
            "",
            "## Correlations",
            "",
            "| Training variant | Pearson train samples vs MAE | Spearman train samples vs MAE |",
            "| --- | ---: | ---: |",
        ]
    )
    for variant in results["variants"]:
        correlations = variant["distribution_correlations"]
        pearson_value = correlations["train_samples_vs_capture_mae_pearson"]
        spearman_value = correlations["train_samples_vs_capture_mae_spearman"]
        lines.append(
            f"| `{variant['name']}` | "
            f"`{pearson_value:+.3f}` | "
            f"`{spearman_value:+.3f}` |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "This is a label-holdout ablation, so it tests training distribution bias "
                "when every populated region is represented in training. It is different "
                "from region holdout, where the target region is intentionally absent from "
                "the training fold."
            ),
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")


def plot_results(results: dict, output_path: Path) -> None:
    labels = REGION_LABELS
    x = np.arange(len(labels))
    natural = next(item for item in results["variants"] if item["name"] == "natural")
    balanced = next(item for item in results["variants"] if item["name"] == "region_balanced")
    natural_errors = [natural["regions"][label]["capture_mae_distance_px"] for label in labels]
    balanced_errors = [balanced["regions"][label]["capture_mae_distance_px"] for label in labels]
    train_samples = [
        results["train_region_counts"]["natural"][label]["sample_count"]
        for label in labels
    ]

    fig, axis_error = plt.subplots(figsize=(11, 5.5))
    width = 0.38
    axis_error.bar(x - width / 2, natural_errors, width=width, label="Natural", color="#2563eb")
    axis_error.bar(
        x + width / 2,
        balanced_errors,
        width=width,
        label="Region balanced",
        color="#ea580c",
    )
    axis_error.set_ylabel("Capture MAE distance (px)")
    axis_error.set_xticks(x)
    axis_error.set_xticklabels(labels, rotation=35, ha="right")
    axis_error.grid(axis="y", alpha=0.25)

    axis_count = axis_error.twinx()
    axis_count.plot(x, train_samples, color="#111827", marker="o", label="Natural train samples")
    axis_count.set_ylabel("Natural train samples")

    handles_a, labels_a = axis_error.get_legend_handles_labels()
    handles_b, labels_b = axis_count.get_legend_handles_labels()
    axis_error.legend(handles_a + handles_b, labels_a + labels_b, loc="upper left")
    fig.suptitle("Region Distribution Ablation")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    captures, screen_size = collect_vision_captures(args.dataset_dir)
    if screen_size is None or not captures:
        raise ValueError("No compatible captures found")

    train_captures, eval_captures = capture_split(captures)
    natural_train_samples = flatten_samples(train_captures)
    balanced_train_samples = balanced_region_samples(
        natural_train_samples,
        seed=args.seed,
        target_total=len(natural_train_samples),
    )
    variants = [
        TrainVariant(
            name="natural",
            train_samples=natural_train_samples,
            description="Original collector train distribution.",
        ),
        TrainVariant(
            name="region_balanced",
            train_samples=balanced_train_samples,
            description="Constant-size sample resampling with equal per-region training weight.",
        ),
    ]
    model_config = scale_config(base_config_for_model(args.model), args.param_multiplier)

    print(f"device: {device}")
    print(
        f"dataset: {len(captures)} captures / {sum(len(c.samples) for c in captures)} samples | "
        f"screen {screen_size[0]}x{screen_size[1]}"
    )
    print(
        f"model: {args.model} | param multiplier {args.param_multiplier:.2f} | "
        f"params {model_param_count(model_config)}"
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model": args.model,
        "param_multiplier": args.param_multiplier,
        "parameter_count": model_param_count(model_config),
        "screen_width": screen_size[0],
        "screen_height": screen_size[1],
        "capture_count": len(captures),
        "sample_count": sum(len(capture.samples) for capture in captures),
        "train_capture_count": len(train_captures),
        "eval_capture_count": len(eval_captures),
        "train_region_counts": {},
        "eval_region_counts": count_regions(eval_captures, flatten_samples(eval_captures)),
        "variants": [],
    }

    for offset, variant in enumerate(variants):
        print(f"\ntraining variant: {variant.name}")
        result = run_variant(
            args=args,
            variant=variant,
            eval_captures=eval_captures,
            screen_size=screen_size,
            device=device,
            model_config=model_config,
            seed_offset=offset,
        )
        train_counts = count_regions(train_captures, variant.train_samples)
        result["distribution_correlations"] = distribution_correlations(
            train_counts=train_counts,
            region_metrics=result["regions"],
        )
        results["train_region_counts"][variant.name] = train_counts
        results["variants"].append(result)
        print(
            f"{variant.name}: "
            f"{result['overall']['capture_mae_distance_px']:.1f}px capture MAE | "
            f"best epoch {result['best_epoch']}"
        )

    (output_dir / "summary.json").write_text(json.dumps(results, indent=2))
    write_region_csv(results, output_dir / "region_metrics.csv")
    write_summary_md(results, output_dir / "summary.md")
    plot_results(results, output_dir / "region_distribution_ablation.svg")
    print(f"\nwrote: {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
