import argparse
import json
from pathlib import Path
import torch

from constants import (
    DATASET_DIR,
    VISION_AUGMENT_TRAIN,
    VISION_BATCH_SIZE,
    VISION_CONCAT_METRICS_PATH,
    VISION_CONCAT_MODEL_PATH,
    VISION_EPOCHS,
    VISION_HEAD_FEATURE_KEYS,
    VISION_LEARNING_RATE,
    VISION_ATTENTION_MATCHED_METRICS_PATH,
    VISION_ATTENTION_MATCHED_MODEL_PATH,
    VISION_WEIGHT_DECAY,
)
from vision_dataset import (
    collect_vision_samples,
    compute_head_normalization,
    split_vision_samples,
)
from vision_training import choose_device, seed_everything, train_frame_model
from vision_model import (
    VisionCheckpointMetadata,
    best_frame_vision_config,
    build_checkpoint_payload,
    matched_attention_frame_vision_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the eye-crop gaze regressor from manual collection sessions.",
    )
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--epochs", type=int, default=VISION_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=VISION_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=VISION_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=VISION_WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--model",
        choices=["attn", "concat"],
        default="attn",
        help="Which live vision model to train.",
    )
    parser.add_argument(
        "--preset",
        choices=["concat_best", "attention_matched"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable training-time eye-crop augmentation.",
    )
    return parser.parse_args()


def normalize_model_choice(model: str | None, preset: str | None) -> str:
    if preset == "attention_matched":
        return "attn"
    if preset == "concat_best":
        return "concat"
    return model or "attn"


def output_paths_for_model(model: str) -> tuple[Path, Path]:
    if model == "attn":
        return VISION_ATTENTION_MATCHED_MODEL_PATH, VISION_ATTENTION_MATCHED_METRICS_PATH
    return VISION_CONCAT_MODEL_PATH, VISION_CONCAT_METRICS_PATH


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    model_choice = normalize_model_choice(args.model, args.preset)

    all_samples, screen_size = collect_vision_samples(args.dataset_dir)
    if screen_size is None:
        raise ValueError(
            "No compatible sessions found for vision-model training. "
            "Collect fresh eye-crop sessions first."
        )

    train_samples, eval_samples = split_vision_samples(all_samples)
    if not train_samples:
        raise ValueError(
            "No train-tagged eye-crop samples are available. "
            "Collect at least one train capture with the current collector."
        )

    head_mean, head_scale = compute_head_normalization(train_samples)
    if model_choice == "attn":
        model_config = matched_attention_frame_vision_config()
    else:
        model_config = best_frame_vision_config()
    augment_train = VISION_AUGMENT_TRAIN and not args.no_augment
    model_path, metrics_path = output_paths_for_model(model_choice)

    print(f"device: {device}")
    print(
        f"screen: {screen_size[0]}x{screen_size[1]} | "
        f"train samples: {len(train_samples)} | eval samples: {len(eval_samples)}"
    )
    print(f"model: {model_choice} | fusion: {model_config.fusion_mode}")
    result = train_frame_model(
        train_samples=train_samples,
        eval_samples=eval_samples,
        screen_size=screen_size,
        head_mean=head_mean,
        head_scale=head_scale,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        model_config=model_config,
        augment_train=augment_train,
        verbose=True,
    )

    model = result.model
    last_train_metrics = result.train_metrics
    last_eval_metrics = result.eval_metrics
    model_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = VisionCheckpointMetadata(
        screen_size=screen_size,
        head_feature_keys=list(VISION_HEAD_FEATURE_KEYS),
        head_mean=head_mean.tolist(),
        head_scale=head_scale.tolist(),
        model_config=model_config.to_dict(),
        train_sample_count=len(train_samples),
        eval_sample_count=len(eval_samples),
        eval_mae_x_px=(last_eval_metrics["mae_x_px"] if last_eval_metrics is not None else None),
        eval_mae_y_px=(last_eval_metrics["mae_y_px"] if last_eval_metrics is not None else None),
    )
    checkpoint = build_checkpoint_payload(model, metadata)
    torch.save(checkpoint, model_path)

    metrics_payload = {
        "screen_width": screen_size[0],
        "screen_height": screen_size[1],
        "device": str(device),
        "train_sample_count": len(train_samples),
        "eval_sample_count": len(eval_samples),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "model": model_choice,
        "preset": ("attention_matched" if model_choice == "attn" else "concat_best"),
        "augment_train": augment_train,
        "model_config": model_config.to_dict(),
        "train_metrics": last_train_metrics,
        "eval_metrics": last_eval_metrics,
        "model_path": str(model_path),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))

    print(f"saved model: {model_path}")
    print(f"saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
