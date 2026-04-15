from dataclasses import dataclass
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from vision_dataset import EyeCropDataset
from vision_model import EyeCropModelConfig, EyeCropRegressor


def choose_device(requested_device: str | None = None) -> torch.device:
    if requested_device:
        return torch.device(requested_device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_loader(
    samples,
    head_mean: np.ndarray,
    head_scale: np.ndarray,
    batch_size: int,
    shuffle: bool,
    augment: bool = False,
) -> DataLoader:
    dataset = EyeCropDataset(
        samples,
        head_mean=head_mean,
        head_scale=head_scale,
        augment=augment,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def mae_in_pixels(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    screen_size: tuple[int, int],
) -> tuple[float, float]:
    errors = (predictions - targets).abs()
    mae_x = float(errors[:, 0].mean().item() * screen_size[0])
    mae_y = float(errors[:, 1].mean().item() * screen_size[1])
    return mae_x, mae_y


def run_epoch(
    model: EyeCropRegressor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    screen_size: tuple[int, int],
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    batch_count = 0
    mae_x_total = 0.0
    mae_y_total = 0.0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            left_eye = batch["left_eye"].to(device)
            right_eye = batch["right_eye"].to(device)
            head_features = batch["head_features"].to(device)
            targets = batch["target"].to(device)

            predictions = model(left_eye, right_eye, head_features)
            loss = nn.functional.smooth_l1_loss(predictions, targets)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_mae_x, batch_mae_y = mae_in_pixels(predictions.detach(), targets, screen_size)
            total_loss += float(loss.item())
            mae_x_total += batch_mae_x
            mae_y_total += batch_mae_y
            batch_count += 1

    if batch_count == 0:
        raise ValueError("No batches were produced for training or evaluation")

    return {
        "loss": total_loss / batch_count,
        "mae_x_px": mae_x_total / batch_count,
        "mae_y_px": mae_y_total / batch_count,
    }


@dataclass
class FrameTrainingResult:
    model: EyeCropRegressor
    device: torch.device
    train_metrics: dict[str, float]
    eval_metrics: dict[str, float] | None
    best_epoch: int | None
    epochs_trained: int


def train_frame_model(
    train_samples,
    eval_samples,
    screen_size: tuple[int, int],
    head_mean: np.ndarray,
    head_scale: np.ndarray,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    model_config: EyeCropModelConfig | None = None,
    augment_train: bool = False,
    early_stopping_patience: int | None = None,
    early_stopping_min_epochs: int = 0,
    early_stopping_min_delta: float = 0.0,
    verbose: bool = True,
) -> FrameTrainingResult:
    train_loader = build_loader(
        train_samples,
        head_mean=head_mean,
        head_scale=head_scale,
        batch_size=batch_size,
        shuffle=True,
        augment=augment_train,
    )
    eval_loader = (
        build_loader(
            eval_samples,
            head_mean=head_mean,
            head_scale=head_scale,
            batch_size=batch_size,
            shuffle=False,
            augment=False,
        )
        if eval_samples
        else None
    )

    model = EyeCropRegressor(
        head_feature_dim=len(head_mean),
        config=model_config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_state_dict = None
    best_eval_loss = None
    best_epoch = None
    last_train_metrics = None
    last_eval_metrics = None
    epochs_without_improvement = 0
    epochs_trained = 0

    for epoch in range(1, epochs + 1):
        epochs_trained = epoch
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            screen_size=screen_size,
        )
        last_train_metrics = train_metrics

        if eval_loader is not None:
            eval_metrics = run_epoch(
                model,
                eval_loader,
                optimizer=None,
                device=device,
                screen_size=screen_size,
            )
            last_eval_metrics = eval_metrics
            score = eval_metrics["loss"]
            improved = best_eval_loss is None or score < (best_eval_loss - early_stopping_min_delta)
            if improved:
                best_eval_loss = score
                best_epoch = epoch
                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        else:
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_epoch = epoch

        if verbose:
            print(
                f"epoch {epoch:02d} | "
                f"train loss {train_metrics['loss']:.4f} | "
                f"train mae {train_metrics['mae_x_px']:.1f}px x / {train_metrics['mae_y_px']:.1f}px y"
                + (
                    f" | eval loss {last_eval_metrics['loss']:.4f}"
                    f" | eval mae {last_eval_metrics['mae_x_px']:.1f}px x / {last_eval_metrics['mae_y_px']:.1f}px y"
                    if last_eval_metrics is not None
                    else ""
                )
            )

        if (
            eval_loader is not None
            and early_stopping_patience is not None
            and epoch >= early_stopping_min_epochs
            and epochs_without_improvement >= early_stopping_patience
        ):
            if verbose:
                print(
                    f"early stop at epoch {epoch:02d} | "
                    f"best epoch {best_epoch:02d} | best eval loss {best_eval_loss:.4f}"
                )
            break

    if best_state_dict is None or last_train_metrics is None:
        raise RuntimeError("Training completed without producing a checkpoint")

    model.load_state_dict(best_state_dict)
    return FrameTrainingResult(
        model=model,
        device=device,
        train_metrics=last_train_metrics,
        eval_metrics=last_eval_metrics,
        best_epoch=best_epoch,
        epochs_trained=epochs_trained,
    )
