from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from constants import VISION_MODEL_PATH
from vision_model import EyeCropModelConfig, EyeCropRegressor


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def choose_device(requested_device: str | None = None) -> torch.device:
    if requested_device:
        return torch.device(requested_device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preprocess_eye_crop(crop: np.ndarray) -> torch.Tensor:
    if crop.ndim == 3:
        grayscale = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        grayscale = crop
    normalized = grayscale.astype(np.float32) / 255.0
    normalized = (normalized - 0.5) / 0.5
    return torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)


@dataclass
class VisionGazePredictor:
    checkpoint_path: Path
    screen_size: tuple[int, int]
    head_feature_keys: list[str]
    head_mean: np.ndarray
    head_scale: np.ndarray
    model: EyeCropRegressor
    device: torch.device
    train_sample_count: int
    eval_sample_count: int
    eval_mae_x_px: float | None
    eval_mae_y_px: float | None

    @property
    def label(self) -> str:
        return self.checkpoint_path.stem

    @property
    def metrics(self) -> dict[str, float] | None:
        if self.eval_mae_x_px is None or self.eval_mae_y_px is None:
            return None
        return {
            "eval_mae_x_px": self.eval_mae_x_px,
            "eval_mae_y_px": self.eval_mae_y_px,
        }

    def predict_normalized(
        self,
        eye_crops: dict[str, np.ndarray],
        payload: dict[str, float],
    ) -> tuple[float, float]:
        head_features = np.array(
            [float(payload.get(key, 0.0)) for key in self.head_feature_keys],
            dtype=np.float32,
        )
        head_features = (head_features - self.head_mean) / self.head_scale

        left_eye = preprocess_eye_crop(eye_crops["left"]).to(self.device)
        right_eye = preprocess_eye_crop(eye_crops["right"]).to(self.device)
        head_tensor = torch.from_numpy(head_features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.model(left_eye, right_eye, head_tensor)
        normalized = prediction.squeeze(0).detach().cpu().numpy()
        return (
            clamp01(float(normalized[0])),
            clamp01(float(normalized[1])),
        )


def load_vision_predictor(
    checkpoint_path: Path = VISION_MODEL_PATH,
    screen_size_filter: tuple[int, int] | None = None,
    requested_device: str | None = None,
) -> VisionGazePredictor | None:
    if not checkpoint_path.exists():
        return None

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    screen_size = tuple(int(value) for value in checkpoint["screen_size"])
    if screen_size_filter is not None and screen_size != screen_size_filter:
        return None

    head_feature_keys = list(checkpoint.get("head_feature_keys") or [])
    if not head_feature_keys:
        raise ValueError("Vision checkpoint is missing auxiliary feature keys")

    model_config = EyeCropModelConfig.from_dict(checkpoint.get("model_config"))
    model = EyeCropRegressor(
        head_feature_dim=len(head_feature_keys),
        config=model_config,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    device = choose_device(requested_device)
    model.to(device)
    model.eval()

    head_mean = np.array(checkpoint["head_mean"], dtype=np.float32)
    head_scale = np.array(checkpoint["head_scale"], dtype=np.float32)
    head_scale[head_scale < 1e-6] = 1.0

    return VisionGazePredictor(
        checkpoint_path=checkpoint_path,
        screen_size=screen_size,
        head_feature_keys=head_feature_keys,
        head_mean=head_mean,
        head_scale=head_scale,
        model=model,
        device=device,
        train_sample_count=int(checkpoint.get("train_sample_count", 0)),
        eval_sample_count=int(checkpoint.get("eval_sample_count", 0)),
        eval_mae_x_px=(
            float(checkpoint["eval_mae_x_px"])
            if checkpoint.get("eval_mae_x_px") is not None
            else None
        ),
        eval_mae_y_px=(
            float(checkpoint["eval_mae_y_px"])
            if checkpoint.get("eval_mae_y_px") is not None
            else None
        ),
    )
