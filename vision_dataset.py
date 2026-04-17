from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from constants import DATASET_DIR, VISION_HEAD_FEATURE_KEYS


def latest_session_dirs(directory: Path = DATASET_DIR) -> list[Path]:
    return sorted(
        [path for path in directory.glob("session-*") if path.is_dir()],
        reverse=True,
    )


@dataclass
class VisionSample:
    session_id: str
    capture_index: int
    split: str
    target_x: float
    target_y: float
    payload: dict[str, float]
    head_features: np.ndarray
    left_path: Path
    right_path: Path


@dataclass
class VisionCapture:
    session_id: str
    capture_index: int
    split: str
    target_x: float
    target_y: float
    samples: list[VisionSample]


def _load_session_metadata(session_dir: Path) -> dict | None:
    session_path = session_dir / "session.json"
    if not session_path.exists():
        return None
    return json.loads(session_path.read_text())


def _capture_records(session_dir: Path) -> list[dict]:
    captures_path = session_dir / "captures.jsonl"
    if not captures_path.exists():
        return []

    records: list[dict] = []
    for line in captures_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _sample_head_features(sample: dict) -> np.ndarray | None:
    values = []
    for key in VISION_HEAD_FEATURE_KEYS:
        if key not in sample:
            return None
        values.append(float(sample[key]))
    return np.array(values, dtype=np.float32)


def _sample_eye_paths(session_dir: Path, sample: dict) -> tuple[Path, Path] | None:
    left_path = sample.get("left_path")
    right_path = sample.get("right_path")
    if not left_path or not right_path:
        return None

    absolute_left = session_dir / left_path
    absolute_right = session_dir / right_path
    if not absolute_left.exists() or not absolute_right.exists():
        return None
    return absolute_left, absolute_right


def collect_vision_samples(
    directory: Path = DATASET_DIR,
    screen_size_filter: tuple[int, int] | None = None,
) -> tuple[list[VisionSample], tuple[int, int] | None]:
    captures, dataset_screen_size = collect_vision_captures(
        directory=directory,
        screen_size_filter=screen_size_filter,
    )
    samples = [sample for capture in captures for sample in capture.samples]
    return samples, dataset_screen_size


def collect_vision_captures(
    directory: Path = DATASET_DIR,
    screen_size_filter: tuple[int, int] | None = None,
) -> tuple[list[VisionCapture], tuple[int, int] | None]:
    captures: list[VisionCapture] = []
    dataset_screen_size: tuple[int, int] | None = screen_size_filter

    for session_dir in latest_session_dirs(directory):
        session_metadata = _load_session_metadata(session_dir)
        if session_metadata is None:
            continue

        session_screen_size = (
            int(session_metadata["screen_width"]),
            int(session_metadata["screen_height"]),
        )
        if dataset_screen_size is None:
            dataset_screen_size = session_screen_size
        if session_screen_size != dataset_screen_size:
            continue

        session_id = str(session_metadata["session_id"])
        for capture_index, capture in enumerate(_capture_records(session_dir), start=1):
            split = str(capture.get("split", "train"))
            capture_samples: list[VisionSample] = []
            for sample in capture.get("samples", []):
                eye_paths = _sample_eye_paths(session_dir, sample)
                head_features = _sample_head_features(sample)
                if eye_paths is None or head_features is None:
                    continue

                capture_samples.append(
                    VisionSample(
                        session_id=session_id,
                        capture_index=capture_index,
                        split=split,
                        target_x=float(sample["target_x"]),
                        target_y=float(sample["target_y"]),
                        payload={key: value for key, value in sample.items()},
                        head_features=head_features,
                        left_path=eye_paths[0],
                        right_path=eye_paths[1],
                    )
                )
            if capture_samples:
                captures.append(
                    VisionCapture(
                        session_id=session_id,
                        capture_index=capture_index,
                        split=split,
                        target_x=float(capture["target_x"]),
                        target_y=float(capture["target_y"]),
                        samples=capture_samples,
                    )
                )

    return captures, dataset_screen_size


def split_vision_samples(
    samples: list[VisionSample],
) -> tuple[list[VisionSample], list[VisionSample]]:
    train_samples = [sample for sample in samples if sample.split != "eval"]
    eval_samples = [sample for sample in samples if sample.split == "eval"]
    return train_samples, eval_samples


def _normalize_feature_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def compute_head_normalization(samples: list[VisionSample]) -> tuple[np.ndarray, np.ndarray]:
    if not samples:
        raise ValueError("Cannot compute normalization without training samples")

    matrix = np.stack([sample.head_features for sample in samples]).astype(np.float32)
    return _normalize_feature_matrix(matrix)


def sample_payload_features(sample: VisionSample, keys: tuple[str, ...]) -> np.ndarray:
    values = []
    for key in keys:
        if key not in sample.payload:
            raise KeyError(f"Sample is missing payload feature: {key}")
        values.append(float(sample.payload[key]))
    return np.asarray(values, dtype=np.float32)


def compute_payload_feature_normalization(
    samples: list[VisionSample],
    keys: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    if not keys:
        return np.zeros((0,), dtype=np.float32), np.ones((0,), dtype=np.float32)
    if not samples:
        raise ValueError("Cannot compute payload-feature normalization without training samples")

    matrix = np.stack([sample_payload_features(sample, keys) for sample in samples]).astype(np.float32)
    return _normalize_feature_matrix(matrix)


def load_grayscale_eye_crop(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read eye crop: {path}")

    image = image.astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5
    return image


class EyeCropDataset(Dataset):
    def __init__(
        self,
        samples: list[VisionSample],
        head_mean: np.ndarray,
        head_scale: np.ndarray,
        augment: bool = False,
        extra_feature_keys: tuple[str, ...] = (),
        extra_mean: np.ndarray | None = None,
        extra_scale: np.ndarray | None = None,
    ) -> None:
        self.samples = samples
        self.head_mean = head_mean.astype(np.float32)
        self.head_scale = head_scale.astype(np.float32)
        self.augment = augment
        self.extra_feature_keys = extra_feature_keys
        self.extra_mean = (
            extra_mean.astype(np.float32)
            if extra_mean is not None
            else np.zeros((len(extra_feature_keys),), dtype=np.float32)
        )
        self.extra_scale = (
            extra_scale.astype(np.float32)
            if extra_scale is not None
            else np.ones((len(extra_feature_keys),), dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        left_eye = load_grayscale_eye_crop(sample.left_path)
        right_eye = load_grayscale_eye_crop(sample.right_path)
        if self.augment:
            left_eye = self._augment_eye(left_eye)
            right_eye = self._augment_eye(right_eye)
        head_features = (sample.head_features - self.head_mean) / self.head_scale
        extra_features = sample_payload_features(sample, self.extra_feature_keys)
        if len(self.extra_feature_keys) > 0:
            extra_features = (extra_features - self.extra_mean) / self.extra_scale
        target = np.array([sample.target_x, sample.target_y], dtype=np.float32)

        return {
            "left_eye": torch.from_numpy(left_eye).unsqueeze(0),
            "right_eye": torch.from_numpy(right_eye).unsqueeze(0),
            "head_features": torch.from_numpy(head_features),
            "extra_features": torch.from_numpy(extra_features.astype(np.float32)),
            "target": torch.from_numpy(target),
        }

    def _augment_eye(self, eye: np.ndarray) -> np.ndarray:
        image = ((eye * 0.5) + 0.5).astype(np.float32)
        gain = float(np.random.uniform(0.9, 1.1))
        bias = float(np.random.uniform(-0.05, 0.05))
        noise = np.random.normal(0.0, 0.01, size=image.shape).astype(np.float32)
        image = np.clip(image * gain + bias + noise, 0.0, 1.0)
        return ((image - 0.5) / 0.5).astype(np.float32)
