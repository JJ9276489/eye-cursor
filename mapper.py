from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from constants import DATASET_DIR, MAPPER_FEATURE_KEYS, POLYNOMIAL_DEGREE, RIDGE_ALPHA


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def polynomial_pairs(feature_count: int) -> list[tuple[int, int]]:
    return [
        (left_index, right_index)
        for left_index in range(feature_count)
        for right_index in range(left_index, feature_count)
    ]


@dataclass
class PolynomialRidgeGazeMapper:
    dataset_label: str
    feature_keys: list[str]
    screen_size: tuple[int, int]
    base_mean: np.ndarray
    base_scale: np.ndarray
    pair_indices: list[tuple[int, int]]
    coef_x: np.ndarray
    coef_y: np.ndarray
    train_sample_count: int
    eval_sample_count: int
    ridge_alpha: float
    eval_mae_x_px: float | None
    eval_mae_y_px: float | None

    @property
    def label(self) -> str:
        return self.dataset_label

    @property
    def metrics(self) -> dict[str, float] | None:
        if self.eval_mae_x_px is None or self.eval_mae_y_px is None:
            return None
        return {
            "eval_mae_x_px": self.eval_mae_x_px,
            "eval_mae_y_px": self.eval_mae_y_px,
        }

    def _expand_from_payload(self, payload: dict[str, float]) -> np.ndarray:
        base_row = np.array(
            [float(payload.get(key, 0.0)) for key in self.feature_keys],
            dtype=np.float64,
        )
        standardized = (base_row - self.base_mean) / self.base_scale
        expanded = [1.0]
        expanded.extend(float(value) for value in standardized)
        if POLYNOMIAL_DEGREE >= 2:
            expanded.extend(
                float(standardized[left_index] * standardized[right_index])
                for left_index, right_index in self.pair_indices
            )
        return np.array(expanded, dtype=np.float64)

    def predict_normalized(self, payload: dict[str, float]) -> tuple[float, float]:
        row = self._expand_from_payload(payload)
        return (
            clamp01(float(row @ self.coef_x)),
            clamp01(float(row @ self.coef_y)),
        )

    def predict_pixels(self, payload: dict[str, float]) -> tuple[int, int]:
        normalized_x, normalized_y = self.predict_normalized(payload)
        return (
            int(normalized_x * (self.screen_size[0] - 1)),
            int(normalized_y * (self.screen_size[1] - 1)),
        )


def latest_session_dirs(directory: Path = DATASET_DIR) -> list[Path]:
    return sorted(
        [path for path in directory.glob("session-*") if path.is_dir()],
        reverse=True,
    )


def iter_capture_files(directory: Path = DATASET_DIR) -> list[Path]:
    capture_files = []
    for session_dir in latest_session_dirs(directory):
        capture_file = session_dir / "captures.jsonl"
        if capture_file.exists():
            capture_files.append(capture_file)
    return capture_files


def iter_capture_records(directory: Path = DATASET_DIR) -> list[tuple[Path, dict]]:
    records: list[tuple[Path, dict]] = []
    for capture_file in iter_capture_files(directory):
        for line in capture_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            records.append((capture_file, json.loads(line)))
    return records


def fit_ridge_regression(
    design: np.ndarray, targets: np.ndarray, alpha: float
) -> np.ndarray:
    regularizer = np.eye(design.shape[1], dtype=np.float64) * alpha
    regularizer[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + regularizer, design.T @ targets)


def build_design_matrix(base_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    base_mean = base_matrix.mean(axis=0)
    base_scale = base_matrix.std(axis=0)
    base_scale[base_scale < 1e-6] = 1.0
    standardized = (base_matrix - base_mean) / base_scale

    pair_indices = polynomial_pairs(standardized.shape[1])
    columns = [np.ones((standardized.shape[0], 1), dtype=np.float64), standardized]
    if POLYNOMIAL_DEGREE >= 2:
        columns.extend(
            (
                standardized[:, left_index] * standardized[:, right_index]
            ).reshape(-1, 1)
            for left_index, right_index in pair_indices
        )
    design = np.hstack(columns)
    return design, base_mean, base_scale, pair_indices


def flatten_samples(
    directory: Path = DATASET_DIR,
    screen_size_filter: tuple[int, int] | None = None,
) -> tuple[list[dict], list[dict], tuple[int, int], str]:
    train_rows: list[dict] = []
    eval_rows: list[dict] = []
    screen_size: tuple[int, int] | None = screen_size_filter
    latest_label = "manual-data"

    records = iter_capture_records(directory)
    if records:
        latest_label = records[0][0].parent.name

    for capture_file, capture in records:
        session_path = capture_file.parent / "session.json"
        session_size: tuple[int, int] | None = None
        if session_path.exists():
            session_data = json.loads(session_path.read_text())
            session_size = (
                int(session_data["screen_width"]),
                int(session_data["screen_height"]),
            )
            if screen_size is None:
                screen_size = session_size

        if screen_size is not None and session_size is not None and session_size != screen_size:
            continue

        samples = [
            sample
            for sample in capture.get("samples", [])
            if all(key in sample for key in MAPPER_FEATURE_KEYS)
        ]
        if not samples:
            continue

        if capture.get("split") == "eval":
            eval_rows.extend(samples)
        else:
            train_rows.extend(samples)

    if screen_size is None:
        screen_size = (1920, 1080)

    return train_rows, eval_rows, screen_size, latest_label


def train_mapper(
    directory: Path = DATASET_DIR,
    screen_size_filter: tuple[int, int] | None = None,
) -> PolynomialRidgeGazeMapper:
    train_rows, eval_rows, screen_size, label = flatten_samples(
        directory,
        screen_size_filter=screen_size_filter,
    )
    if len(train_rows) < len(MAPPER_FEATURE_KEYS) + 1:
        raise ValueError("Not enough compatible training samples in manual collection data")

    train_matrix = np.array(
        [[float(sample[key]) for key in MAPPER_FEATURE_KEYS] for sample in train_rows],
        dtype=np.float64,
    )
    design, base_mean, base_scale, pair_indices = build_design_matrix(train_matrix)

    target_x = np.array([float(sample["target_x"]) for sample in train_rows], dtype=np.float64)
    target_y = np.array([float(sample["target_y"]) for sample in train_rows], dtype=np.float64)
    coef_x = fit_ridge_regression(design, target_x, RIDGE_ALPHA)
    coef_y = fit_ridge_regression(design, target_y, RIDGE_ALPHA)

    mapper = PolynomialRidgeGazeMapper(
        dataset_label=label,
        feature_keys=list(MAPPER_FEATURE_KEYS),
        screen_size=screen_size,
        base_mean=base_mean,
        base_scale=base_scale,
        pair_indices=pair_indices,
        coef_x=coef_x,
        coef_y=coef_y,
        train_sample_count=len(train_rows),
        eval_sample_count=len(eval_rows),
        ridge_alpha=RIDGE_ALPHA,
        eval_mae_x_px=None,
        eval_mae_y_px=None,
    )

    if eval_rows:
        errors_x = []
        errors_y = []
        for sample in eval_rows:
            pred_x, pred_y = mapper.predict_normalized(sample)
            errors_x.append(abs(pred_x - float(sample["target_x"])))
            errors_y.append(abs(pred_y - float(sample["target_y"])))
        mapper.eval_mae_x_px = float(np.mean(errors_x) * screen_size[0])
        mapper.eval_mae_y_px = float(np.mean(errors_y) * screen_size[1])

    return mapper


def load_latest_mapper(
    directory: Path = DATASET_DIR,
    screen_size_filter: tuple[int, int] | None = None,
) -> PolynomialRidgeGazeMapper | None:
    try:
        return train_mapper(directory, screen_size_filter=screen_size_filter)
    except ValueError:
        return None
