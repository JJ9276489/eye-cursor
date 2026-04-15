from pathlib import Path

import cv2
import numpy as np

from constants import (
    EYE_CROP_HEIGHT,
    EYE_CROP_WIDTH,
    LEFT_EYE_CORNER_POINTS,
    LEFT_EYE_LOWER_LID_POINTS,
    LEFT_EYE_UPPER_LID_POINTS,
    RIGHT_EYE_CORNER_POINTS,
    RIGHT_EYE_LOWER_LID_POINTS,
    RIGHT_EYE_UPPER_LID_POINTS,
)
from features import mean_point_array, point_array


def _normalized_to_pixel(point: np.ndarray, frame_shape: tuple[int, int, int]) -> np.ndarray:
    height, width = frame_shape[:2]
    return np.array(
        [
            float(np.clip(point[0] * width, 0, width - 1)),
            float(np.clip(point[1] * height, 0, height - 1)),
        ],
        dtype=np.float32,
    )


def _eye_axes(
    face_landmarks,
    corner_indices: tuple[int, int],
    upper_lid_indices: list[int],
    lower_lid_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    first_corner = point_array(face_landmarks, corner_indices[0])
    second_corner = point_array(face_landmarks, corner_indices[1])
    left_corner, right_corner = sorted(
        (first_corner, second_corner), key=lambda point: point[0]
    )

    upper_lid = mean_point_array(face_landmarks, upper_lid_indices)
    lower_lid = mean_point_array(face_landmarks, lower_lid_indices)

    horizontal_axis = right_corner - left_corner
    eye_width = max(float(np.linalg.norm(horizontal_axis)), 1e-6)
    horizontal_unit = horizontal_axis / eye_width

    vertical_unit = np.array([-horizontal_unit[1], horizontal_unit[0]], dtype=np.float64)
    if float(np.dot(lower_lid - upper_lid, vertical_unit)) < 0.0:
        vertical_unit *= -1.0

    eye_height = max(float(np.dot(lower_lid - upper_lid, vertical_unit)), 1e-6)
    center = (left_corner + right_corner + upper_lid + lower_lid) / 4.0
    return center, horizontal_unit, vertical_unit, eye_width, eye_height


def _extract_eye_crop(
    frame_bgr: np.ndarray,
    face_landmarks,
    corner_indices: tuple[int, int],
    upper_lid_indices: list[int],
    lower_lid_indices: list[int],
    flip_horizontal: bool,
) -> np.ndarray:
    center, horizontal_unit, vertical_unit, eye_width, eye_height = _eye_axes(
        face_landmarks,
        corner_indices,
        upper_lid_indices,
        lower_lid_indices,
    )
    center_px = _normalized_to_pixel(center, frame_bgr.shape)

    crop_width_px = eye_width * frame_bgr.shape[1] * 1.8
    crop_height_px = max(
        eye_height * frame_bgr.shape[0] * 3.2,
        crop_width_px * (EYE_CROP_HEIGHT / EYE_CROP_WIDTH),
    )

    x_axis = horizontal_unit.astype(np.float32)
    y_axis = vertical_unit.astype(np.float32)
    x_axis_px = x_axis * (crop_width_px / 2.0)
    y_axis_px = y_axis * (crop_height_px / 2.0)

    source_points = np.array(
        [
            center_px - x_axis_px - y_axis_px,
            center_px + x_axis_px - y_axis_px,
            center_px - x_axis_px + y_axis_px,
        ],
        dtype=np.float32,
    )
    destination_points = np.array(
        [
            [0.0, 0.0],
            [float(EYE_CROP_WIDTH - 1), 0.0],
            [0.0, float(EYE_CROP_HEIGHT - 1)],
        ],
        dtype=np.float32,
    )
    transform = cv2.getAffineTransform(source_points, destination_points)
    crop = cv2.warpAffine(
        frame_bgr,
        transform,
        (EYE_CROP_WIDTH, EYE_CROP_HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if flip_horizontal:
        crop = cv2.flip(crop, 1)
    return crop


def extract_eye_crops(frame_bgr: np.ndarray, face_landmarks) -> dict[str, np.ndarray]:
    return {
        "left": _extract_eye_crop(
            frame_bgr,
            face_landmarks,
            LEFT_EYE_CORNER_POINTS,
            LEFT_EYE_UPPER_LID_POINTS,
            LEFT_EYE_LOWER_LID_POINTS,
            flip_horizontal=False,
        ),
        "right": _extract_eye_crop(
            frame_bgr,
            face_landmarks,
            RIGHT_EYE_CORNER_POINTS,
            RIGHT_EYE_UPPER_LID_POINTS,
            RIGHT_EYE_LOWER_LID_POINTS,
            flip_horizontal=True,
        ),
    }


def save_eye_crop_pair(
    capture_dir: Path,
    sample_index: int,
    timestamp_ms: int,
    crops: dict[str, np.ndarray],
) -> dict[str, str]:
    capture_dir.mkdir(parents=True, exist_ok=True)

    left_name = f"sample-{sample_index:04d}-{timestamp_ms}-left.png"
    right_name = f"sample-{sample_index:04d}-{timestamp_ms}-right.png"
    left_path = capture_dir / left_name
    right_path = capture_dir / right_name

    if not cv2.imwrite(str(left_path), crops["left"]):
        raise RuntimeError(f"Failed to write left eye crop to {left_path}")
    if not cv2.imwrite(str(right_path), crops["right"]):
        raise RuntimeError(f"Failed to write right eye crop to {right_path}")

    return {
        "left": left_name,
        "right": right_name,
    }
