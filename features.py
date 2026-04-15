from dataclasses import dataclass

import numpy as np

from constants import (
    LEFT_EYE_CORNER_POINTS,
    LEFT_IRIS_POINTS,
    LEFT_EYE_LOWER_LID_POINTS,
    LEFT_EYE_UPPER_LID_POINTS,
    RIGHT_IRIS_POINTS,
    RIGHT_EYE_CORNER_POINTS,
    RIGHT_EYE_LOWER_LID_POINTS,
    RIGHT_EYE_UPPER_LID_POINTS,
)


@dataclass
class FeatureFrame:
    face_landmarks: list
    left_eye: dict[str, float]
    right_eye: dict[str, float]
    face_feature: dict[str, float]
    head_pose: dict[str, float] | None
    payload: dict[str, float]
    avg_x: float
    avg_y: float


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def to_pixel(landmark, width: int, height: int) -> tuple[int, int]:
    x = int(np.clip(landmark.x * width, 0, width - 1))
    y = int(np.clip(landmark.y * height, 0, height - 1))
    return x, y


def average_point(face_landmarks, point_indices: list[int]) -> tuple[float, float]:
    x = sum(face_landmarks[index].x for index in point_indices) / len(point_indices)
    y = sum(face_landmarks[index].y for index in point_indices) / len(point_indices)
    return x, y


def point_array(face_landmarks, point_index: int) -> np.ndarray:
    landmark = face_landmarks[point_index]
    return np.array([landmark.x, landmark.y], dtype=np.float64)


def mean_point_array(face_landmarks, point_indices: list[int]) -> np.ndarray:
    return np.mean(
        [point_array(face_landmarks, point_index) for point_index in point_indices],
        axis=0,
    )


def compute_eye_feature(
    face_landmarks,
    corner_indices: tuple[int, int],
    upper_lid_indices: list[int],
    lower_lid_indices: list[int],
    iris_points: list[int],
) -> dict[str, float]:
    first_corner = point_array(face_landmarks, corner_indices[0])
    second_corner = point_array(face_landmarks, corner_indices[1])
    left_corner, right_corner = sorted(
        (first_corner, second_corner), key=lambda point: point[0]
    )

    iris_center = mean_point_array(face_landmarks, iris_points)
    upper_lid = mean_point_array(face_landmarks, upper_lid_indices)
    lower_lid = mean_point_array(face_landmarks, lower_lid_indices)

    horizontal_axis = right_corner - left_corner
    eye_width = max(float(np.linalg.norm(horizontal_axis)), 1e-6)
    horizontal_unit = horizontal_axis / eye_width

    orthogonal_unit = np.array([-horizontal_unit[1], horizontal_unit[0]])
    if float(np.dot(lower_lid - upper_lid, orthogonal_unit)) < 0.0:
        orthogonal_unit *= -1.0

    vertical_extent = max(float(np.dot(lower_lid - upper_lid, orthogonal_unit)), 1e-6)
    eye_center = (left_corner + right_corner + upper_lid + lower_lid) / 4.0

    x_projection = float(np.dot(iris_center - left_corner, horizontal_unit)) / eye_width
    y_projection = float(np.dot(iris_center - upper_lid, orthogonal_unit)) / vertical_extent
    orthogonal_offset = float(np.dot(iris_center - eye_center, orthogonal_unit)) / eye_width
    upper_gap = float(np.linalg.norm(iris_center - upper_lid)) / eye_width
    lower_gap = float(np.linalg.norm(lower_lid - iris_center)) / eye_width

    return {
        "iris_x": float(iris_center[0]),
        "iris_y": float(iris_center[1]),
        "x_ratio": clamp01(x_projection),
        "y_ratio": clamp01(y_projection),
        "orth_y": orthogonal_offset,
        "upper_gap": upper_gap,
        "lower_gap": lower_gap,
        "eye_width": eye_width,
        "eye_height": vertical_extent,
        "eye_openness": vertical_extent / eye_width,
    }


def compute_face_feature(face_landmarks) -> dict[str, float]:
    x_coords = [landmark.x for landmark in face_landmarks]
    y_coords = [landmark.y for landmark in face_landmarks]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    width = max(max_x - min_x, 1e-6)
    height = max(max_y - min_y, 1e-6)

    return {
        "center_x": (min_x + max_x) / 2.0,
        "center_y": (min_y + max_y) / 2.0,
        "width": width,
        "height": height,
        "scale": (width + height) / 2.0,
    }


def compute_head_pose(matrix: np.ndarray) -> dict[str, float]:
    rotation = np.array(matrix[:3, :3], dtype=np.float64)
    u, _, vh = np.linalg.svd(rotation)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh

    sy = np.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(-rotation[2, 0], sy)
        roll = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        pitch = np.arctan2(-rotation[1, 2], rotation[1, 1])
        yaw = np.arctan2(-rotation[2, 0], sy)
        roll = 0.0

    translation = np.array(matrix[:3, 3], dtype=np.float64)
    return {
        "yaw_deg": float(np.degrees(yaw)),
        "pitch_deg": float(np.degrees(pitch)),
        "roll_deg": float(np.degrees(roll)),
        "tx": float(translation[0]),
        "ty": float(translation[1]),
        "tz": float(translation[2]),
    }


def build_feature_payload(
    left_eye: dict[str, float],
    right_eye: dict[str, float],
    face_feature: dict[str, float],
    head_pose: dict[str, float] | None,
) -> dict[str, float]:
    payload = {
        "left_x": left_eye["x_ratio"],
        "left_y": left_eye["y_ratio"],
        "left_orth_y": left_eye["orth_y"],
        "left_openness": left_eye["eye_openness"],
        "left_upper_gap": left_eye["upper_gap"],
        "left_lower_gap": left_eye["lower_gap"],
        "right_x": right_eye["x_ratio"],
        "right_y": right_eye["y_ratio"],
        "right_orth_y": right_eye["orth_y"],
        "right_openness": right_eye["eye_openness"],
        "right_upper_gap": right_eye["upper_gap"],
        "right_lower_gap": right_eye["lower_gap"],
        "avg_x": (left_eye["x_ratio"] + right_eye["x_ratio"]) / 2.0,
        "avg_y": (left_eye["y_ratio"] + right_eye["y_ratio"]) / 2.0,
        "face_center_x": face_feature["center_x"],
        "face_center_y": face_feature["center_y"],
        "face_width": face_feature["width"],
        "face_height": face_feature["height"],
        "face_scale": face_feature["scale"],
    }
    if head_pose is not None:
        payload.update(
            {
                "head_yaw_deg": head_pose["yaw_deg"],
                "head_pitch_deg": head_pose["pitch_deg"],
                "head_roll_deg": head_pose["roll_deg"],
                "head_tx": head_pose["tx"],
                "head_ty": head_pose["ty"],
                "head_tz": head_pose["tz"],
            }
        )
    return payload


def extract_feature_frame(result) -> FeatureFrame | None:
    if not result or not result.face_landmarks:
        return None

    face_landmarks = result.face_landmarks[0]
    left_eye = compute_eye_feature(
        face_landmarks,
        LEFT_EYE_CORNER_POINTS,
        LEFT_EYE_UPPER_LID_POINTS,
        LEFT_EYE_LOWER_LID_POINTS,
        LEFT_IRIS_POINTS,
    )
    right_eye = compute_eye_feature(
        face_landmarks,
        RIGHT_EYE_CORNER_POINTS,
        RIGHT_EYE_UPPER_LID_POINTS,
        RIGHT_EYE_LOWER_LID_POINTS,
        RIGHT_IRIS_POINTS,
    )
    face_feature = compute_face_feature(face_landmarks)
    head_pose = None
    if result.facial_transformation_matrixes:
        head_pose = compute_head_pose(result.facial_transformation_matrixes[0])
    payload = build_feature_payload(left_eye, right_eye, face_feature, head_pose)
    avg_x = (left_eye["x_ratio"] + right_eye["x_ratio"]) / 2.0
    avg_y = (left_eye["y_ratio"] + right_eye["y_ratio"]) / 2.0

    return FeatureFrame(
        face_landmarks=face_landmarks,
        left_eye=left_eye,
        right_eye=right_eye,
        face_feature=face_feature,
        head_pose=head_pose,
        payload=payload,
        avg_x=avg_x,
        avg_y=avg_y,
    )
