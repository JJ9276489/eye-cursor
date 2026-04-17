from collections import deque

import cv2
import numpy as np

from constants import (
    FACE_OVAL,
    GAZE_PREVIEW_WINDOW_NAME,
    LEFT_EYE,
    LEFT_IRIS,
    LEFT_IRIS_POINTS,
    RIGHT_EYE,
    RIGHT_IRIS,
    RIGHT_IRIS_POINTS,
)
from features import FeatureFrame, average_point, clamp01, to_pixel


def draw_connections(
    frame: np.ndarray,
    face_landmarks,
    connections: list[tuple[int, int]],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    height, width = frame.shape[:2]
    for start_idx, end_idx in connections:
        start = to_pixel(face_landmarks[start_idx], width, height)
        end = to_pixel(face_landmarks[end_idx], width, height)
        cv2.line(frame, start, end, color, thickness, cv2.LINE_AA)


def draw_points(
    frame: np.ndarray,
    face_landmarks,
    point_indices: list[int],
    color: tuple[int, int, int],
    radius: int,
) -> None:
    height, width = frame.shape[:2]
    for point_idx in point_indices:
        point = to_pixel(face_landmarks[point_idx], width, height)
        cv2.circle(frame, point, radius, color, -1, cv2.LINE_AA)


def draw_overlay(frame: np.ndarray, face_landmarks) -> None:
    draw_connections(frame, face_landmarks, FACE_OVAL, (180, 180, 180), 1)
    draw_connections(frame, face_landmarks, LEFT_EYE, (0, 220, 0), 1)
    draw_connections(frame, face_landmarks, RIGHT_EYE, (0, 220, 0), 1)
    draw_connections(frame, face_landmarks, LEFT_IRIS, (0, 200, 255), 2)
    draw_connections(frame, face_landmarks, RIGHT_IRIS, (0, 200, 255), 2)
    draw_points(frame, face_landmarks, LEFT_IRIS_POINTS, (0, 255, 255), 2)
    draw_points(frame, face_landmarks, RIGHT_IRIS_POINTS, (0, 255, 255), 2)

    height, width = frame.shape[:2]
    left_iris = average_point(face_landmarks, LEFT_IRIS_POINTS)
    right_iris = average_point(face_landmarks, RIGHT_IRIS_POINTS)

    cv2.circle(
        frame,
        (int(left_iris[0] * width), int(left_iris[1] * height)),
        4,
        (255, 255, 0),
        -1,
        cv2.LINE_AA,
    )
    cv2.circle(
        frame,
        (int(right_iris[0] * width), int(right_iris[1] * height)),
        4,
        (255, 255, 0),
        -1,
        cv2.LINE_AA,
    )


def draw_panel_background(
    frame: np.ndarray, top_left: tuple[int, int], bottom_right: tuple[int, int]
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, top_left, bottom_right, (70, 70, 70), 1)


def draw_gaze_panel(
    frame: np.ndarray,
    feature_frame: FeatureFrame,
    history: deque[tuple[float, float]],
    paused: bool,
    model_label: str | None,
    model_metrics: dict[str, float] | None,
    predicted_point: tuple[float, float] | None,
    screen_size: tuple[int, int],
    preview_enabled: bool,
    active_mode: str | None,
) -> None:
    panel_width = 320
    panel_height = 410
    margin = 16
    panel_x = frame.shape[1] - panel_width - margin
    panel_y = margin
    draw_panel_background(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_width, panel_y + panel_height),
    )

    status = "PAUSED" if paused else "LIVE"
    cv2.putText(
        frame,
        f"Gaze Signal {status}",
        (panel_x + 12, panel_y + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    graph_x = panel_x + 12
    graph_y = panel_y + 40
    graph_size = 110
    cv2.rectangle(
        frame,
        (graph_x, graph_y),
        (graph_x + graph_size, graph_y + graph_size),
        (90, 90, 90),
        1,
    )
    cv2.line(
        frame,
        (graph_x + graph_size // 2, graph_y),
        (graph_x + graph_size // 2, graph_y + graph_size),
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )
    cv2.line(
        frame,
        (graph_x, graph_y + graph_size // 2),
        (graph_x + graph_size, graph_y + graph_size // 2),
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )

    history_points = list(history)
    for index, (hist_x, hist_y) in enumerate(history_points):
        alpha = (index + 1) / len(history_points)
        color = (
            int(40 + 120 * alpha),
            int(150 + 80 * alpha),
            int(255 * alpha),
        )
        point = (
            graph_x + int(clamp01(hist_x) * graph_size),
            graph_y + int(clamp01(hist_y) * graph_size),
        )
        cv2.circle(frame, point, 2, color, -1, cv2.LINE_AA)

    current_point = (
        graph_x + int(feature_frame.avg_x * graph_size),
        graph_y + int(feature_frame.avg_y * graph_size),
    )
    cv2.circle(frame, current_point, 5, (0, 255, 255), -1, cv2.LINE_AA)

    text_x = graph_x + graph_size + 16
    cv2.putText(
        frame,
        f"avg x {feature_frame.avg_x:.3f}",
        (text_x, graph_y + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"avg y {feature_frame.avg_y:.3f}",
        (text_x, graph_y + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"L {feature_frame.left_eye['x_ratio']:.3f} {feature_frame.left_eye['y_ratio']:.3f}",
        (text_x, graph_y + 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (180, 255, 180),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"R {feature_frame.right_eye['x_ratio']:.3f} {feature_frame.right_eye['y_ratio']:.3f}",
        (text_x, graph_y + 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (180, 255, 180),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"face {feature_frame.face_feature['center_x']:.3f} {feature_frame.face_feature['center_y']:.3f}",
        (panel_x + 12, panel_y + 172),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (200, 220, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"scale {feature_frame.face_feature['scale']:.3f}",
        (panel_x + 12, panel_y + 194),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (200, 220, 255),
        1,
        cv2.LINE_AA,
    )

    if feature_frame.head_pose is not None:
        cv2.putText(
            frame,
            f"yaw {feature_frame.head_pose['yaw_deg']:+5.1f}",
            (panel_x + 12, panel_y + 224),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 220, 180),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"pitch {feature_frame.head_pose['pitch_deg']:+5.1f}",
            (panel_x + 12, panel_y + 246),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 220, 180),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"roll {feature_frame.head_pose['roll_deg']:+5.1f}",
            (panel_x + 12, panel_y + 268),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 220, 180),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame,
            "head pose unavailable",
            (panel_x + 12, panel_y + 246),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (170, 170, 170),
            1,
            cv2.LINE_AA,
        )

    model_text = model_label if model_label else "model unavailable"
    if len(model_text) > 28:
        model_text = model_text[:25] + "..."
    mode_text = active_mode if active_mode else "none"
    cv2.putText(
        frame,
        f"model {mode_text} | {model_text}",
        (panel_x + 12, panel_y + 296),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    preview_x = panel_x + 12
    preview_y = panel_y + 314
    preview_width = 180
    preview_height = 52
    cv2.rectangle(
        frame,
        (preview_x, preview_y),
        (preview_x + preview_width, preview_y + preview_height),
        (90, 90, 90),
        1,
    )
    if predicted_point is not None:
        pred_px = preview_x + int(predicted_point[0] * preview_width)
        pred_py = preview_y + int(predicted_point[1] * preview_height)
        cv2.circle(frame, (pred_px, pred_py), 5, (0, 255, 255), -1, cv2.LINE_AA)
        actual_x = int(predicted_point[0] * (screen_size[0] - 1))
        actual_y = int(predicted_point[1] * (screen_size[1] - 1))
        cv2.putText(
            frame,
            f"pred {actual_x:4d},{actual_y:4d}",
            (panel_x + 202, preview_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame,
            "pred unavailable",
            (panel_x + 202, preview_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

    preview_label = f"preview {'on' if preview_enabled else 'off'}"
    cv2.putText(
        frame,
        preview_label,
        (panel_x + 202, preview_y + 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )

    if model_metrics:
        cv2.putText(
            frame,
            f"eval {model_metrics['eval_mae_x_px']:.0f}px x  {model_metrics['eval_mae_y_px']:.0f}px y",
            (panel_x + 12, panel_y + 374),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame,
            "eval unavailable",
            (panel_x + 12, panel_y + 374),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (170, 170, 170),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        "space record  c clear  g preview  v model",
        (panel_x + 12, panel_y + panel_height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )


def draw_notification(frame: np.ndarray, message: str) -> None:
    draw_panel_background(frame, (16, 48), (16 + 420, 92))
    cv2.putText(
        frame,
        message,
        (28, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_status(frame: np.ndarray, status: str) -> None:
    cv2.putText(
        frame,
        status,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def open_gaze_preview_window() -> None:
    cv2.namedWindow(GAZE_PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        GAZE_PREVIEW_WINDOW_NAME,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )


def close_gaze_preview_window() -> None:
    try:
        cv2.destroyWindow(GAZE_PREVIEW_WINDOW_NAME)
    except cv2.error:
        pass


def preview_marker_color(index: int, active: bool) -> tuple[int, int, int]:
    if active:
        return (0, 255, 255)
    palette = [
        (80, 220, 80),
        (255, 170, 80),
        (220, 120, 255),
        (255, 210, 80),
        (120, 220, 255),
    ]
    return palette[index % len(palette)]


def draw_preview_marker(
    canvas: np.ndarray,
    point: tuple[float, float],
    color: tuple[int, int, int],
    active: bool,
) -> tuple[int, int]:
    height, width = canvas.shape[:2]
    point_x = int(clamp01(point[0]) * (width - 1))
    point_y = int(clamp01(point[1]) * (height - 1))
    radius = 24 if active else 16
    cv2.circle(canvas, (point_x, point_y), radius, color, 2, cv2.LINE_AA)
    cv2.circle(canvas, (point_x, point_y), 6 if active else 4, color, -1, cv2.LINE_AA)
    cv2.line(
        canvas,
        (point_x - radius - 10, point_y),
        (point_x + radius + 10, point_y),
        color,
        1,
        cv2.LINE_AA,
    )
    cv2.line(
        canvas,
        (point_x, point_y - radius - 10),
        (point_x, point_y + radius + 10),
        color,
        1,
        cv2.LINE_AA,
    )
    return point_x, point_y


def render_gaze_preview(
    screen_size: tuple[int, int],
    predicted_point: tuple[float, float] | None,
    model_label: str | None,
    active_mode: str | None,
    model_predictions: list[tuple[str, str | None, tuple[float, float], bool]] | None = None,
) -> None:
    width, height = screen_size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    label = model_label if model_label else "model unavailable"
    mode_text = active_mode if active_mode else "none"
    predictions = model_predictions or []

    active_point = predicted_point
    if predictions:
        for index, (mode_name, item_label, point, active) in enumerate(predictions):
            color = preview_marker_color(index, active)
            point_x, point_y = draw_preview_marker(canvas, point, color, active)
            if active:
                active_point = point
            if active or index < 5:
                text_y = max(24, point_y - 18)
                short_name = mode_name if len(mode_name) <= 22 else mode_name[:19] + "..."
                cv2.putText(
                    canvas,
                    short_name,
                    (min(point_x + 20, width - 260), text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2 if active else 1,
                    cv2.LINE_AA,
                )
    elif predicted_point is not None:
        draw_preview_marker(canvas, predicted_point, (0, 255, 255), True)

    if active_point is not None:
        point_x = int(clamp01(active_point[0]) * (width - 1))
        point_y = int(clamp01(active_point[1]) * (height - 1))
        cv2.putText(
            canvas,
            f"active gaze {point_x}, {point_y}",
            (60, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            canvas,
            "No prediction available",
            (60, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    legend_y = 185
    if predictions:
        cv2.putText(
            canvas,
            "Loaded models",
            (60, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        for index, (mode_name, item_label, point, active) in enumerate(predictions[:8]):
            color = preview_marker_color(index, active)
            y = legend_y + 36 + index * 30
            cv2.circle(canvas, (72, y - 5), 7, color, -1, cv2.LINE_AA)
            display_label = item_label or mode_name
            if len(display_label) > 34:
                display_label = display_label[:31] + "..."
            suffix = " active" if active else ""
            cv2.putText(
                canvas,
                f"{mode_name}: {display_label}{suffix}",
                (92, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                color if active else (210, 210, 210),
                2 if active else 1,
                cv2.LINE_AA,
            )

    cv2.putText(
        canvas,
        "Gaze Preview",
        (60, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"model {mode_text} | {label}",
        (60, height - 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (180, 180, 180),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Press g to close preview | v to switch model",
        (60, height - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (180, 180, 180),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow(GAZE_PREVIEW_WINDOW_NAME, canvas)
