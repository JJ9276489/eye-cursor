import argparse
from collections import deque
import time

import cv2
import mediapipe as mp

from camera import open_camera
from collection_window import CollectionWindow
from collector import (
    begin_recording,
    cleanup_empty_session,
    create_collection_state,
    current_target,
    record_sample,
    save_sample_eye_crops,
    set_status_message,
    update_recording,
)
from constants import DEBUG_WINDOW_NAME, PREDICTION_ALPHA
from eye_crops import extract_eye_crops
from features import extract_feature_frame
from landmarker import create_landmarker, ensure_model
from runtime_models import RuntimeModels
from ui import (
    close_gaze_preview_window,
    draw_gaze_panel,
    draw_notification,
    draw_overlay,
    draw_status,
    open_gaze_preview_window,
    render_gaze_preview,
)


def smooth_prediction(
    previous: tuple[float, float] | None,
    current: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if current is None:
        return None
    if previous is None:
        return current
    return (
        previous[0] * (1.0 - PREDICTION_ALPHA) + current[0] * PREDICTION_ALPHA,
        previous[1] * (1.0 - PREDICTION_ALPHA) + current[1] * PREDICTION_ALPHA,
    )


def start_recording(collection, screen_size: tuple[int, int]) -> bool:
    if not begin_recording(collection):
        return False
    target = current_target(collection)
    set_status_message(
        collection,
        (
            f"Recording {collection.record_split} capture at "
            f"{int(target[0] * (screen_size[0] - 1))}, "
            f"{int(target[1] * (screen_size[1] - 1))}"
        ),
        duration=1.2,
    )
    return True


def run(start_preview: bool = False, collection_visible: bool = True) -> None:
    model_path = ensure_model()
    cap = open_camera()
    collection_window = CollectionWindow()
    screen_size = collection_window.screen_size
    if not collection_visible:
        collection_window.toggle_visibility()

    history: deque[tuple[float, float]] = deque(maxlen=90)
    collection = create_collection_state(screen_size=screen_size)
    runtime_models = RuntimeModels(screen_size=screen_size)
    runtime_models.load_initial(collection)
    if runtime_models.active_mode and runtime_models.active_mode.startswith("vision"):
        runtime_models.announce_active_mode(collection, include_compare_hint=True)

    preview_enabled = start_preview
    last_frame = None
    last_result = None
    last_timestamp_ms = 0
    smoothed_prediction: tuple[float, float] | None = None
    smoothed_model_predictions: dict[str, tuple[float, float]] = {}
    preview_model_predictions: list[tuple[str, str | None, tuple[float, float], bool]] = []
    if preview_enabled:
        open_gaze_preview_window()

    try:
        with create_landmarker(model_path) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Failed to read a frame from the webcam")

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(time.monotonic() * 1000)
                last_result = landmarker.detect_for_video(mp_image, timestamp_ms)
                last_frame = frame
                last_timestamp_ms = timestamp_ms

                frame = last_frame.copy()
                feature_frame = extract_feature_frame(last_result)

                if feature_frame is not None:
                    history.append((feature_frame.avg_x, feature_frame.avg_y))

                    eye_crops = None
                    if collection.recording or (
                        runtime_models.active_mode is not None
                        and runtime_models.active_mode.startswith("vision")
                    ) or (
                        preview_enabled
                        and not collection.recording
                        and runtime_models.has_vision_predictors()
                    ):
                        eye_crops = extract_eye_crops(last_frame, feature_frame.face_landmarks)

                    raw_prediction = runtime_models.predict_normalized(
                        feature_frame.payload,
                        eye_crops,
                    )
                    smoothed_prediction = smooth_prediction(
                        smoothed_prediction,
                        raw_prediction,
                    )

                    if preview_enabled and not collection.recording:
                        raw_model_predictions = runtime_models.predict_all_normalized(
                            feature_frame.payload,
                            eye_crops,
                        )
                        smoothed_model_predictions = {
                            mode: smooth_prediction(
                                smoothed_model_predictions.get(mode),
                                point,
                            )
                            for mode, point in raw_model_predictions.items()
                        }
                        preview_model_predictions = []
                        for mode in runtime_models.available_modes():
                            point = smoothed_model_predictions.get(mode)
                            if point is None:
                                continue
                            mode_label = runtime_models.mode_display_name(mode) or mode
                            checkpoint_label, _ = runtime_models.mode_details(mode)
                            preview_model_predictions.append(
                                (
                                    mode_label,
                                    checkpoint_label,
                                    point,
                                    mode == runtime_models.active_mode,
                                )
                            )
                    else:
                        smoothed_model_predictions.clear()
                        preview_model_predictions = []

                    facial_matrix = (
                        last_result.facial_transformation_matrixes[0]
                        if last_result and last_result.facial_transformation_matrixes
                        else None
                    )
                    eye_crop_filenames = None
                    if collection.recording:
                        eye_crop_filenames = save_sample_eye_crops(
                            collection,
                            last_timestamp_ms,
                            eye_crops,
                        )
                    record_sample(
                        collection,
                        last_timestamp_ms,
                        feature_frame.payload,
                        feature_frame.face_landmarks,
                        facial_matrix,
                        eye_crop_filenames=eye_crop_filenames,
                    )

                    active_label, active_metrics = runtime_models.active_details()
                    draw_overlay(frame, feature_frame.face_landmarks)
                    draw_gaze_panel(
                        frame,
                        feature_frame,
                        history,
                        paused=False,
                        model_label=active_label,
                        model_metrics=active_metrics,
                        predicted_point=(
                            smoothed_prediction if preview_enabled and not collection.recording else None
                        ),
                        screen_size=screen_size,
                        preview_enabled=preview_enabled,
                        active_mode=runtime_models.active_mode,
                    )
                    draw_status(frame, "Tracking face and irises")
                else:
                    smoothed_prediction = None
                    smoothed_model_predictions.clear()
                    preview_model_predictions = []
                    draw_status(frame, "No face detected")

                if update_recording(collection) is not None:
                    runtime_models.reload_ridge(collection)

                collection_window.render(collection)

                active_label, _ = runtime_models.active_details()
                if preview_enabled and not collection.recording:
                    render_gaze_preview(
                        screen_size,
                        smoothed_prediction,
                        active_label,
                        runtime_models.active_mode,
                        preview_model_predictions,
                    )

                if collection.status_message:
                    if time.monotonic() < collection.status_expires_at:
                        draw_notification(frame, collection.status_message)
                    else:
                        collection.status_message = ""

                cv2.imshow(DEBUG_WINDOW_NAME, frame)
                debug_key = cv2.waitKey(1) & 0xFF

                events = collection_window.poll_events(collection)
                if events.quit_requested:
                    break
                if debug_key in (27, ord("q")):
                    break
                if events.clear_history:
                    history.clear()
                if debug_key == ord("c"):
                    history.clear()
                if events.begin_recording:
                    if start_recording(collection, screen_size):
                        preview_enabled = False
                        smoothed_model_predictions.clear()
                        preview_model_predictions = []
                        close_gaze_preview_window()
                if debug_key == ord(" ") and not events.begin_recording:
                    if start_recording(collection, screen_size):
                        preview_enabled = False
                        smoothed_model_predictions.clear()
                        preview_model_predictions = []
                        close_gaze_preview_window()
                if events.toggle_preview and not collection.recording:
                    preview_enabled = not preview_enabled
                    smoothed_model_predictions.clear()
                    preview_model_predictions = []
                    if preview_enabled:
                        open_gaze_preview_window()
                    else:
                        close_gaze_preview_window()
                if debug_key == ord("g") and not collection.recording and not events.toggle_preview:
                    preview_enabled = not preview_enabled
                    smoothed_model_predictions.clear()
                    preview_model_predictions = []
                    if preview_enabled:
                        open_gaze_preview_window()
                    else:
                        close_gaze_preview_window()
                if events.toggle_model or (debug_key == ord("v") and not events.toggle_model):
                    next_mode = runtime_models.cycle_active_mode(collection)
                    if next_mode is not None:
                        smoothed_prediction = None
                        smoothed_model_predictions.clear()
                        preview_model_predictions = []
                if events.toggle_collection or debug_key == ord("m"):
                    collection_window.toggle_visibility()
                    set_status_message(
                        collection,
                        "Collection screen shown" if collection_window.visible else "Collection screen hidden",
                        duration=1.2,
                    )
    finally:
        close_gaze_preview_window()
        collection_window.close()
        cleanup_empty_session(collection)
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the webcam gaze cursor prototype.")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Start with the fullscreen gaze preview enabled.",
    )
    parser.add_argument(
        "--hide-collection",
        action="store_true",
        help="Start with the fullscreen collection target hidden.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(start_preview=args.preview, collection_visible=not args.hide_collection)
