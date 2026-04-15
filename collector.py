from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import random
import shutil
import time

from constants import (
    COLLECTION_LOCKOUT_SECONDS,
    COLLECTION_MIN_SAMPLES,
    COLLECTION_RECORD_SECONDS,
    DATASET_DIR,
    EVAL_SPLIT_RATIO,
    EYE_CROP_HEIGHT,
    EYE_CROP_WIDTH,
    STATUS_MESSAGE_SECONDS,
)
from eye_crops import save_eye_crop_pair


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def split_counts_default() -> dict[str, int]:
    return {"train": 0, "eval": 0}


def next_eval_probability(split_counts: dict[str, int], eval_ratio: float) -> float:
    total_captures = split_counts["train"] + split_counts["eval"]
    desired_eval_after_next = eval_ratio * (total_captures + 1)
    eval_gap = desired_eval_after_next - split_counts["eval"]
    return clamp01(eval_gap)


def sample_next_split(rng: random.Random, split_counts: dict[str, int], eval_ratio: float) -> str:
    probability = next_eval_probability(split_counts, eval_ratio)
    return "eval" if rng.random() < probability else "train"


def serialize_landmarks(face_landmarks) -> list[dict[str, float]]:
    return [
        {
            "x": float(landmark.x),
            "y": float(landmark.y),
            "z": float(landmark.z),
        }
        for landmark in face_landmarks
    ]


@dataclass
class CollectionState:
    screen_size: tuple[int, int]
    session_dir: Path
    session_path: Path
    captures_path: Path
    eye_crops_dir: Path
    eval_ratio: float
    split_seed: int
    rng: random.Random = field(repr=False)
    target: tuple[float, float] = (0.5, 0.5)
    recording: bool = False
    record_started_at: float = 0.0
    record_target: tuple[float, float] | None = None
    record_split: str = "train"
    record_capture_dir: Path | None = None
    next_split: str = "train"
    split_counts: dict[str, int] = field(default_factory=split_counts_default)
    pending_samples: list[dict] = field(default_factory=list)
    capture_count: int = 0
    status_message: str = ""
    status_expires_at: float = 0.0
    last_completed_at: float = 0.0


def create_collection_state(screen_size: tuple[int, int]) -> CollectionState:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    session_id = datetime.now().strftime("session-%Y%m%d-%H%M%S")
    session_dir = DATASET_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    session_path = session_dir / "session.json"
    captures_path = session_dir / "captures.jsonl"
    eye_crops_dir = session_dir / "eye_crops"
    captures_path.touch()
    eye_crops_dir.mkdir(parents=True, exist_ok=True)
    split_seed = random.SystemRandom().randrange(1, 2**63)
    rng = random.Random(split_seed)

    session_payload = {
        "session_id": session_id,
        "created_at": datetime.now().isoformat(),
        "screen_width": screen_size[0],
        "screen_height": screen_size[1],
        "record_seconds": COLLECTION_RECORD_SECONDS,
        "split_policy": "random_ratio",
        "eval_split_ratio": EVAL_SPLIT_RATIO,
        "split_seed": split_seed,
        "eye_crop_format": "png",
        "eye_crop_width": EYE_CROP_WIDTH,
        "eye_crop_height": EYE_CROP_HEIGHT,
    }
    session_path.write_text(json.dumps(session_payload, indent=2))

    state = CollectionState(
        screen_size=screen_size,
        session_dir=session_dir,
        session_path=session_path,
        captures_path=captures_path,
        eye_crops_dir=eye_crops_dir,
        eval_ratio=EVAL_SPLIT_RATIO,
        split_seed=split_seed,
        rng=rng,
    )
    state.next_split = sample_next_split(state.rng, state.split_counts, state.eval_ratio)
    return state


def set_status_message(
    state: CollectionState,
    message: str,
    duration: float = STATUS_MESSAGE_SECONDS,
) -> None:
    state.status_message = message
    state.status_expires_at = time.monotonic() + duration


def set_target_from_pixels(state: CollectionState, x: int, y: int) -> None:
    if state.recording:
        return

    width, height = state.screen_size
    state.target = (
        clamp01(x / max(width - 1, 1)),
        clamp01(y / max(height - 1, 1)),
    )


def current_target(state: CollectionState) -> tuple[float, float]:
    return state.record_target if state.recording and state.record_target else state.target


def begin_recording(state: CollectionState) -> bool:
    now = time.monotonic()
    if state.recording or now - state.last_completed_at < COLLECTION_LOCKOUT_SECONDS:
        return False

    target = state.target
    split = state.next_split

    state.recording = True
    state.record_started_at = now
    state.record_target = target
    state.record_split = split
    state.record_capture_dir = state.eye_crops_dir / f"capture-{state.capture_count + 1:05d}"
    state.pending_samples = []
    return True


def record_sample(
    state: CollectionState,
    timestamp_ms: int,
    payload: dict[str, float] | None,
    face_landmarks,
    facial_matrix,
    eye_crop_filenames: dict[str, str] | None = None,
) -> None:
    if not state.recording or payload is None or face_landmarks is None:
        return

    sample_index = len(state.pending_samples) + 1
    eye_crops = None
    if eye_crop_filenames is not None:
        eye_crops = {
            "left_path": str(Path("eye_crops") / state.record_capture_dir.name / eye_crop_filenames["left"]),
            "right_path": str(Path("eye_crops") / state.record_capture_dir.name / eye_crop_filenames["right"]),
        }

    sample_payload = {
        "timestamp_ms": timestamp_ms,
        "target_x": float(state.record_target[0]),
        "target_y": float(state.record_target[1]),
        "sample_index": sample_index,
        **payload,
        "raw_landmarks": serialize_landmarks(face_landmarks),
        "facial_transformation_matrix": (
            [[float(value) for value in row] for row in facial_matrix]
            if facial_matrix is not None
            else None
        ),
    }
    if eye_crops is not None:
        sample_payload.update(eye_crops)

    state.pending_samples.append(sample_payload)


def save_sample_eye_crops(
    state: CollectionState,
    timestamp_ms: int,
    crops: dict[str, object] | None,
) -> dict[str, str] | None:
    if not state.recording or crops is None or state.record_capture_dir is None:
        return None
    sample_index = len(state.pending_samples) + 1
    return save_eye_crop_pair(
        state.record_capture_dir,
        sample_index,
        timestamp_ms,
        crops,
    )


def finalize_recording(state: CollectionState) -> Path | None:
    capture = {
        "captured_at": datetime.now().isoformat(),
        "duration_seconds": COLLECTION_RECORD_SECONDS,
        "split": state.record_split,
        "target_x": float(state.record_target[0]),
        "target_y": float(state.record_target[1]),
        "eye_crop_dir": (
            str(Path("eye_crops") / state.record_capture_dir.name)
            if state.record_capture_dir is not None
            else None
        ),
        "sample_count": len(state.pending_samples),
        "samples": state.pending_samples,
    }

    state.recording = False
    state.record_target = None
    state.last_completed_at = time.monotonic()

    if len(state.pending_samples) < COLLECTION_MIN_SAMPLES:
        state.pending_samples = []
        if state.record_capture_dir is not None:
            shutil.rmtree(state.record_capture_dir, ignore_errors=True)
        state.record_capture_dir = None
        state.next_split = sample_next_split(state.rng, state.split_counts, state.eval_ratio)
        set_status_message(state, "Capture discarded: not enough valid samples")
        return None

    with state.captures_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(capture))
        handle.write("\n")

    state.capture_count += 1
    state.split_counts[capture["split"]] += 1
    state.next_split = sample_next_split(state.rng, state.split_counts, state.eval_ratio)
    state.pending_samples = []
    state.record_capture_dir = None
    set_status_message(
        state,
        f"Saved {capture['split']} capture #{state.capture_count} ({capture['sample_count']} samples)",
    )
    return state.captures_path


def cleanup_empty_session(state: CollectionState) -> None:
    if state.capture_count > 0:
        return
    shutil.rmtree(state.session_dir, ignore_errors=True)


def update_recording(state: CollectionState) -> Path | None:
    if not state.recording:
        return None
    if time.monotonic() - state.record_started_at < COLLECTION_RECORD_SECONDS:
        return None
    return finalize_recording(state)
