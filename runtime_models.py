from dataclasses import dataclass, field
from pathlib import Path

from collector import set_status_message
from constants import (
    VISION_ATTENTION_MATCHED_MODEL_PATH,
    VISION_CLIFFORD_MODEL_PATH,
    VISION_CONCAT_MODEL_PATH,
    VISION_SPATIAL_GEOMETRY_MODEL_PATH,
    VISION_SPATIAL_MODEL_PATH,
)
from mapper import PolynomialRidgeGazeMapper, load_latest_mapper
from vision_runtime import VisionGazePredictor, load_vision_predictor

VisionCheckpointSpec = tuple[str, str, Path]


def format_metrics(metrics: dict[str, float] | None) -> str | None:
    if metrics is None:
        return None
    return (
        f"eval {metrics['eval_mae_x_px']:.0f}px x / "
        f"{metrics['eval_mae_y_px']:.0f}px y"
    )


def vision_checkpoints() -> list[VisionCheckpointSpec]:
    return [
        ("vision_spatial_geom", "vision-spatial-geom", VISION_SPATIAL_GEOMETRY_MODEL_PATH),
        ("vision_spatial", "vision-spatial", VISION_SPATIAL_MODEL_PATH),
        ("vision_clifford", "vision-clifford", VISION_CLIFFORD_MODEL_PATH),
        ("vision_concat", "vision-concat", VISION_CONCAT_MODEL_PATH),
        ("vision_attention_matched", "vision-attn", VISION_ATTENTION_MATCHED_MODEL_PATH),
    ]


@dataclass
class RuntimeModels:
    screen_size: tuple[int, int]
    ridge_mapper: PolynomialRidgeGazeMapper | None = None
    vision_predictors: dict[str, VisionGazePredictor] = field(default_factory=dict)
    active_mode: str | None = None

    def available_modes(self) -> list[str]:
        modes: list[str] = []
        for mode, _, _ in vision_checkpoints():
            if mode in self.vision_predictors:
                modes.append(mode)
        if self.ridge_mapper is not None:
            modes.append("ridge")
        return modes

    def normalize_active_mode(self) -> None:
        modes = self.available_modes()
        if not modes:
            self.active_mode = None
            return
        if self.active_mode not in modes:
            self.active_mode = modes[0]

    def mode_display_name(self, mode: str | None) -> str | None:
        if mode is None:
            return None
        if mode == "ridge":
            return "ridge"
        for key, display_name, _ in vision_checkpoints():
            if mode == key:
                return display_name
        return mode

    def active_details(self) -> tuple[str | None, dict[str, float] | None]:
        return self.mode_details(self.active_mode)

    def mode_details(self, mode: str | None) -> tuple[str | None, dict[str, float] | None]:
        if mode is None:
            return None, None
        predictor = self.vision_predictors.get(mode)
        if predictor is not None:
            return predictor.label, predictor.metrics
        if mode == "ridge" and self.ridge_mapper is not None:
            return self.ridge_mapper.label, self.ridge_mapper.metrics
        return None, None

    def has_vision_predictors(self) -> bool:
        return bool(self.vision_predictors)

    def predict_normalized(
        self,
        payload: dict[str, float],
        eye_crops,
    ) -> tuple[float, float] | None:
        predictor = self.vision_predictors.get(self.active_mode or "")
        if predictor is not None and eye_crops is not None:
            return predictor.predict_normalized(eye_crops, payload)
        if self.active_mode == "ridge" and self.ridge_mapper is not None:
            return self.ridge_mapper.predict_normalized(payload)
        return None

    def predict_all_normalized(
        self,
        payload: dict[str, float],
        eye_crops,
    ) -> dict[str, tuple[float, float]]:
        predictions: dict[str, tuple[float, float]] = {}
        for mode in self.available_modes():
            predictor = self.vision_predictors.get(mode)
            if predictor is not None:
                if eye_crops is not None:
                    predictions[mode] = predictor.predict_normalized(eye_crops, payload)
                continue
            if mode == "ridge" and self.ridge_mapper is not None:
                predictions[mode] = self.ridge_mapper.predict_normalized(payload)
        return predictions

    def reload_ridge(self, collection) -> None:
        try:
            self.ridge_mapper = load_latest_mapper(screen_size_filter=self.screen_size)
        except Exception as error:
            set_status_message(collection, f"Mapper load failed: {error}")
            self.ridge_mapper = None
            self.normalize_active_mode()
            return

        self.normalize_active_mode()
        if self.ridge_mapper is not None:
            metrics_text = format_metrics(self.ridge_mapper.metrics)
            if metrics_text is not None:
                set_status_message(
                    collection,
                    f"Loaded mapper: {self.ridge_mapper.label} | {metrics_text}",
                )
            else:
                set_status_message(collection, f"Loaded mapper: {self.ridge_mapper.label}")
            return

        set_status_message(
            collection,
            "No manual training data yet. Move the target and press space to record.",
        )

    def reload_vision(self, collection) -> None:
        self.vision_predictors = {}
        errors: list[str] = []
        for mode, _, checkpoint_path in vision_checkpoints():
            try:
                predictor = load_vision_predictor(
                    checkpoint_path=checkpoint_path,
                    screen_size_filter=self.screen_size,
                )
            except Exception as error:
                predictor = None
                errors.append(f"{checkpoint_path.name}: {error}")
            if predictor is not None:
                self.vision_predictors[mode] = predictor
        if errors:
            set_status_message(collection, f"Vision load failed: {' | '.join(errors)}")
        self.normalize_active_mode()

    def load_initial(self, collection) -> None:
        self.reload_ridge(collection)
        self.reload_vision(collection)
        modes = self.available_modes()
        self.active_mode = modes[0] if modes else None

    def announce_active_mode(self, collection, include_compare_hint: bool = False) -> None:
        label, metrics = self.active_details()
        if self.active_mode is None:
            return

        mode_name = self.mode_display_name(self.active_mode)
        metrics_text = format_metrics(metrics)
        if metrics_text is not None:
            message = f"Active model: {mode_name} ({label}) | {metrics_text}"
        else:
            message = f"Active model: {mode_name} ({label})"
        if include_compare_hint and len(self.available_modes()) > 1:
            message = f"{message} | press v to compare"
        set_status_message(collection, message, duration=1.5)

    def cycle_active_mode(self, collection) -> str | None:
        # Reload on explicit model cycling so newly trained checkpoints appear
        # in the preview without requiring an app restart.
        self.reload_vision(collection)

        modes = self.available_modes()
        if not modes:
            set_status_message(
                collection,
                "No prediction model available yet. Train a model or collect ridge data first.",
                duration=1.5,
            )
            return None

        if self.active_mode not in modes:
            self.active_mode = modes[0]
        else:
            next_index = (modes.index(self.active_mode) + 1) % len(modes)
            self.active_mode = modes[next_index]
        self.announce_active_mode(collection)
        return self.active_mode
