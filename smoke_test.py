from __future__ import annotations

import py_compile
import subprocess
import warnings
from pathlib import Path

import torch

from constants import EYE_CROP_HEIGHT, EYE_CROP_WIDTH, VISION_HEAD_FEATURE_KEYS
from runtime_models import vision_checkpoints
from vision_model import (
    EyeCropModelConfig,
    EyeCropRegressor,
    best_frame_vision_config,
    matched_attention_frame_vision_config,
    spatial_frame_vision_config,
    spatial_geometry_frame_vision_config,
    tiny_patch_transformer_frame_vision_config,
)


ROOT_DIR = Path(__file__).resolve().parent
warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True.*",
    category=UserWarning,
)


def tracked_python_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return [ROOT_DIR / line for line in result.stdout.splitlines() if line]
    return sorted(ROOT_DIR.glob("*.py"))


def compile_python_files() -> None:
    for path in tracked_python_files():
        py_compile.compile(str(path), doraise=True)


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def check_model_forward(name: str, config: EyeCropModelConfig) -> None:
    extra_dim = len(config.extra_feature_keys)
    model = EyeCropRegressor(
        head_feature_dim=len(VISION_HEAD_FEATURE_KEYS),
        extra_feature_dim=extra_dim,
        config=config,
    )
    model.eval()

    batch_size = 2
    left_eye = torch.rand(batch_size, 1, EYE_CROP_HEIGHT, EYE_CROP_WIDTH)
    right_eye = torch.rand(batch_size, 1, EYE_CROP_HEIGHT, EYE_CROP_WIDTH)
    head_features = torch.zeros(batch_size, len(VISION_HEAD_FEATURE_KEYS))
    extra_features = torch.zeros(batch_size, extra_dim) if extra_dim else None

    with torch.no_grad():
        output = model(left_eye, right_eye, head_features, extra_features)

    if tuple(output.shape) != (batch_size, 2):
        raise AssertionError(f"{name} output shape mismatch: {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise AssertionError(f"{name} produced non-finite outputs")
    if not ((0.0 <= output).all() and (output <= 1.0).all()):
        raise AssertionError(f"{name} outputs must be normalized to [0, 1]")

    print(f"ok model {name}: {parameter_count(model):,} params")


def main() -> None:
    compile_python_files()
    print("ok compile tracked Python files")

    model_configs = {
        "concat": best_frame_vision_config(),
        "attn": matched_attention_frame_vision_config(),
        "spatial": spatial_frame_vision_config(),
        "spatial_geom": spatial_geometry_frame_vision_config(),
        "vit": tiny_patch_transformer_frame_vision_config(),
    }
    for name, config in model_configs.items():
        check_model_forward(name, config)

    checkpoint_names = ", ".join(label for _, label, _ in vision_checkpoints())
    print(f"ok checkpoint priority: {checkpoint_names}")


if __name__ == "__main__":
    main()
