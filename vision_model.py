from dataclasses import asdict, dataclass

import torch
from torch import nn

from constants import (
    EYE_CROP_HEIGHT,
    EYE_CROP_WIDTH,
    VISION_EYE_GEOMETRY_FEATURE_KEYS,
    VISION_HEAD_FEATURE_KEYS,
)


@dataclass
class EyeCropModelConfig:
    encoder_channels: tuple[int, ...] = (16, 32, 64, 64)
    encoder_type: str = "cnn"
    encoder_pooling: str = "avg"
    eye_coord_channels: bool = False
    head_hidden_dims: tuple[int, ...] = (32, 32)
    extra_feature_keys: tuple[str, ...] = ()
    extra_hidden_dims: tuple[int, ...] = (32,)
    regressor_hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.1
    fusion_mode: str = "concat"
    token_dim: int = 128
    attention_heads: int = 4
    attention_layers: int = 1
    attention_dropout: float = 0.1
    patch_size: tuple[int, int] = (8, 8)
    patch_layers: int = 2
    patch_heads: int = 4
    patch_dropout: float = 0.1
    clifford_blades: int = 4
    clifford_kernel_size: int = 3

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict | None) -> "EyeCropModelConfig":
        if not payload:
            return cls()
        return cls(
            encoder_channels=tuple(payload.get("encoder_channels", cls().encoder_channels)),
            encoder_type=str(payload.get("encoder_type", cls().encoder_type)),
            encoder_pooling=str(payload.get("encoder_pooling", cls().encoder_pooling)),
            eye_coord_channels=bool(payload.get("eye_coord_channels", cls().eye_coord_channels)),
            head_hidden_dims=tuple(payload.get("head_hidden_dims", cls().head_hidden_dims)),
            extra_feature_keys=tuple(payload.get("extra_feature_keys", cls().extra_feature_keys)),
            extra_hidden_dims=tuple(payload.get("extra_hidden_dims", cls().extra_hidden_dims)),
            regressor_hidden_dims=tuple(payload.get("regressor_hidden_dims", cls().regressor_hidden_dims)),
            dropout=float(payload.get("dropout", cls().dropout)),
            fusion_mode=str(payload.get("fusion_mode", cls().fusion_mode)),
            token_dim=int(payload.get("token_dim", cls().token_dim)),
            attention_heads=int(payload.get("attention_heads", cls().attention_heads)),
            attention_layers=int(payload.get("attention_layers", cls().attention_layers)),
            attention_dropout=float(payload.get("attention_dropout", cls().attention_dropout)),
            patch_size=tuple(payload.get("patch_size", cls().patch_size)),
            patch_layers=int(payload.get("patch_layers", cls().patch_layers)),
            patch_heads=int(payload.get("patch_heads", cls().patch_heads)),
            patch_dropout=float(payload.get("patch_dropout", cls().patch_dropout)),
            clifford_blades=int(payload.get("clifford_blades", cls().clifford_blades)),
            clifford_kernel_size=int(payload.get("clifford_kernel_size", cls().clifford_kernel_size)),
        )


def best_frame_vision_config() -> EyeCropModelConfig:
    return EyeCropModelConfig(
        encoder_channels=(24, 48, 96, 96),
        head_hidden_dims=(48, 48),
        regressor_hidden_dims=(192, 96),
        dropout=0.1,
    )


def matched_attention_frame_vision_config() -> EyeCropModelConfig:
    return EyeCropModelConfig(
        encoder_channels=(24, 48, 96, 96),
        head_hidden_dims=(48, 48),
        regressor_hidden_dims=(192, 80),
        dropout=0.1,
        fusion_mode="attention",
        token_dim=48,
        attention_heads=4,
        attention_layers=1,
        attention_dropout=0.1,
    )


def spatial_frame_vision_config() -> EyeCropModelConfig:
    return EyeCropModelConfig(
        encoder_channels=(24, 48, 96, 96),
        encoder_pooling="flatten",
        eye_coord_channels=True,
        head_hidden_dims=(48, 48),
        regressor_hidden_dims=(256, 128),
        dropout=0.15,
    )


def spatial_geometry_frame_vision_config() -> EyeCropModelConfig:
    return EyeCropModelConfig(
        encoder_channels=(24, 48, 96, 96),
        encoder_pooling="flatten",
        eye_coord_channels=True,
        head_hidden_dims=(48, 48),
        extra_feature_keys=tuple(VISION_EYE_GEOMETRY_FEATURE_KEYS),
        extra_hidden_dims=(48, 48),
        regressor_hidden_dims=(256, 128),
        dropout=0.15,
    )


def tiny_patch_transformer_frame_vision_config() -> EyeCropModelConfig:
    return EyeCropModelConfig(
        encoder_type="patch_transformer",
        eye_coord_channels=True,
        head_hidden_dims=(48, 48),
        regressor_hidden_dims=(192, 96),
        dropout=0.1,
        token_dim=64,
        patch_size=(8, 8),
        patch_layers=2,
        patch_heads=4,
        patch_dropout=0.1,
    )


def clifford_frame_vision_config() -> EyeCropModelConfig:
    return EyeCropModelConfig(
        encoder_channels=(16, 24, 32, 32),
        encoder_type="clifford",
        eye_coord_channels=True,
        head_hidden_dims=(48, 48),
        regressor_hidden_dims=(128, 64),
        dropout=0.15,
        clifford_blades=4,
        clifford_kernel_size=3,
    )


class EyeEncoder(nn.Module):
    def __init__(
        self,
        channels: tuple[int, ...],
        input_channels: int = 1,
        pooling: str = "avg",
    ) -> None:
        super().__init__()
        if pooling not in {"avg", "flatten"}:
            raise ValueError(f"Unsupported CNN encoder pooling: {pooling}")
        self.pooling = pooling
        layers: list[nn.Module] = []
        in_channels = input_channels
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.GELU(),
                ]
            )
            in_channels = out_channels
        self.layers = nn.Sequential(*layers)
        if pooling == "avg":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.output_dim = channels[-1]
        else:
            self.pool = nn.Identity()
            with torch.no_grad():
                dummy = torch.zeros(1, input_channels, EYE_CROP_HEIGHT, EYE_CROP_WIDTH)
                encoded = self.pool(self.layers(dummy))
            self.output_dim = int(encoded.flatten(start_dim=1).shape[1])

    def forward(self, eye_image: torch.Tensor) -> torch.Tensor:
        encoded = self.pool(self.layers(eye_image))
        return encoded.flatten(start_dim=1)


class PatchEyeEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        token_dim: int,
        patch_size: tuple[int, int],
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if token_dim % heads != 0:
            raise ValueError("token_dim must be divisible by patch transformer heads")
        if EYE_CROP_HEIGHT % patch_size[0] != 0 or EYE_CROP_WIDTH % patch_size[1] != 0:
            raise ValueError("Eye crop size must be divisible by patch_size")

        self.patch_embed = nn.Conv2d(
            input_channels,
            token_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        token_count = (EYE_CROP_HEIGHT // patch_size[0]) * (EYE_CROP_WIDTH // patch_size[1])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, token_count + 1, token_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(token_dim)
        self.output_dim = token_dim

    def forward(self, eye_image: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(eye_image).flatten(start_dim=2).transpose(1, 2)
        cls_token = self.cls_token.expand(eye_image.shape[0], -1, -1)
        tokens = torch.cat([cls_token, patches], dim=1)
        tokens = tokens + self.pos_embedding
        encoded = self.encoder(tokens)
        return self.norm(encoded[:, 0, :])


class CliffordProductBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        blades: int = 4,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if blades != 4:
            raise ValueError("CliffordProductBlock currently implements Cl(2,0) with 4 blades")
        if kernel_size % 2 == 0:
            raise ValueError("clifford_kernel_size must be odd")

        self.channels = channels
        self.blades = blades
        width = channels * blades
        padding = kernel_size // 2
        self.norm = nn.GroupNorm(num_groups=blades, num_channels=width)
        self.local_mixer = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=kernel_size, padding=padding, groups=width),
            nn.Conv2d(width, width, kernel_size=1, groups=blades),
            nn.GELU(),
        )
        self.product_projection = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=1),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(features)
        local = self.local_mixer(normalized)
        product = self.geometric_product(normalized, local)
        return features + self.residual_scale * self.product_projection(product)

    def geometric_product(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = left.shape
        left_blades = left.reshape(batch_size, self.blades, self.channels, height, width)
        right_blades = right.reshape(batch_size, self.blades, self.channels, height, width)
        left_scalar, left_e1, left_e2, left_bivector = left_blades.unbind(dim=1)
        right_scalar, right_e1, right_e2, right_bivector = right_blades.unbind(dim=1)

        # Cl(2,0) geometric product:
        # scalar + e1 + e2 + e12 components are multiplied with the sign
        # pattern induced by e1^2 = e2^2 = 1 and e12^2 = -1. This gives the
        # block a local bilinear interaction that is more structured than a
        # generic pointwise MLP while remaining lightweight and stable.
        product_scalar = (
            left_scalar * right_scalar
            + left_e1 * right_e1
            + left_e2 * right_e2
            - left_bivector * right_bivector
        )
        product_e1 = (
            left_scalar * right_e1
            + left_e1 * right_scalar
            - left_e2 * right_bivector
            + left_bivector * right_e2
        )
        product_e2 = (
            left_scalar * right_e2
            + left_e2 * right_scalar
            + left_e1 * right_bivector
            - left_bivector * right_e1
        )
        product_bivector = (
            left_scalar * right_bivector
            + left_bivector * right_scalar
            + left_e1 * right_e2
            - left_e2 * right_e1
        )
        return torch.stack(
            [product_scalar, product_e1, product_e2, product_bivector],
            dim=1,
        ).reshape(batch_size, self.blades * self.channels, height, width)


class CliffordEyeEncoder(nn.Module):
    def __init__(
        self,
        channels: tuple[int, ...],
        input_channels: int = 1,
        blades: int = 4,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if not channels:
            raise ValueError("CliffordEyeEncoder requires at least one channel width")
        if blades != 4:
            raise ValueError("CliffordEyeEncoder currently implements Cl(2,0) with 4 blades")

        first_width = channels[0] * blades
        layers: list[nn.Module] = [
            nn.Conv2d(input_channels, first_width, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            CliffordProductBlock(channels[0], blades=blades, kernel_size=kernel_size),
        ]
        in_channels = channels[0]
        for out_channels in channels[1:]:
            layers.extend(
                [
                    nn.GroupNorm(num_groups=blades, num_channels=in_channels * blades),
                    nn.Conv2d(
                        in_channels * blades,
                        out_channels * blades,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.GELU(),
                    CliffordProductBlock(out_channels, blades=blades, kernel_size=kernel_size),
                ]
            )
            in_channels = out_channels

        self.layers = nn.Sequential(*layers)
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, EYE_CROP_HEIGHT, EYE_CROP_WIDTH)
            encoded = self.layers(dummy)
        self.output_dim = int(encoded.flatten(start_dim=1).shape[1])

    def forward(self, eye_image: torch.Tensor) -> torch.Tensor:
        encoded = self.layers(eye_image)
        return encoded.flatten(start_dim=1)


class EyeCropRegressor(nn.Module):
    def __init__(
        self,
        head_feature_dim: int = len(VISION_HEAD_FEATURE_KEYS),
        extra_feature_dim: int = 0,
        config: EyeCropModelConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or EyeCropModelConfig()
        self.extra_feature_dim = extra_feature_dim
        self.eye_input_channels = 3 if self.config.eye_coord_channels else 1
        if self.config.encoder_type == "cnn":
            self.eye_encoder = EyeEncoder(
                self.config.encoder_channels,
                input_channels=self.eye_input_channels,
                pooling=self.config.encoder_pooling,
            )
        elif self.config.encoder_type == "patch_transformer":
            self.eye_encoder = PatchEyeEncoder(
                input_channels=self.eye_input_channels,
                token_dim=self.config.token_dim,
                patch_size=self.config.patch_size,
                layers=self.config.patch_layers,
                heads=self.config.patch_heads,
                dropout=self.config.patch_dropout,
            )
        elif self.config.encoder_type == "clifford":
            self.eye_encoder = CliffordEyeEncoder(
                self.config.encoder_channels,
                input_channels=self.eye_input_channels,
                blades=self.config.clifford_blades,
                kernel_size=self.config.clifford_kernel_size,
            )
        else:
            raise ValueError(f"Unsupported eye encoder type: {self.config.encoder_type}")

        head_layers: list[nn.Module] = []
        in_dim = head_feature_dim
        for hidden_dim in self.config.head_hidden_dims:
            head_layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU()])
            in_dim = hidden_dim
        self.head_mlp = nn.Sequential(*head_layers)
        head_output_dim = in_dim

        extra_layers: list[nn.Module] = []
        in_dim = extra_feature_dim
        for hidden_dim in self.config.extra_hidden_dims:
            if in_dim <= 0:
                break
            extra_layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU()])
            in_dim = hidden_dim
        self.extra_mlp = nn.Sequential(*extra_layers)
        extra_output_dim = in_dim if extra_feature_dim > 0 else 0

        self.fusion_mode = self.config.fusion_mode
        if self.fusion_mode not in {"concat", "attention"}:
            raise ValueError(f"Unsupported fusion mode: {self.fusion_mode}")

        if self.fusion_mode == "attention":
            if self.config.token_dim % self.config.attention_heads != 0:
                raise ValueError("token_dim must be divisible by attention_heads")
            self.left_token_projection = nn.Linear(self.eye_encoder.output_dim, self.config.token_dim)
            self.right_token_projection = nn.Linear(self.eye_encoder.output_dim, self.config.token_dim)
            self.head_token_projection = nn.Linear(head_output_dim + extra_output_dim, self.config.token_dim)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.config.token_dim))
            self.modality_embeddings = nn.Parameter(torch.zeros(4, self.config.token_dim))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.config.token_dim,
                nhead=self.config.attention_heads,
                dim_feedforward=self.config.token_dim * 4,
                dropout=self.config.attention_dropout,
                activation="gelu",
                batch_first=True,
            )
            self.fusion_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=self.config.attention_layers,
            )
            self.fusion_norm = nn.LayerNorm(self.config.token_dim)
            regressor_in_dim = self.config.token_dim
        else:
            regressor_in_dim = self.eye_encoder.output_dim * 2 + head_output_dim + extra_output_dim

        regressor_layers: list[nn.Module] = []
        for index, hidden_dim in enumerate(self.config.regressor_hidden_dims):
            regressor_layers.extend([nn.Linear(regressor_in_dim, hidden_dim), nn.GELU()])
            if self.config.dropout > 0.0 and index == 0:
                regressor_layers.append(nn.Dropout(p=self.config.dropout))
            regressor_in_dim = hidden_dim
        regressor_layers.append(nn.Linear(regressor_in_dim, 2))
        self.regressor = nn.Sequential(*regressor_layers)

    def forward(
        self,
        left_eye: torch.Tensor,
        right_eye: torch.Tensor,
        head_features: torch.Tensor,
        extra_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        left_eye = self._prepare_eye(left_eye)
        right_eye = self._prepare_eye(right_eye)
        left_embedding = self.eye_encoder(left_eye)
        right_embedding = self.eye_encoder(right_eye)
        head_embedding = self.head_mlp(head_features)
        if self.extra_feature_dim > 0 and extra_features is None:
            extra_features = head_features.new_zeros((head_features.shape[0], self.extra_feature_dim))
        if extra_features is not None and extra_features.shape[-1] > 0:
            extra_embedding = self.extra_mlp(extra_features)
            head_embedding = torch.cat([head_embedding, extra_embedding], dim=1)
        if self.fusion_mode == "attention":
            batch_size = left_embedding.shape[0]
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            tokens = torch.stack(
                [
                    self.left_token_projection(left_embedding),
                    self.right_token_projection(right_embedding),
                    self.head_token_projection(head_embedding),
                ],
                dim=1,
            )
            cls_token = cls_token + self.modality_embeddings[0].view(1, 1, -1)
            tokens = tokens + self.modality_embeddings[1:].unsqueeze(0)
            fused_tokens = self.fusion_encoder(torch.cat([cls_token, tokens], dim=1))
            fused = self.fusion_norm(fused_tokens[:, 0, :])
        else:
            fused = torch.cat([left_embedding, right_embedding, head_embedding], dim=1)
        logits = self.regressor(fused)
        return torch.sigmoid(logits)

    def _prepare_eye(self, eye: torch.Tensor) -> torch.Tensor:
        if not self.config.eye_coord_channels:
            return eye
        batch_size, _, height, width = eye.shape
        x_coords = torch.linspace(-1.0, 1.0, width, device=eye.device, dtype=eye.dtype)
        y_coords = torch.linspace(-1.0, 1.0, height, device=eye.device, dtype=eye.dtype)
        x_grid = x_coords.view(1, 1, 1, width).expand(batch_size, 1, height, width)
        y_grid = y_coords.view(1, 1, height, 1).expand(batch_size, 1, height, width)
        return torch.cat([eye, x_grid, y_grid], dim=1)


@dataclass
class VisionCheckpointMetadata:
    screen_size: tuple[int, int]
    head_feature_keys: list[str]
    head_mean: list[float]
    head_scale: list[float]
    extra_feature_keys: list[str]
    extra_mean: list[float]
    extra_scale: list[float]
    model_config: dict
    train_sample_count: int
    eval_sample_count: int
    eval_mae_x_px: float | None
    eval_mae_y_px: float | None


def build_checkpoint_payload(
    model: EyeCropRegressor,
    metadata: VisionCheckpointMetadata,
) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "screen_size": list(metadata.screen_size),
        "head_feature_keys": metadata.head_feature_keys,
        "head_mean": metadata.head_mean,
        "head_scale": metadata.head_scale,
        "extra_feature_keys": metadata.extra_feature_keys,
        "extra_mean": metadata.extra_mean,
        "extra_scale": metadata.extra_scale,
        "model_config": metadata.model_config,
        "train_sample_count": metadata.train_sample_count,
        "eval_sample_count": metadata.eval_sample_count,
        "eval_mae_x_px": metadata.eval_mae_x_px,
        "eval_mae_y_px": metadata.eval_mae_y_px,
    }
