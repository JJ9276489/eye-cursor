from dataclasses import asdict, dataclass

import torch
from torch import nn

from constants import VISION_HEAD_FEATURE_KEYS


@dataclass
class EyeCropModelConfig:
    encoder_channels: tuple[int, ...] = (16, 32, 64, 64)
    head_hidden_dims: tuple[int, ...] = (32, 32)
    regressor_hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.1
    fusion_mode: str = "concat"
    token_dim: int = 128
    attention_heads: int = 4
    attention_layers: int = 1
    attention_dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict | None) -> "EyeCropModelConfig":
        if not payload:
            return cls()
        return cls(
            encoder_channels=tuple(payload.get("encoder_channels", cls().encoder_channels)),
            head_hidden_dims=tuple(payload.get("head_hidden_dims", cls().head_hidden_dims)),
            regressor_hidden_dims=tuple(payload.get("regressor_hidden_dims", cls().regressor_hidden_dims)),
            dropout=float(payload.get("dropout", cls().dropout)),
            fusion_mode=str(payload.get("fusion_mode", cls().fusion_mode)),
            token_dim=int(payload.get("token_dim", cls().token_dim)),
            attention_heads=int(payload.get("attention_heads", cls().attention_heads)),
            attention_layers=int(payload.get("attention_layers", cls().attention_layers)),
            attention_dropout=float(payload.get("attention_dropout", cls().attention_dropout)),
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


class EyeEncoder(nn.Module):
    def __init__(self, channels: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.GELU(),
                ]
            )
            in_channels = out_channels
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.layers = nn.Sequential(*layers)
        self.output_dim = channels[-1]

    def forward(self, eye_image: torch.Tensor) -> torch.Tensor:
        encoded = self.layers(eye_image)
        return encoded.flatten(start_dim=1)


class EyeCropRegressor(nn.Module):
    def __init__(
        self,
        head_feature_dim: int = len(VISION_HEAD_FEATURE_KEYS),
        config: EyeCropModelConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or EyeCropModelConfig()
        self.eye_encoder = EyeEncoder(self.config.encoder_channels)

        head_layers: list[nn.Module] = []
        in_dim = head_feature_dim
        for hidden_dim in self.config.head_hidden_dims:
            head_layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU()])
            in_dim = hidden_dim
        self.head_mlp = nn.Sequential(*head_layers)
        head_output_dim = in_dim

        self.fusion_mode = self.config.fusion_mode
        if self.fusion_mode not in {"concat", "attention"}:
            raise ValueError(f"Unsupported fusion mode: {self.fusion_mode}")

        if self.fusion_mode == "attention":
            if self.config.token_dim % self.config.attention_heads != 0:
                raise ValueError("token_dim must be divisible by attention_heads")
            self.left_token_projection = nn.Linear(self.eye_encoder.output_dim, self.config.token_dim)
            self.right_token_projection = nn.Linear(self.eye_encoder.output_dim, self.config.token_dim)
            self.head_token_projection = nn.Linear(head_output_dim, self.config.token_dim)
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
            regressor_in_dim = self.eye_encoder.output_dim * 2 + head_output_dim

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
    ) -> torch.Tensor:
        left_embedding = self.eye_encoder(left_eye)
        right_embedding = self.eye_encoder(right_eye)
        head_embedding = self.head_mlp(head_features)
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


@dataclass
class VisionCheckpointMetadata:
    screen_size: tuple[int, int]
    head_feature_keys: list[str]
    head_mean: list[float]
    head_scale: list[float]
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
        "model_config": metadata.model_config,
        "train_sample_count": metadata.train_sample_count,
        "eval_sample_count": metadata.eval_sample_count,
        "eval_mae_x_px": metadata.eval_mae_x_px,
        "eval_mae_y_px": metadata.eval_mae_y_px,
    }
