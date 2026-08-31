"""Frozen semantic/forensic encoders, fusion heads, and checkpoint I/O."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from transformers import AutoImageProcessor, CLIPVisionModelWithProjection

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_hf_cache() -> str | None:
    """Prefer weights downloaded into this workspace, especially in offline mode."""
    cache = PROJECT_ROOT / ".hf-cache" / "hub"
    model_cache = cache / "models--openai--clip-vit-base-patch32"
    return str(cache) if model_cache.exists() else None


@dataclass
class ModelConfig:
    clip_model: str = "openai/clip-vit-base-patch32"
    hidden_dim: int = 256
    dropout: float = 0.3
    clip_dim: int = 512
    forensic_dim: int = 1280
    forensic_mode: str = "laplacian"
    head_type: str = "fusion"
    gate_mode: str = "features"
    quality_dim: int = 0


def image_quality_statistics(images: list[Image.Image]) -> torch.Tensor:
    """Six inexpensive degradation descriptors without raw size/format shortcuts."""
    rows = []
    for image in images:
        # Bound diagnostic cost without exposing raw width or height to a gate.
        gray_image = image.convert("L")
        width, height = gray_image.size
        scale = min(1.0, 256.0 / max(width, height))
        if scale < 1.0:
            gray_image = gray_image.resize((max(8, round(width * scale)), max(8, round(height * scale))), Image.Resampling.BILINEAR)
        gray = np.asarray(gray_image, dtype=np.float32) / 255.0
        lap = -4 * gray + np.roll(gray, 1, 0) + np.roll(gray, -1, 0) + np.roll(gray, 1, 1) + np.roll(gray, -1, 1)
        lap_energy = float(np.log1p(1000 * np.var(lap[1:-1, 1:-1])))
        spectrum = np.abs(np.fft.fftshift(np.fft.fft2((gray - gray.mean()) * np.outer(np.hanning(gray.shape[0]), np.hanning(gray.shape[1]))))) ** 2
        # Summaries capture sharpness, spectral falloff, blocking, and clipping.
        yy, xx = np.indices(gray.shape)
        radius = np.sqrt(((yy - (gray.shape[0] - 1) / 2) / max(gray.shape[0], 1)) ** 2 + ((xx - (gray.shape[1] - 1) / 2) / max(gray.shape[1], 1)) ** 2)
        low = float(spectrum[radius < 0.12].mean()) + 1e-8
        high = float(spectrum[radius > 0.3].mean()) + 1e-8
        hf_ratio = float(np.clip(np.log(high / low + 1e-8), -12, 4) / 8)
        bins = np.linspace(0.03, 0.5, 12)
        radial = np.array([spectrum[(radius >= a) & (radius < b)].mean() for a, b in zip(bins[:-1], bins[1:])])
        slope = float(np.polyfit(np.log((bins[:-1] + bins[1:]) / 2), np.log(radial + 1e-8), 1)[0] / 10)
        x_boundaries = np.arange(8, gray.shape[1], 8)
        y_boundaries = np.arange(8, gray.shape[0], 8)
        vertical = np.abs(gray[:, x_boundaries] - gray[:, x_boundaries - 1]).mean() if len(x_boundaries) else 0.0
        horizontal = np.abs(gray[y_boundaries, :] - gray[y_boundaries - 1, :]).mean() if len(y_boundaries) else 0.0
        block_energy = float((vertical + horizontal) * 5)
        contrast_span = float(np.quantile(gray, 0.95) - np.quantile(gray, 0.05))
        clipping_fraction = float(((gray <= 1 / 255) | (gray >= 254 / 255)).mean())
        rows.append((lap_energy, hf_ratio, slope, block_energy, contrast_span, clipping_fraction))
    return torch.tensor(rows, dtype=torch.float32)


class FrozenEncoders(nn.Module):
    """CLIP semantics plus EfficientNet features from reproducible forensic views."""

    def __init__(self, config: ModelConfig, device: torch.device) -> None:
        super().__init__()
        self.config = config
        self.device = device
        cache_dir = _project_hf_cache()
        # Both backbones are pretrained feature extractors; only the fusion head trains.
        self.processor = AutoImageProcessor.from_pretrained(config.clip_model, cache_dir=cache_dir)
        self.clip = CLIPVisionModelWithProjection.from_pretrained(config.clip_model, cache_dir=cache_dir)
        project_torch_hub = PROJECT_ROOT / ".torch-cache" / "hub"
        if project_torch_hub.exists():
            torch.hub.set_dir(str(project_torch_hub))
        self.forensic = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.forensic.classifier = nn.Identity()
        expected_dim = 2560 if config.forensic_mode == "laplacian_fft" else 1280
        if config.forensic_mode not in {"laplacian", "fft", "laplacian_fft"}:
            raise ValueError(f"Unknown forensic mode: {config.forensic_mode!r}")
        if config.forensic_dim != expected_dim:
            raise ValueError(
                f"forensic_dim={config.forensic_dim} does not match "
                f"forensic_mode={config.forensic_mode!r} (expected {expected_dim})"
            )
        # Freezing here prevents later callers from accidentally enabling gradients.
        self.eval().requires_grad_(False).to(device)

    @staticmethod
    def _image_tensor(images: list[Image.Image], device: torch.device) -> torch.Tensor:
        tensors = []
        for image in images:
            resized = image.resize((224, 224), Image.Resampling.BILINEAR)
            array = torch.from_numpy(__import__("numpy").array(resized, dtype="float32")).permute(2, 0, 1) / 255.0
            tensors.append(array)
        return torch.stack(tensors).to(device)

    @staticmethod
    def _imagenet_normalize(view: torch.Tensor, device: torch.device) -> torch.Tensor:
        view = view.repeat(1, 3, 1, 1)
        mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
        return (view - mean) / std

    @classmethod
    def _laplacian_tensor(cls, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=device)
        edge = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), kernel.view(1, 1, 3, 3)).abs()
        # Per-image robust scaling keeps a few strong edges from saturating the view.
        scale = torch.quantile(edge.flatten(1), 0.99, dim=1).clamp_min(1e-4).view(-1, 1, 1, 1)
        return cls._imagenet_normalize((edge / scale).clamp(0, 1), device)

    @classmethod
    def _fft_tensor(cls, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Centered, windowed log-magnitude spectrum with robust per-image scaling."""
        gray = x.mean(dim=1, keepdim=True)
        height, width = gray.shape[-2:]
        window = torch.outer(
            torch.hann_window(height, periodic=False, device=device),
            torch.hann_window(width, periodic=False, device=device),
        ).view(1, 1, height, width)
        # Mean removal suppresses DC; the Hann window reduces artificial border energy.
        spectrum = torch.fft.fftshift(torch.fft.fft2((gray - gray.mean(dim=(-2, -1), keepdim=True)) * window))
        magnitude = torch.log1p(spectrum.abs())
        flat = magnitude.flatten(1)
        low = torch.quantile(flat, 0.01, dim=1).view(-1, 1, 1, 1)
        high = torch.quantile(flat, 0.99, dim=1).view(-1, 1, 1, 1)
        # Robust per-image scaling makes spectra comparable across exposure levels.
        normalized = ((magnitude - low) / (high - low).clamp_min(1e-4)).clamp(0, 1)
        return cls._imagenet_normalize(normalized, device)

    def _forensic_features(self, images: list[Image.Image]) -> torch.Tensor:
        x = self._image_tensor(images, self.device)
        views = []
        if self.config.forensic_mode in {"laplacian", "laplacian_fft"}:
            views.append(self._laplacian_tensor(x, self.device))
        if self.config.forensic_mode in {"fft", "laplacian_fft"}:
            views.append(self._fft_tensor(x, self.device))
        # The same frozen EfficientNet processes each forensic view independently.
        return torch.cat([F.normalize(self.forensic(view), dim=1) for view in views], dim=1)

    @torch.inference_mode()
    def forward(self, images: list[Image.Image]) -> torch.Tensor:
        # CLIP keeps its own official processor; forensic views use fixed 224px inputs.
        clip_inputs = self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)
        clip_features = self.clip(pixel_values=clip_inputs).image_embeds
        forensic_features = self._forensic_features(images)
        features = torch.cat((F.normalize(clip_features, dim=1), forensic_features), dim=1).cpu()
        # Quality descriptors exist only for historical gated heads, not the selected head.
        if self.config.quality_dim:
            quality = image_quality_statistics(images)
            if quality.shape[1] != self.config.quality_dim:
                raise ValueError(f"quality_dim={self.config.quality_dim}, extracted {quality.shape[1]} statistics")
            features = torch.cat((features, quality), dim=1)
        return features


class FusionHead(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        # Feature blocks are concatenated in the fixed CLIP/Laplacian/FFT order.
        input_dim = config.clip_dim + config.forensic_dim
        # Keep the only trainable component small relative to the frozen backbones.
        self.network = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)


class ExpertMixtureHead(nn.Module):
    """Mixture of CLIP+Laplacian and CLIP+FFT experts with an optional gate."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.forensic_mode != "laplacian_fft" or config.forensic_dim != 2560:
            raise ValueError("Expert mixture requires laplacian_fft features")
        expert_config = ModelConfig(
            clip_model=config.clip_model, hidden_dim=config.hidden_dim, dropout=config.dropout,
            clip_dim=config.clip_dim, forensic_dim=1280, forensic_mode="laplacian",
        )
        self.config = config
        self.laplacian_expert = FusionHead(expert_config)
        self.fft_expert = FusionHead(expert_config)
        if config.gate_mode in {"features", "quality"}:
            gate_dim = config.clip_dim + config.forensic_dim + (config.quality_dim if config.gate_mode == "quality" else 0)
            self.gate = nn.Sequential(
                nn.LayerNorm(gate_dim), nn.Linear(gate_dim, 32), nn.GELU(), nn.Linear(32, 1),
            )
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.constant_(self.gate[-1].bias, -1.3862944)  # start at 20% FFT
        elif config.gate_mode != "fixed":
            raise ValueError(f"Unknown gate mode: {config.gate_mode!r}")

    def expert_logits(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        clip = features[:, : self.config.clip_dim]
        laplacian = features[:, self.config.clip_dim : self.config.clip_dim + 1280]
        fft = features[:, self.config.clip_dim + 1280 : self.config.clip_dim + 2560]
        return self.laplacian_expert(torch.cat((clip, laplacian), 1)), self.fft_expert(torch.cat((clip, fft), 1))

    def gate_weights(self, features: torch.Tensor) -> torch.Tensor:
        if self.config.gate_mode == "fixed":
            return torch.full((len(features),), 0.5, device=features.device)
        gate_features = features if self.config.gate_mode == "quality" else features[:, : self.config.clip_dim + self.config.forensic_dim]
        return torch.sigmoid(self.gate(gate_features).squeeze(1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        laplacian_logit, fft_logit = self.expert_logits(features)
        fft_weight = self.gate_weights(features)
        return (1 - fft_weight) * laplacian_logit + fft_weight * fft_logit


class AdaptiveTriExpertHead(nn.Module):
    """Adaptive CLIP, CLIP+Laplacian, and CLIP+FFT ensemble.

    The semantic branch is intentionally independent of fragile high-frequency
    evidence, while the gate can favor forensic experts on high-quality inputs.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.forensic_mode != "laplacian_fft" or config.forensic_dim != 2560:
            raise ValueError("Three-expert mixture requires laplacian_fft features")
        self.config = config
        semantic_config = ModelConfig(
            clip_model=config.clip_model, hidden_dim=config.hidden_dim, dropout=config.dropout,
            clip_dim=config.clip_dim, forensic_dim=0, forensic_mode="laplacian",
        )
        forensic_config = ModelConfig(
            clip_model=config.clip_model, hidden_dim=config.hidden_dim, dropout=config.dropout,
            clip_dim=config.clip_dim, forensic_dim=1280, forensic_mode="laplacian",
        )
        self.semantic_expert = FusionHead(semantic_config)
        self.laplacian_expert = FusionHead(forensic_config)
        self.fft_expert = FusionHead(forensic_config)
        if config.gate_mode in {"features", "quality"}:
            gate_dim = config.clip_dim + config.forensic_dim + (
                config.quality_dim if config.gate_mode == "quality" else 0
            )
            self.gate = nn.Sequential(
                nn.LayerNorm(gate_dim), nn.Linear(gate_dim, 48), nn.GELU(), nn.Linear(48, 3),
            )
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.constant_(self.gate[-1].bias[0], -0.7985077)  # log(0.45)
            nn.init.constant_(self.gate[-1].bias[1], -1.0498221)  # log(0.35)
            nn.init.constant_(self.gate[-1].bias[2], -1.6094379)  # log(0.20)
        elif config.gate_mode != "fixed":
            raise ValueError(f"Unknown gate mode: {config.gate_mode!r}")

    def expert_logits(self, features: torch.Tensor) -> torch.Tensor:
        clip = features[:, : self.config.clip_dim]
        laplacian = features[:, self.config.clip_dim : self.config.clip_dim + 1280]
        fft = features[:, self.config.clip_dim + 1280 : self.config.clip_dim + 2560]
        return torch.stack((
            self.semantic_expert(clip),
            self.laplacian_expert(torch.cat((clip, laplacian), 1)),
            self.fft_expert(torch.cat((clip, fft), 1)),
        ), dim=1)

    def gate_weights(self, features: torch.Tensor) -> torch.Tensor:
        if self.config.gate_mode == "fixed":
            return torch.tensor((0.45, 0.35, 0.20), device=features.device).expand(len(features), -1)
        gate_features = features if self.config.gate_mode == "quality" else features[
            :, : self.config.clip_dim + self.config.forensic_dim
        ]
        return torch.softmax(self.gate(gate_features), dim=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return (self.expert_logits(features) * self.gate_weights(features)).sum(dim=1)


def save_checkpoint(
    path: str | Path, head: nn.Module, config: ModelConfig, temperature: float, metadata: dict
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"head_state_dict": head.state_dict(), "config": asdict(config), "temperature": temperature, "metadata": metadata},
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[nn.Module, ModelConfig, float, dict]:
    # weights_only avoids executing arbitrary pickled code from checkpoint files.
    payload = torch.load(path, map_location=device, weights_only=True)
    config = ModelConfig(**payload["config"])
    if config.head_type == "mixture":
        head = ExpertMixtureHead(config)
    elif config.head_type == "tri_mixture":
        head = AdaptiveTriExpertHead(config)
    elif config.head_type == "fusion":
        head = FusionHead(config)
    else:
        raise ValueError(f"Unknown head type: {config.head_type!r}")
    # Construct from serialized configuration before enforcing exact parameter shapes.
    head = head.to(device)
    head.load_state_dict(payload["head_state_dict"])
    head.eval()
    return head, config, float(payload.get("temperature", 1.0)), payload.get("metadata", {})
