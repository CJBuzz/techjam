from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class FrozenEncoders(nn.Module):
    """CLIP semantics plus EfficientNet features from reproducible forensic views."""

    def __init__(self, config: ModelConfig, device: torch.device) -> None:
        super().__init__()
        self.config = config
        self.device = device
        cache_dir = _project_hf_cache()
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
        spectrum = torch.fft.fftshift(torch.fft.fft2((gray - gray.mean(dim=(-2, -1), keepdim=True)) * window))
        magnitude = torch.log1p(spectrum.abs())
        flat = magnitude.flatten(1)
        low = torch.quantile(flat, 0.01, dim=1).view(-1, 1, 1, 1)
        high = torch.quantile(flat, 0.99, dim=1).view(-1, 1, 1, 1)
        normalized = ((magnitude - low) / (high - low).clamp_min(1e-4)).clamp(0, 1)
        return cls._imagenet_normalize(normalized, device)

    def _forensic_features(self, images: list[Image.Image]) -> torch.Tensor:
        x = self._image_tensor(images, self.device)
        views = []
        if self.config.forensic_mode in {"laplacian", "laplacian_fft"}:
            views.append(self._laplacian_tensor(x, self.device))
        if self.config.forensic_mode in {"fft", "laplacian_fft"}:
            views.append(self._fft_tensor(x, self.device))
        return torch.cat([F.normalize(self.forensic(view), dim=1) for view in views], dim=1)

    @torch.inference_mode()
    def forward(self, images: list[Image.Image]) -> torch.Tensor:
        clip_inputs = self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)
        clip_features = self.clip(pixel_values=clip_inputs).image_embeds
        forensic_features = self._forensic_features(images)
        return torch.cat((F.normalize(clip_features, dim=1), forensic_features), dim=1).cpu()


class FusionHead(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        input_dim = config.clip_dim + config.forensic_dim
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


def save_checkpoint(
    path: str | Path, head: FusionHead, config: ModelConfig, temperature: float, metadata: dict
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"head_state_dict": head.state_dict(), "config": asdict(config), "temperature": temperature, "metadata": metadata},
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[FusionHead, ModelConfig, float, dict]:
    payload = torch.load(path, map_location=device, weights_only=True)
    config = ModelConfig(**payload["config"])
    head = FusionHead(config).to(device)
    head.load_state_dict(payload["head_state_dict"])
    head.eval()
    return head, config, float(payload.get("temperature", 1.0)), payload.get("metadata", {})
