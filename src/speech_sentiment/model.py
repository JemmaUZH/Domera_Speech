"""Small frozen-feature classifier shared by the three experiment arms."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .features import SYSTEM_FEATURES


class SentimentHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.2, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def concatenate(features: dict[str, np.ndarray], system: str) -> np.ndarray:
    return np.concatenate([features[kind] for kind in SYSTEM_FEATURES[system]]).astype(np.float32)


def predict_vector(checkpoint: dict, vector: np.ndarray, device: str = "cpu") -> tuple[int, float, np.ndarray]:
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["scale"], dtype=np.float32)
    if vector.shape != mean.shape:
        raise ValueError(f"Expected feature shape {mean.shape}; got {vector.shape}")
    model = SentimentHead(len(vector), checkpoint["hidden_dim"], checkpoint["dropout"], len(checkpoint.get("label_names", ("negative", "neutral", "positive")))).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.inference_mode():
        tensor = torch.from_numpy((vector - mean) / scale).to(device)[None]
        probabilities = torch.softmax(model(tensor), dim=-1)[0].cpu().numpy()
    label = int(probabilities.argmax())
    return label, float(probabilities[label]), probabilities
