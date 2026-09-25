"""Frozen pretrained encoders, loaded only when their experiment is requested."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def _vector(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 2:
        array = array.mean(axis=0)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite 1-D feature vector; got {array.shape}")
    return array


class Emotion2Vec:
    model_id = "iic/emotion2vec_base"

    def __init__(self, device: str = "cpu"):
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise RuntimeError("Install the fusion extra: pip install -e '.[fusion]'") from exc
        self.model = AutoModel(model=self.model_id, device=device, disable_update=True)

    @torch.inference_mode()
    def encode(self, audio_path: Path) -> np.ndarray:
        # FunASR's documented output is a .npy under output_dir. Some versions
        # also return the embedding directly, so accept either representation.
        with tempfile.TemporaryDirectory() as directory:
            result = self.model.generate(
                input=str(audio_path), output_dir=directory,
                granularity="frame", extract_embedding=True,
            )
            for item in result if isinstance(result, list) else [result]:
                if isinstance(item, dict):
                    for key in ("embedding", "embeddings", "feats"):
                        if key in item and item[key] is not None:
                            return _vector(item[key], "emotion2vec")
            files = list(Path(directory).rglob("*.npy"))
            if len(files) != 1:
                raise RuntimeError(f"emotion2vec returned {len(files)} embedding files; check FunASR version")
            return _vector(np.load(files[0]), "emotion2vec")


class LocalASR:
    model_id = "base.en"

    def __init__(self, device: str = "cpu", compute_type: str = "int8"):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("Install the fusion extra: pip install -e '.[fusion]'") from exc
        # CTranslate2/faster-whisper supports CPU and CUDA, but not Apple MPS.
        backend_device = "cpu" if device == "mps" else device
        backend_type = "int8" if backend_device == "cpu" else compute_type
        self.model = WhisperModel(self.model_id, device=backend_device, compute_type=backend_type)

    def transcribe(self, audio_path: Path) -> str:
        segments, _ = self.model.transcribe(str(audio_path), language="en", beam_size=1, vad_filter=False)
        return " ".join(segment.text.strip() for segment in segments).strip()


class TextEncoder:
    model_id = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, device: str = "cpu"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install the fusion extra: pip install -e '.[fusion]'") from exc
        self.model = SentenceTransformer(self.model_id, device=device)

    def encode(self, text: str) -> np.ndarray:
        return _vector(self.model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0], "MiniLM")


class ContinuousMimi:
    model_id = "kyutai/moshiko-pytorch-bf16/tokenizer-e351c8d8-checkpoint125.safetensors"

    def __init__(self, device: str = "cpu"):
        try:
            from huggingface_hub import hf_hub_download
            from moshi.models.loaders import get_mimi
        except ImportError as exc:
            raise RuntimeError("Install the Mimi extra: pip install -e '.[mimi]'") from exc
        self.device = torch.device(device)
        weights = hf_hub_download("kyutai/moshiko-pytorch-bf16", "tokenizer-e351c8d8-checkpoint125.safetensors")
        self.model = get_mimi(weights, device=self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, audio_path: Path) -> np.ndarray:
        from scipy.signal import resample_poly

        audio, rate = sf.read(audio_path, dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        if rate != self.model.sample_rate:
            from math import gcd
            divisor = gcd(rate, self.model.sample_rate)
            mono = resample_poly(mono, self.model.sample_rate // divisor, rate // divisor)
        waveform = torch.from_numpy(np.asarray(mono, dtype=np.float32)).to(self.device)[None, None, :]
        # Explicitly bypass residual vector quantization.
        latent = self.model.encode_to_latent(waveform, quantize=False)
        return _vector(latent[0].mean(dim=-1).float().cpu().numpy(), "continuous Mimi")


def audio_duration(path: Path) -> float:
    return sf.info(path).duration
