"""End-to-end inference; the fused system always runs local ASR."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from . import TASK_LABELS
from .encoders import ContinuousMimi, Emotion2Vec, LocalASR, TextEncoder
from .model import SentimentHead, concatenate


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


class Predictor:
    def __init__(self, checkpoint_path: Path, device: str = "cpu", *, load_encoders: bool = True):
        self.checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.system = self.checkpoint["system"]
        self.task = self.checkpoint.get("task", "sentiment")
        self.label_names = tuple(self.checkpoint.get("label_names", TASK_LABELS[self.task]))
        self.device = device
        self.mean = np.asarray(self.checkpoint["mean"], dtype=np.float32)
        self.scale = np.asarray(self.checkpoint["scale"], dtype=np.float32)
        self.head = SentimentHead(len(self.mean), self.checkpoint["hidden_dim"], self.checkpoint["dropout"], len(self.label_names)).to(device)
        self.head.load_state_dict(self.checkpoint["state_dict"])
        self.head.eval()
        self.audio = self.asr = self.text = self.mimi = None
        self.encoders_loaded = load_encoders
        if load_encoders:
            if self.system in ("audio", "fusion"):
                self.audio = Emotion2Vec(device)
            if self.system in ("text", "fusion"):
                self.asr = LocalASR(device, "int8" if device == "cpu" else "float16")
                self.text = TextEncoder(device)
            if self.system == "mimi":
                self.mimi = ContinuousMimi(device)

    def predict(self, audio_path: Path) -> dict:
        if not self.encoders_loaded:
            raise RuntimeError("This predictor contains only a classifier head; load encoders to predict from audio")
        total_start = time.perf_counter()
        stages = {}
        vectors: dict[str, np.ndarray] = {}
        transcript = None
        if self.audio is not None:
            start = time.perf_counter()
            vectors["audio"] = self.audio.encode(audio_path)
            synchronize(self.device)
            stages["emotion2vec_seconds"] = time.perf_counter() - start
        if self.asr is not None and self.text is not None:
            start = time.perf_counter()
            transcript = self.asr.transcribe(audio_path)
            synchronize(self.device)
            stages["asr_seconds"] = time.perf_counter() - start
            start = time.perf_counter()
            vectors["text"] = self.text.encode(transcript)
            synchronize(self.device)
            stages["text_seconds"] = time.perf_counter() - start
        if self.mimi is not None:
            start = time.perf_counter()
            vectors["mimi"] = self.mimi.encode(audio_path)
            synchronize(self.device)
            stages["mimi_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        vector = concatenate(vectors, self.system)
        if vector.shape != self.mean.shape:
            raise ValueError(f"Expected feature shape {self.mean.shape}; got {vector.shape}")
        with torch.inference_mode():
            normalized = torch.from_numpy((vector - self.mean) / self.scale).to(self.device)[None]
            probabilities = torch.softmax(self.head(normalized), dim=-1)[0].cpu().numpy()
        label = int(probabilities.argmax())
        confidence = float(probabilities[label])
        synchronize(self.device)
        stages["head_seconds"] = time.perf_counter() - start
        stages["end_to_end_seconds"] = time.perf_counter() - total_start
        return {"label": label, "confidence": confidence, "probabilities": probabilities.tolist(), "asr_transcript": transcript, "timings": stages}


def normalized_wav(path: Path):
    """Context manager for an input WAV normalized to mono 16 kHz PCM."""
    from contextlib import contextmanager
    import subprocess

    @contextmanager
    def context():
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "input.wav"
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(path), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(output)], check=True)
            yield output

    return context()
