"""Official MELD CSV/clip mapping. No network access or archive extraction here."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import wave
from collections import Counter
from pathlib import Path

from . import LABEL_TO_ID, TASK_LABELS

CSV_NAMES = {"train": "train_sent_emo.csv", "val": "dev_sent_emo.csv", "test": "test_sent_emo.csv"}
CLIP_DIRS = {"train": "train_splits", "val": "dev_splits_complete", "test": "output_repeated_splits_test"}


def find_unique(root: Path, filename: str) -> Path:
    matches = sorted(root.rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {filename} under {root}; found {len(matches)}")
    return matches[0]


def index_clips(root: Path) -> dict[tuple[str, str], Path]:
    """Index raw clips once; the same dialogue/utterance IDs recur across splits."""
    index = {}
    for path in root.rglob("*.mp4"):
        parts = set(path.parts)
        candidates = [split for split, directory in CLIP_DIRS.items() if directory in parts]
        if len(candidates) != 1:
            continue
        key = (candidates[0], path.stem)
        if key in index:
            raise ValueError(f"Duplicate MELD clip {key}: {index[key]} and {path}")
        index[key] = path
    return index


def convert_clip(clip: Path, output: Path, ffmpeg: str = "ffmpeg") -> float:
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reuse a converted WAV only when its source archive member is unchanged.
    # Older WAVs without a sidecar are rebuilt once to establish provenance.
    source_digest = hashlib.sha256()
    with clip.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            source_digest.update(block)
    source_hash = source_digest.hexdigest()
    metadata = output.with_suffix(output.suffix + ".source.json")
    reusable = False
    if output.is_file() and metadata.is_file():
        try:
            reusable = json.loads(metadata.read_text(encoding="utf-8")).get("source_sha256") == source_hash
        except (OSError, json.JSONDecodeError):
            reusable = False
    if not reusable:
        tmp = output.with_suffix(".tmp.wav")
        try:
            subprocess.run(
                [ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(clip), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(tmp)],
                check=True,
            )
            tmp.replace(output)
            metadata_tmp = metadata.with_suffix(metadata.suffix + ".tmp")
            metadata_tmp.write_text(json.dumps({"source_sha256": source_hash}) + "\n", encoding="utf-8")
            metadata_tmp.replace(metadata)
        finally:
            tmp.unlink(missing_ok=True)
    with wave.open(str(output), "rb") as wav:
        if wav.getframerate() != 16000 or wav.getnchannels() != 1 or wav.getnframes() == 0:
            raise ValueError(f"Invalid converted audio: {output}")
        return wav.getnframes() / wav.getframerate()


def prepare(meld_root: Path, output_dir: Path, *, limit_per_split: int | None = None, strict: bool = False) -> dict:
    """Create a MELD-only JSONL manifest and local 16 kHz WAVs."""
    if limit_per_split is not None and limit_per_split <= 0:
        raise ValueError("limit_per_split must be positive")
    meld_root = meld_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not meld_root.is_dir():
        raise FileNotFoundError(meld_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    clips = index_clips(meld_root)
    if not clips:
        raise FileNotFoundError(f"No extracted MELD mp4 clips found under {meld_root}")
    summary = {}
    skipped = []
    for split, filename in CSV_NAMES.items():
        csv_path = find_unique(meld_root, filename)
        rows = []
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            needed = {"Dialogue_ID", "Utterance_ID", "Utterance", "Speaker", "Sentiment"}
            if not needed.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing MELD columns in {csv_path}: {sorted(needed - set(reader.fieldnames or []))}")
            for row in reader:
                label = row["Sentiment"].strip().lower()
                if label not in LABEL_TO_ID:
                    raise ValueError(f"Unexpected MELD sentiment {label!r}")
                stem = f"dia{int(row['Dialogue_ID'])}_utt{int(row['Utterance_ID'])}"
                try:
                    clip = clips[(split, stem)]
                    audio = output_dir / "audio" / split / f"{stem}.wav"
                    duration = convert_clip(clip, audio)
                except (KeyError, subprocess.CalledProcessError, ValueError, wave.Error, EOFError) as exc:
                    if strict:
                        raise RuntimeError(f"Cannot prepare {split}/{stem}") from exc
                    skipped.append({"split": split, "id": stem, "reason": str(exc)})
                    continue
                rows.append({
                    "id": f"{split}_{stem}",
                    "audio_path": str(audio),
                    "transcript": row["Utterance"].strip(),  # reference only; ASR drives the model
                    "label": LABEL_TO_ID[label],
                    "speaker_id": row["Speaker"].strip() or None,
                    "split": split,
                    "duration_seconds": duration,
                    "dialogue_id": int(row["Dialogue_ID"]),
                    "utterance_id": int(row["Utterance_ID"]),
                })
                if limit_per_split is not None and len(rows) >= limit_per_split:
                    break
        if not rows:
            raise ValueError(f"Empty split: {csv_path}")
        ids = [r["id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate MELD IDs in {split}")
        manifest = output_dir / f"{split}.jsonl"
        with manifest.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary[split] = {"count": len(rows), "labels": dict(Counter(r["label"] for r in rows))}
    summary["skipped"] = skipped
    (output_dir / "skipped.json").write_text(json.dumps(skipped, indent=2) + "\n")
    return summary


def read_manifest(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    required = {"id", "audio_path", "label", "split", "duration_seconds"}
    for row in rows:
        labels = TASK_LABELS.get(row.get("task", "sentiment"))
        if not required.issubset(row) or labels is None or row["label"] not in range(len(labels)):
            raise ValueError(f"Invalid manifest row: {row}")
    return rows
