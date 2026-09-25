"""Deterministic, stratified MELD subsets from extracted clips or MELD.Raw.tar.gz."""

from __future__ import annotations

import csv
import json
import math
import random
import shutil
import subprocess
import tarfile
import tempfile
import wave
from collections import Counter
from pathlib import Path

from . import LABEL_TO_ID
from .meld import CSV_NAMES, convert_clip, find_unique, index_clips

DEFAULT_SIZES = {"train": 1000, "val": 200, "test": 200}


def read_annotations(root: Path, split: str) -> list[dict]:
    path = find_unique(root, CSV_NAMES[split])
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"Dialogue_ID", "Utterance_ID", "Utterance", "Speaker", "Sentiment"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing MELD columns in {path}")
        rows = []
        for source_index, row in enumerate(reader):
            sentiment = row["Sentiment"].strip().lower()
            if sentiment not in LABEL_TO_ID:
                raise ValueError(f"Unknown sentiment: {sentiment}")
            dialogue = int(row["Dialogue_ID"])
            utterance = int(row["Utterance_ID"])
            rows.append({
                "id": f"{split}_dia{dialogue}_utt{utterance}",
                "stem": f"dia{dialogue}_utt{utterance}",
                "transcript": row["Utterance"].strip(),
                "label": LABEL_TO_ID[sentiment],
                "speaker_id": row["Speaker"].strip() or None,
                "split": split,
                "dialogue_id": dialogue,
                "utterance_id": utterance,
                "source_index": source_index,
            })
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate annotation IDs in {path}")
    return rows


def stratified_candidates(rows: list[dict], size: int, seed: int, reserve_per_class: int = 20) -> tuple[dict[int, int], dict[int, list[dict]]]:
    if size <= 0 or size > len(rows):
        raise ValueError(f"Invalid subset size {size} for {len(rows)} MELD rows")
    groups = {label: [row for row in rows if row["label"] == label] for label in range(3)}
    exact = {label: size * len(group) / len(rows) for label, group in groups.items()}
    quotas = {label: math.floor(exact[label]) for label in groups}
    remaining = size - sum(quotas.values())
    for label in sorted(groups, key=lambda item: (-(exact[item] - quotas[item]), item))[:remaining]:
        quotas[label] += 1
    candidates = {}
    for label, group in groups.items():
        shuffled = group.copy()
        random.Random(seed + label * 1009).shuffle(shuffled)
        candidates[label] = shuffled[: min(len(shuffled), quotas[label] + reserve_per_class)]
    return quotas, candidates


def _build_manifest(output_dir: Path, plans: dict, converted: dict, failures: list[dict], seed: int, sizes: dict) -> dict:
    summary = {"dataset": "MELD", "seed": seed, "requested_sizes": sizes, "skipped": failures}
    output_dir.mkdir(parents=True, exist_ok=True)
    retained_paths = set()
    for split in ("train", "val", "test"):
        quotas, candidates = plans[split]
        chosen = []
        for label in range(3):
            good = [row for row in candidates[label] if (split, row["stem"]) in converted]
            if len(good) < quotas[label]:
                raise RuntimeError(f"Only {len(good)}/{quotas[label]} valid {split} class {label} clips; increase --reserve-per-class")
            chosen.extend(good[:quotas[label]])
        chosen.sort(key=lambda row: row["source_index"])
        if len(chosen) != sizes[split]:
            raise AssertionError("Subset size mismatch")
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in chosen:
                audio_path, duration = converted[(split, row["stem"])]
                retained_paths.add(audio_path)
                item = {key: value for key, value in row.items() if key not in ("stem", "source_index")}
                item.update({"audio_path": str(audio_path), "duration_seconds": duration})
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        summary[split] = {"count": len(chosen), "labels": dict(Counter(row["label"] for row in chosen)), "ids": [row["id"] for row in chosen]}
    # A few reserve clips may have been converted but were not needed.
    for audio_path, _ in converted.values():
        if audio_path not in retained_paths:
            audio_path.unlink(missing_ok=True)
            audio_path.with_suffix(audio_path.suffix + ".source.json").unlink(missing_ok=True)
    (output_dir / "selection.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (output_dir / "skipped.json").write_text(json.dumps(failures, indent=2) + "\n")
    return {key: value if key not in ("train", "val", "test") else {"count": value["count"], "labels": value["labels"]} for key, value in summary.items()}


def prepare_subset(source: Path, annotations: Path, output_dir: Path, sizes: dict[str, int], seed: int = 42, reserve_per_class: int = 20) -> dict:
    """Prepare only selected MELD rows; source may be a directory or raw archive."""
    source = source.expanduser().resolve()
    annotations = annotations.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if set(sizes) != {"train", "val", "test"}:
        raise ValueError("Expected train, val, test sizes")
    plans = {split: stratified_candidates(read_annotations(annotations, split), sizes[split], seed + index * 100000, reserve_per_class) for index, split in enumerate(("train", "val", "test"))}
    converted: dict[tuple[str, str], tuple[Path, float]] = {}
    failures = []
    if source.is_dir():
        clips = index_clips(source)
        for split, (_, candidates) in plans.items():
            for group in candidates.values():
                for row in group:
                    key = (split, row["stem"])
                    clip = clips.get(key)
                    if clip is None:
                        failures.append({"split": split, "id": row["stem"], "reason": "missing clip"})
                        continue
                    output = output_dir / "audio" / split / f"{row['stem']}.wav"
                    try:
                        converted[key] = (output, convert_clip(clip, output))
                    except (subprocess.CalledProcessError, ValueError, wave.Error, EOFError) as exc:
                        failures.append({"split": split, "id": row["stem"], "reason": str(exc)})
    elif source.is_file():
        wanted = {split: {row["stem"] for group in candidates.values() for row in group} for split, (_, candidates) in plans.items()}
        seen = {split: set() for split in wanted}
        outer_names = {"train.tar.gz": "train", "dev.tar.gz": "val", "test.tar.gz": "test"}
        with tarfile.open(source, "r|gz") as outer:
            for outer_member in outer:
                split = outer_names.get(Path(outer_member.name).name)
                if split is None or not outer_member.isfile():
                    continue
                print(f"Scanning {split} clips in {source.name}", flush=True)
                stream = outer.extractfile(outer_member)
                if stream is None:
                    raise RuntimeError(f"Cannot read {outer_member.name}")
                with tarfile.open(fileobj=stream, mode="r|gz") as inner:
                    for member in inner:
                        if not member.isfile() or not member.name.endswith(".mp4"):
                            continue
                        stem = Path(member.name).stem
                        if stem not in wanted[split]:
                            continue
                        seen[split].add(stem)
                        if (split, stem) in converted:
                            raise ValueError(f"Duplicate {split}/{stem} in archive")
                        clip_data = inner.extractfile(member)
                        if clip_data is None:
                            failures.append({"split": split, "id": stem, "reason": "unreadable archive member"})
                            continue
                        with tempfile.TemporaryDirectory(prefix="meld-clip-") as directory:
                            clip = Path(directory) / f"{stem}.mp4"
                            with clip.open("wb") as handle:
                                shutil.copyfileobj(clip_data, handle)
                            output = output_dir / "audio" / split / f"{stem}.wav"
                            try:
                                converted[(split, stem)] = (output, convert_clip(clip, output))
                            except (subprocess.CalledProcessError, ValueError, wave.Error, EOFError) as exc:
                                failures.append({"split": split, "id": stem, "reason": str(exc)})
                print(f"Converted {len([key for key in converted if key[0] == split])} {split} candidate clips", flush=True)
        for split in wanted:
            for stem in wanted[split] - seen[split]:
                failures.append({"split": split, "id": stem, "reason": "missing archive member"})
    else:
        raise FileNotFoundError(source)
    return _build_manifest(output_dir, plans, converted, failures, seed, sizes)
