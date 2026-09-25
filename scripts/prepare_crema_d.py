"""Prepare a fixed, stratified 200-example CREMA-D test subset.

The speaker-exclusive split follows the public TensorFlow Datasets CREMA-D
builder (RandomState(0), 70/10/20 by speaker). Only the selected test audio is
downloaded; MELD assets and manifests are never touched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

REPO = "https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master/processedResults/summaryTable.csv"
WAV_BASE = "https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/AudioWAV/"
MISSING = {"FileName", "1040_ITH_SAD_XX", "1006_TIE_NEU_XX", "1013_WSI_DIS_XX", "1017_IWW_FEA_XX"}
CODE_TO_LABEL = {"ANG": "anger", "DIS": "disgust", "FEA": "fear", "HAP": "joy", "NEU": "neutral", "SAD": "sadness"}
LABELS = tuple(CODE_TO_LABEL.values())


def read_summary(path: Path) -> tuple[list[dict], str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    files = []
    for row in rows:
        name = (row.get("FileName") or "").strip()
        if not name or name in MISSING:
            continue
        parts = name.split("_")
        if len(parts) != 4 or parts[2] not in CODE_TO_LABEL:
            continue
        files.append({
            "id": name,
            "speaker_id": parts[0],
            "sentence_code": parts[1],
            "emotion_code": parts[2],
            "intensity_code": parts[3],
            "label": CODE_TO_LABEL[parts[2]],
            "voice_vote": (row.get("VoiceVote") or "").strip(),
        })
    return files, digest


def official_test_speakers(files: list[dict]) -> set[str]:
    speakers = sorted({row["speaker_id"] for row in files})
    rng = np.random.RandomState(0)
    rng.shuffle(speakers)
    start, end = int(0.8 * len(speakers)), len(speakers)
    return set(speakers[start:end])


def stratified_fixed_subset(rows: list[dict], count: int, seed: int) -> list[dict]:
    if count > len(rows):
        raise ValueError(f"Requested {count} rows from only {len(rows)} official test samples")
    by_label = {label: [r for r in rows if r["label"] == label] for label in LABELS}
    exact = {label: count * len(group) / len(rows) for label, group in by_label.items()}
    allocations = {label: math.floor(value) for label, value in exact.items()}
    remainder = count - sum(allocations.values())
    for label in sorted(LABELS, key=lambda k: (-(exact[k] - allocations[k]), LABELS.index(k)))[:remainder]:
        allocations[label] += 1
    rng = random.Random(seed)
    selected = []
    for label in LABELS:
        group = sorted(by_label[label], key=lambda r: r["id"])
        selected.extend(rng.sample(group, allocations[label]))
    return sorted(selected, key=lambda r: r["id"])


def download_one(row: dict, audio_dir: Path, attempts: int = 5) -> dict:
    name = row["id"]
    destination = audio_dir / f"{name}.wav"
    if destination.exists() and destination.stat().st_size > 44:
        return {**row, "audio_path": str(destination.resolve()), "audio_sha256": sha256(destination)}
    url = WAV_BASE + name + ".wav"
    audio_dir.mkdir(parents=True, exist_ok=True)
    error = None
    for attempt in range(attempts):
        temp = destination.with_suffix(f".part{os.getpid()}")
        try:
            request = Request(url, headers={"User-Agent": "speech-sentiment-meld/1.0"})
            with urlopen(request, timeout=45) as response, temp.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            if temp.stat().st_size <= 44:
                raise IOError(f"Downloaded empty or invalid WAV: {url}")
            temp.replace(destination)
            return {**row, "audio_path": str(destination.resolve()), "audio_sha256": sha256(destination)}
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            error = exc
            temp.unlink(missing_ok=True)
            if isinstance(exc, HTTPError) and exc.code not in (408, 425, 429, 500, 502, 503, 504):
                break
            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"Failed to download {name}: {error}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/crema_d"))
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--summary-csv", type=Path, help="Use a local official summaryTable.csv instead of downloading it")
    args = parser.parse_args()
    if args.count != 200:
        raise ValueError("This experiment is fixed to exactly 200 CREMA-D test examples")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary_csv or args.data_dir / "summaryTable.csv"
    if not summary_path.exists():
        request = Request(REPO, headers={"User-Agent": "speech-sentiment-meld/1.0"})
        with urlopen(request, timeout=45) as response:
            summary_path.write_bytes(response.read())
    all_rows, summary_hash = read_summary(summary_path)
    test_speakers = official_test_speakers(all_rows)
    official_test = [r for r in all_rows if r["speaker_id"] in test_speakers]
    chosen = stratified_fixed_subset(official_test, args.count, args.seed)
    audio_dir = args.data_dir / "audio" / "test"
    completed: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, row, audio_dir): row["id"] for row in chosen}
        for future in as_completed(futures):
            row = future.result()
            completed[row["id"]] = row
            print(f"downloaded {len(completed)}/{len(chosen)}: {row['id']}", flush=True)
    manifest_rows = []
    for row in chosen:
        item = completed[row["id"]]
        item.update({"dataset": "CREMA-D", "split": "test", "task": "emotion", "duration_seconds": None})
        manifest_rows.append(item)
    manifest = args.data_dir / "test.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.data_dir / "subset_config.json").write_text(json.dumps({
        "dataset": "CREMA-D", "official_source": "https://github.com/CheyneyComputerScience/CREMA-D",
        "split_protocol": "TensorFlow Datasets crema_d 1.0.0 speaker-disjoint split: sorted speaker IDs, NumPy RandomState(0) shuffle, first 70% train, next 10% validation, final 20% test",
        "official_test_count": len(official_test), "official_test_speaker_count": len(test_speakers),
        "selected_count": len(manifest_rows), "selection": "stratified random sample without replacement from official test split",
        "selection_seed": args.seed, "summary_table_sha256": summary_hash,
        "label_mapping": CODE_TO_LABEL, "selected_class_counts": dict(Counter(r["label"] for r in manifest_rows)),
        "selected_speakers": sorted({r["speaker_id"] for r in manifest_rows}),
        "input_audio": "official AudioWAV clips; original downloaded WAV files retained",
    }, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest), "selected_count": len(manifest_rows), "class_counts": dict(Counter(r["label"] for r in manifest_rows))}, indent=2))


if __name__ == "__main__":
    main()
