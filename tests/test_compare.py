import csv
import json
import subprocess
import sys
from pathlib import Path


def test_compare_checks_same_samples_and_saves_failure_sets(tmp_path):
    reports = {}
    ids = [f"test_{i}" for i in range(200)]
    for system in ("audio", "fusion", "mimi"):
        path = tmp_path / f"{system}_test.json"
        reports[system] = path
        report = {"system": system, "split": "test", "macro_f1": 0.5, "accuracy": 0.5,
                  "train_sample_hash": "train", "val_sample_hash": "val", "test_sample_hash": "test",
                  "per_class": {label: {"f1-score": 0.5} for label in ("negative", "neutral", "positive")},
                  "confusion_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
        path.write_text(json.dumps(report))
        with path.with_suffix(".predictions.jsonl").open("w") as handle:
            for i, ident in enumerate(ids):
                predicted = 0
                if (system, i) in (("audio", 0), ("fusion", 1), ("audio", 2), ("fusion", 2), ("mimi", 3)):
                    predicted = 1
                handle.write(json.dumps({"id": ident, "gold": 0, "predicted": predicted, "probabilities": [0.7, 0.2, 0.1], "reference_transcript": "reference"}) + "\n")
        benchmark = {"device": "cpu", "batch_size": 1, "mean_end_to_end_seconds": 1.0, "rtf_total": 0.5,
                     "peak_process_rss_bytes": 2**30, "known_model_parameters": 100, "known_model_state_bytes": 400,
                     "per_utterance": [{"id": ident} for ident in ids],
                     "stage_means_seconds": {"end_to_end_seconds": 1.0}}
        path.with_suffix(".benchmark.json").write_text(json.dumps(benchmark))
    output = tmp_path / "comparison.md"
    script = Path(__file__).resolve().parents[1] / "scripts" / "compare.py"
    command = [sys.executable, str(script), "--audio", str(reports["audio"]), "--fusion", str(reports["fusion"]), "--mimi", str(reports["mimi"]), "--output", str(output)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    with output.with_suffix(".failure_cases.csv").open(newline="") as handle:
        rows = {row["id"]: row for row in csv.DictReader(handle)}
    assert rows["test_0"]["audio_wrong_fusion_correct"] == "1"
    assert rows["test_1"]["fusion_wrong_audio_correct"] == "1"
    assert rows["test_2"]["both_wrong"] == "1"
    assert rows["test_3"]["mimi_disagrees_fusion"] == "1"
    reports["mimi"].write_text(json.dumps({**json.loads(reports["mimi"].read_text()), "train_sample_hash": "different"}))
    failed = subprocess.run(command, capture_output=True, text=True)
    assert failed.returncode != 0
    assert "samples or labels differ" in failed.stderr
