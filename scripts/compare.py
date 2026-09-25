"""Compare exactly emotion2vec audio, fusion, and Mimi on identical MELD samples."""

import argparse
import csv
import json
from pathlib import Path


def load_run(report_path: Path):
    report = json.loads(report_path.read_text())
    predictions = [json.loads(line) for line in report_path.with_suffix(".predictions.jsonl").read_text().splitlines()]
    benchmark_path = report_path.with_suffix(".benchmark.json")
    live = json.loads(benchmark_path.read_text()) if benchmark_path.exists() else None
    return report, predictions, live


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["sentiment", "emotion"], default="sentiment")
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--fusion", type=Path)
    parser.add_argument("--mimi", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_dir = Path("runs/emotion") if args.task == "emotion" else Path("runs")
    report_paths = {name: getattr(args, name) or run_dir / f"{name}_test.json" for name in ("audio", "fusion", "mimi")}
    output = args.output or run_dir / "comparison.md"
    runs = {name: load_run(path) for name, path in report_paths.items()}
    reference_report, reference_predictions, reference_live = runs["audio"]
    label_names = reference_report.get("label_names", ["negative", "neutral", "positive"])
    hashes = (reference_report["train_sample_hash"], reference_report["val_sample_hash"], reference_report["test_sample_hash"])
    if any(value is None for value in hashes):
        raise ValueError("Reports lack train/validation/test sample hashes; retrain with the current pipeline")
    ids = [item["id"] for item in reference_predictions]
    for name, (report, predictions, live) in runs.items():
        if report["system"] != name or report["split"] != "test":
            raise ValueError(f"{name} is not a test report for the requested system")
        if report.get("task", "sentiment") != args.task or report.get("label_names", ["negative", "neutral", "positive"]) != label_names:
            raise ValueError("Task or label order differs between systems")
        if (report["train_sample_hash"], report["val_sample_hash"], report["test_sample_hash"]) != hashes:
            raise ValueError("Train/validation/test samples or labels differ between systems")
        if [item["id"] for item in predictions] != ids:
            raise ValueError("Prediction IDs/order differ between systems")
        if len(predictions) != 200:
            raise ValueError(f"Expected exactly 200 MELD test utterances; got {len(predictions)}")
        if live is None:
            raise ValueError(f"Missing live benchmark for {name}")
        if (live["device"], live["batch_size"], [r["id"] for r in live["per_utterance"]]) != (reference_live["device"], reference_live["batch_size"], [r["id"] for r in reference_live["per_utterance"]]):
            raise ValueError("Benchmarks differ in hardware setting, batch size, or clip subset")
    lines = [
        f"# MELD {args.task} controlled comparison", "", "Official splits; same selected 1,000 train / 200 validation / 200 test utterances.", "",
        "| System | Macro-F1 | Weighted-F1 | Accuracy | Mean latency (s) | RTF | Peak RSS (GiB) | Known parameters | Known model state (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("audio", "fusion", "mimi"):
        report, _, live = runs[name]
        weighted_f1 = report["weighted_f1"] if "weighted_f1" in report else report["per_class"].get("weighted avg", {}).get("f1-score", float("nan"))
        lines.append(
            f"| {name} | {report['macro_f1']:.3f} | {weighted_f1:.3f} | {report['accuracy']:.3f} | "
            f"{live['mean_end_to_end_seconds']:.3f} | {live['rtf_total']:.3f} | "
            f"{live['peak_process_rss_bytes'] / 2**30:.3f} | {live['known_model_parameters']:,} | "
            f"{live['known_model_state_bytes'] / 2**20:.1f} |"
        )
    lines.extend(["", "Parameter and model-state totals exclude faster-whisper's CTranslate2 ASR model in fusion. Peak RSS is the process high-water mark measured in separate fresh processes.", "", "## Per-class F1", "", "| System | " + " | ".join(label_names) + " |", "|---|" + "---:|" * len(label_names)])
    for name in ("audio", "fusion", "mimi"):
        report = runs[name][0]
        scores = [report["per_class"][label]["f1-score"] for label in label_names]
        lines.append(f"| {name} | " + " | ".join(f"{score:.3f}" for score in scores) + " |")
    lines.extend(["", "## Confusion matrices", "", "Rows are true labels; columns are predicted labels, ordered " + " / ".join(label_names) + ".", ""])
    for name in ("audio", "fusion", "mimi"):
        lines.append(f"- {name}: `{runs[name][0]['confusion_matrix']}`")
    lines.extend(["", "## Fusion stage latency", "", "Mean seconds per utterance, measured sequentially:", ""])
    for stage, seconds in runs["fusion"][2]["stage_means_seconds"].items():
        lines.append(f"- {stage.replace('_seconds', '').replace('_', ' ')}: {seconds:.3f}")
    lines.extend(["", "## Failure sets", ""])
    grouped = {"audio_wrong_fusion_correct": [], "fusion_wrong_audio_correct": [], "both_wrong": [], "mimi_disagrees_fusion": []}
    failures = []
    for audio_item, fusion_item, mimi_item in zip(*(runs[name][1] for name in ("audio", "fusion", "mimi"))):
        gold = audio_item["gold"]
        if fusion_item["gold"] != gold or mimi_item["gold"] != gold:
            raise ValueError("Gold labels differ across prediction files")
        flags = {
            "audio_wrong_fusion_correct": audio_item["predicted"] != gold and fusion_item["predicted"] == gold,
            "fusion_wrong_audio_correct": fusion_item["predicted"] != gold and audio_item["predicted"] == gold,
            "both_wrong": audio_item["predicted"] != gold and fusion_item["predicted"] != gold,
            "mimi_disagrees_fusion": mimi_item["predicted"] != fusion_item["predicted"],
        }
        for group, matched in flags.items():
            if matched:
                grouped[group].append(audio_item["id"])
        if any(flags.values()):
            asr_path = args.cache_dir / "asr" / "test" / f"{audio_item['id']}.json"
            asr_text = json.loads(asr_path.read_text())["text"] if asr_path.exists() else ""
            failures.append({
                "id": audio_item["id"], "gold": gold, "audio_prediction": audio_item["predicted"],
                "fusion_prediction": fusion_item["predicted"], "mimi_prediction": mimi_item["predicted"],
                **{key: int(value) for key, value in flags.items()},
                "reference_transcript": audio_item.get("reference_transcript") or "", "asr_transcript": asr_text,
                "manual_notes": "",
            })
    for group, values in grouped.items():
        lines.append(f"- {group.replace('_', ' ')}: {len(values)}")
    lines.extend(["", "See the matching `.failure_cases.csv` to select a few representative utterances for manual listening and annotation.", ""])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    csv_path = output.with_suffix(".failure_cases.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["id", "gold", "audio_prediction", "fusion_prediction", "mimi_prediction", *grouped, "reference_transcript", "asr_transcript", "manual_notes"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failures)
    print(f"{output}\n{csv_path}")


if __name__ == "__main__":
    main()
