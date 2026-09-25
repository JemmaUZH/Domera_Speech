import argparse
import json
from pathlib import Path

import torch

from speech_sentiment.evaluate import benchmark, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--benchmark", action="store_true", help="Run live whole-pipeline latency on the selected test clips")
    parser.add_argument("--benchmark-limit", type=int, default=200)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    task = checkpoint.get("task", "sentiment")
    data_dir = args.data_dir or (Path("data/emotion") if task == "emotion" else Path("data/processed"))
    manifest = data_dir / f"{args.split}.jsonl"
    output = args.output or args.checkpoint.parent / f"{args.checkpoint.stem}_{args.split}.json"
    report = evaluate(args.checkpoint, manifest, args.cache_dir, output, device=args.device)
    if args.benchmark:
        live = benchmark(args.checkpoint, manifest, device=args.device, limit=args.benchmark_limit)
        output.with_suffix(".benchmark.json").write_text(json.dumps(live, indent=2) + "\n")
        report["benchmark"] = {k: v for k, v in live.items() if k != "per_utterance"}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
