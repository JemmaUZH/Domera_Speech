import argparse
import json
from pathlib import Path

from speech_sentiment.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["audio", "text", "fusion", "mimi"], required=True)
    parser.add_argument("--task", choices=["sentiment", "emotion"], default="sentiment")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args()
    data_dir = args.data_dir or (Path("data/emotion") if args.task == "emotion" else Path("data/processed"))
    output = args.output or (Path("runs/emotion") if args.task == "emotion" else Path("runs")) / f"{args.system}.pt"
    print(json.dumps(train(data_dir, args.cache_dir, output, args.system, device=args.device, seed=args.seed, epochs=args.epochs, patience=args.patience, task=args.task), indent=2))


if __name__ == "__main__":
    main()
