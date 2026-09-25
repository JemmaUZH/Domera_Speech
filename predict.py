import argparse
from pathlib import Path

from speech_sentiment.inference import Predictor, normalized_wav


def main():
    parser = argparse.ArgumentParser(description="Predict sentiment from a WAV file")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/fusion.pt"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    predictor = Predictor(args.checkpoint, args.device)
    with normalized_wav(args.audio) as audio:
        result = predictor.predict(audio)
    print(f"{predictor.task}: {predictor.label_names[result['label']]}")
    print(f"confidence: {result['confidence']:.2f}")


if __name__ == "__main__":
    main()
