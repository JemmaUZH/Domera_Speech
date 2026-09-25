# Speech Emotion and Sentiment Analyzer

This project compares three utterance-level speech systems on MELD. **Three-class sentiment (negative / neutral / positive) is the primary task; seven-class emotion is evaluated separately.** The approach, results, limitations, and next steps are summarized in the [final report](docs/final_report.md).

The project includes source code, six trained task-specific classifier checkpoints, and the fixed sample IDs used for the reported results. It does not include MELD audio, MELD annotation files, pretrained encoder weights, feature caches, working experiment runs, or API credentials. The demo downloads pretrained encoder weights on first use. Dataset evaluation requires downloading MELD and regenerating the feature caches.

## Contents

- [Interactive demo](#interactive-demo): record or upload one utterance and inspect local model predictions.
- [Optional GPT Realtime baseline](#optional-gpt-realtime-api-baseline): enter your own OpenAI API key to compare with the audio-native API system.
- [Reproduce evaluation on MELD](#reproduce-evaluation-on-meld): download the official data, recreate the fixed stratified subset, verify the sample IDs, extract features, and evaluate the included checkpoints.
- [Optional retraining](#optional-retraining): train new classifier heads without changing the included checkpoints.

## Interactive demo

### 1. Install the requirements

Use Python 3.10, 3.11, or 3.12 and install FFmpeg. The project includes [`uv.lock`](uv.lock), which pins the Python dependencies and package hashes. From the project directory, install `uv` and sync the locked demo environment:

```bash
python -m pip install uv
uv sync --extra all --extra demo --locked
```

`uv sync` creates the project environment in `.venv`. Run project commands through `uv run` to use the locked environment. For example, start the demo with `uv run python demo/app.py`. To run the test suite too, use `uv sync --extra all --extra demo --extra test --locked` and then `uv run pytest -q`.

On macOS, FFmpeg can be installed with `brew install ffmpeg`. On Ubuntu or Debian, use `sudo apt install ffmpeg`.

### 2. Start the demo

```bash
uv run python demo/app.py
```

Open the local address printed in the terminal, normally `http://127.0.0.1:7860`. The app listens on your computer only by default; it does not create a public Gradio share link.

If port 7860 is already in use, stop the earlier demo process or choose another port:

```bash
SPEECH_SENTIMENT_PORT=7861 uv run python demo/app.py
```

To choose an inference device, set `SPEECH_SENTIMENT_DEVICE` to `cpu`, `mps`, `cuda`, or `auto`. CPU is the default. The selected device must be supported by your PyTorch installation.

### 3. Record or upload an utterance

In the **Record or upload one utterance** control, either record a short utterance with your microphone or upload an audio file. Provide one utterance per run. Choose one of the local systems in **Model**:

| Model | Input used | What it tests |
| --- | --- | --- |
| Emotion2Vec audio-only | Audio | Emotion2Vec acoustic representation and classifier |
| Emotion2Vec + Text | Audio and ASR text | Emotion2Vec features fused with the text representation |
| Continuous Mimi | Audio | Continuous Mimi representation and classifier |
| Compare all models | The same audio for each selected system | Side-by-side quality and latency inspection |

Click **Analyze audio**. The interactive demo is mainly for trying your own spoken utterances or a few uploaded clips and inspecting predictions. It reports both tasks for each run:

- **3-class sentiment:** negative, neutral, or positive.
- **7-class emotion:** anger, disgust, fear, joy, neutral, sadness, or surprise.

The results table shows each prediction, its confidence, and end-to-end latency. The tabs below it show the model's probabilities across all labels. Emotion2Vec + Text also displays the ASR transcript used by its text branch. Compare all models reuses the same recording for all three local pipelines; it does not ask you to record or upload again. For a systematic, reproducible evaluation on MELD, use the command-line workflow in [Reproduce evaluation on MELD](#reproduce-evaluation-on-meld) instead of manually testing examples in the demo.

The included files in [`artifacts/final/checkpoints/`](artifacts/final/checkpoints/) are the trained sentiment and emotion classifier heads. The larger pretrained feature encoders are downloaded separately on first use and may take time and disk space. The demo analyzes complete utterances, so it is an **interactive, near-real-time utterance-level analyzer**, not a streaming system. Latency depends on the audio duration, device, and whether model weights need to be downloaded or initialized.

## Optional GPT Realtime API baseline

The demo can also send the same audio utterance to the GPT Realtime audio-native baseline. This is optional; the three local systems work without an OpenAI key.

1. Create an API key in your OpenAI account.
2. Paste it into the **OpenAI API key (for GPT Realtime)** password field in the local demo.
3. Select **GPT Realtime** to run that system alone, or select **Compare all models** to include it alongside the three local systems. In compare mode, GPT Realtime is included only when the key field is non-empty.

The key is submitted to the local app backend for the request. The demo code does not save it. The field is a password field, but still treat the key as a secret: do not put it in source code, this README, a screenshot, a shared zip, or a Git commit. Revoke a key if it is exposed. OpenAI's [API key safety guidance](https://developers.openai.com/api/docs/guides/production-best-practices) recommends keeping standard API keys on the server side. This app sends the key from the browser to the local Gradio backend, which makes the API request; do not expose the demo server publicly without adding appropriate authentication and secret handling.

GPT Realtime receives native audio and returns a text emotion/sentiment label. It does not provide the calibrated class-probability distribution shown for local classifiers, so confidence/probabilities are unavailable for this system. API requests may incur charges under your OpenAI account. Its latency includes network and service overhead and should not be directly compared with local inference latency. See the official [Realtime API documentation](https://developers.openai.com/api/docs/guides/realtime) for API details.

## Reproduce evaluation on MELD

The reported evaluation uses the official MELD train/dev/test partitions and a deterministic stratified subset of **1,000 train, 200 validation (official dev), and 200 test utterances**. The included checkpoints are evaluated as-is; the commands below do not train or overwrite them. The fixed selected IDs are listed in [`artifacts/final/split_ids.json`](artifacts/final/split_ids.json).

### 1. Download the official MELD files

MELD's [official repository](https://github.com/declare-lab/MELD) provides the dataset and describes its split files. Create local directories for the archive and annotations:

```bash
mkdir -p data/downloads data/annotations
```

Download the raw audio/video archive, which is about 11 GB, from the dataset authors' [MELD.Raw.tar.gz file](https://huggingface.co/datasets/declare-lab/MELD/blob/main/MELD.Raw.tar.gz):

```bash
curl -fL --retry 3 \
  'https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz?download=true' \
  -o data/downloads/MELD.Raw.tar.gz
```

Download the official emotion/sentiment annotation CSVs from the MELD authors' repository:

```bash
for split in train dev test; do
  curl -fL --retry 3 \
    "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/${split}_sent_emo.csv" \
    -o "data/annotations/${split}_sent_emo.csv"
done
```

The annotation directory must contain exactly these filenames:

```text
data/annotations/train_sent_emo.csv
data/annotations/dev_sent_emo.csv
data/annotations/test_sent_emo.csv
```

If downloading through a browser, save the files under those names. The raw archive contains nested split archives. You do not need to unpack them manually: the preparation script scans the official archive and converts only the selected clips into local 16 kHz WAV files. Keep enough free disk space for the archive and the processed audio.

### 2. Prepare the same subset

If you synced the locked `all` and `demo` extras above, run:

```bash
uv run python scripts/prepare_data.py \
  --meld-archive data/downloads/MELD.Raw.tar.gz \
  --annotations-root data/annotations \
  --output-dir data/processed \
  --train-size 1000 --val-size 200 --test-size 200 --seed 42
```

The script samples within each official split, preserving MELD's train/dev/test separation. It writes `train.jsonl`, `val.jsonl`, and `test.jsonl` under `data/processed/` and converts the selected source clips into `data/processed/audio/`. `val` maps to the official MELD `dev` split. Audio that is missing or cannot be converted is skipped and replaced from a reserve list where possible.

Verify that the regenerated ordered sample IDs are identical to the IDs used for the included results:

```bash
python - <<'PY'
import json
from pathlib import Path

expected = json.loads(Path("artifacts/final/split_ids.json").read_text())["ids"]
for split in ("train", "val", "test"):
    path = Path("data/processed") / f"{split}.jsonl"
    actual = [json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()]
    if actual != expected[split]:
        raise SystemExit(f"{split}: IDs differ from artifacts/final/split_ids.json")
    print(f"{split}: {len(actual)} IDs match")
PY
```

This check should print 1,000, 200, and 200 matching IDs. If an archive is incomplete or a source clip is missing, the regenerated list may differ; use the same official archive and annotation files before evaluating if you need an exact reproduction.

### 3. Extract or reuse features

Extract the features required by each system:

```bash
for system in audio fusion mimi; do
  uv run python scripts/extract_features.py --system "$system"
done
uv run python scripts/prepare_emotion_labels.py
```

Feature extraction downloads or initializes the pretrained encoders when needed and may take substantially longer than scoring. Existing valid entries in `cache/` are reused. The three systems use the same prepared sample manifests.

### 4. Evaluate the included checkpoints

Run both tasks for each system on the fixed test manifest:

```bash
for system in audio fusion mimi; do
  uv run python scripts/evaluate.py \
    --checkpoint "artifacts/final/checkpoints/sentiment/${system}.pt" \
    --output "runs/final/sentiment/${system}_test.json"
  uv run python scripts/evaluate.py \
    --checkpoint "artifacts/final/checkpoints/emotion/${system}.pt" \
    --output "runs/final/emotion/${system}_test.json"
done
```

Each evaluation scores the 200 examples in `data/processed/test.jsonl`. Per-run metrics and predictions are written under `runs/final/`. The aggregate reported results, per-class scores, confusion matrices, and split metadata are in [`artifacts/final/results.json`](artifacts/final/results.json). Additional detail is available in the [sentiment report](docs/results.md), [emotion report](docs/emotion_results.md), and [sentiment diagnostic](docs/sentiment_diagnostic.md).

## Supplementary CREMA-D experiment

The primary assignment result is the MELD sentiment evaluation above. A separate follow-up trained new six-class emotion classifier heads on CREMA-D while keeping the same pretrained encoders frozen. It uses 1,000 train, 200 validation, and the fixed 200-example speaker-disjoint test subset. This is a distinct supervised CREMA-D experiment; it is not part of the MELD test metrics.

| System | Macro-F1 | Weighted-F1 | Accuracy |
| --- | ---: | ---: | ---: |
| Emotion2Vec audio-only | 0.649 | 0.647 | 65.0% |
| Emotion2Vec + Text | 0.643 | 0.643 | 64.5% |
| Continuous Mimi | 0.644 | 0.643 | 64.5% |

The complete metrics, per-class results, predictions, confusion matrices, run configuration, checksums, and CREMA-D-trained classifier heads are included in [`artifacts/crema_d_supervised/`](artifacts/crema_d_supervised/). No CREMA-D audio is included. To recreate this experiment, the preparation scripts download the public metadata and selected audio from the [CREMA-D source repository](https://github.com/CheyneyComputerScience/CREMA-D); follow the dataset's terms when obtaining and using the media.

```bash
uv run python scripts/prepare_crema_d.py --data-dir data/crema_d --count 200 --seed 42
uv run python scripts/prepare_crema_d_supervised.py
uv run python scripts/train_crema_d_supervised.py --system all
```

This writes the generated audio and manifests under `data/crema_d/` and `data/crema_d_supervised/`, feature caches under `cache/crema_d_supervised/`, and new checkpoints/results under `runs/crema_d_supervised/`. The run does not overwrite the included MELD checkpoints or results. MELD-trained cross-dataset scores are added to the comparison only when those separate results are present.

## Optional retraining

To train new classifier heads, first prepare MELD and extract the features as described above, then run:

```bash
for system in audio fusion mimi; do
  uv run python scripts/train.py --system "$system" --task sentiment
  uv run python scripts/train.py --system "$system" --task emotion
done
```

Training uses class-weighted cross-entropy, seed 42, and validation Macro-F1 for checkpoint selection. The test split is not used for model selection. New checkpoints are written under `runs/`; the included checkpoints under `artifacts/final/checkpoints/` are not changed.

## Tests and release contents

Run the unit tests with:

```bash
uv run pytest -q
```

The [`artifacts/speech_sentiment_take_home.zip`](artifacts/speech_sentiment_take_home.zip) bundle contains the application code, documentation, tests, locked dependencies, MELD and CREMA-D classifier checkpoints, and fixed split/result metadata. It does **not** contain MELD or CREMA-D audio/annotations, downloaded pretrained encoder weights, feature caches, API keys, or generated working runs. To use it, extract the zip, follow the locked demo installation steps above, and download the corresponding dataset only if you also want to reproduce its evaluation.
