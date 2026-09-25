# Controlled MELD comparison

## Research question

How much sentiment quality does explicit lexical evidence add to an emotion2vec audio pipeline, and can Continuous Mimi offer a useful quality–efficiency trade-off? The primary MELD comparison evaluates **emotion2vec Base audio-only**, **emotion2vec Base + ASR/MiniLM concat**, and **Continuous Mimi**, without changing the fusion architecture or adding conversational context. The separate CREMA-D follow-up is documented in the final report and its own artifact directory; it is not part of the MELD comparison.

## Sampling and training

Use the official MELD train/dev/test boundary. Within each boundary, stratify by the three sentiment labels and sample 1,000 / 200 / 200 items respectively with seed 42. A deterministic same-class reserve fills missing/corrupt media while keeping final class quotas. The manifest and `selection.json` define the experiment, and every system consumes those same IDs. Sample hashes in checkpoints and reports prevent accidental mismatched comparisons. Official speaker IDs are retained, but actors overlap official splits; do not claim speaker-independent results.

Audio is converted to mono 16 kHz WAV. Fusion uses cached local faster-whisper `base.en` transcription and frozen `all-MiniLM-L6-v2` embedding. MELD's reference `Utterance` field is excluded from features and retained for failure review. The audio-only and fusion branches use the same frozen emotion2vec Base frame features and mean pooling. Mimi uses the official `encode_to_latent(..., quantize=False)` path and mean pooling, resampling input to 24 kHz. Each classifier uses a 128-unit GELU MLP with 0.2 dropout and a three-logit output. Train heads with class-weighted cross entropy, fixed seed, and validation macro-F1 selection. Never tune on test labels.

The seven-emotion follow-up keeps these exact selected utterances and frozen features. It maps the official `Emotion` column to anger, disgust, fear, joy, neutral, sadness, and surprise, and uses the same MLP with a seven-logit output. The emotion manifests and model outputs live in `data/emotion/` and `runs/emotion/`, leaving sentiment artifacts intact. Training still uses class-weighted cross entropy and validation macro-F1 selection.

## Quality and cost

For all three systems, report test macro-F1 (primary), accuracy, per-class F1, and confusion matrix. Time raw WAV → final prediction after a warmup pass, batch size one, on the same 200 test clips and hardware. Report per-utterance latency, mean latency, RTF (total inference time / total audio duration), peak process RSS, and known model parameters/state bytes. Fusion stage timing records emotion2vec, ASR, MiniLM, and classifier separately. Its acoustic and ASR/text branches have no dependency until concatenation, so a future implementation could run them concurrently. This V1 times a sequential execution to keep implementation and comparison simple.

The ASR CTranslate2 model does not expose a PyTorch parameter/state count through this implementation, so reported known parameter/state totals exclude ASR for fusion. Peak RSS still includes it. Run each system benchmark in a separate fresh process; `ru_maxrss` is a high-water mark. On the current Apple M1 Pro, Mimi's convolution can fail on MPS for longer clips, so all three live benchmarks use CPU for a comparable hardware setting.

## Failure review

Test predictions and softmax probabilities are generated locally. The small review sheet flags audio-only wrong/fusion correct, the reverse, both wrong, and Mimi/fusion disagreement. Per-utterance text and audio-linked review files are excluded from the release bundle; aggregate counts and confusion matrices are in the result reports.

## Sources

- [Official MELD repository and split description](https://github.com/declare-lab/MELD)
- [emotion2vec Base model card](https://huggingface.co/emotion2vec/emotion2vec_base)
- [Kyutai Mimi continuous latent method](https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/models/compression.py)
