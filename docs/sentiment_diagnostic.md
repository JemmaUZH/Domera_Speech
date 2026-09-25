# MELD Sentiment Diagnostic

This diagnostic re-evaluated the existing checkpoints and trained a text-only MiniLM baseline using the same text embeddings as Emotion2Vec + Text. All systems use the same 200 MELD test utterances. The primary comparison and class distribution are in [the sentiment results report](results.md).

| System | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Emotion2Vec audio-only | 0.445 | 0.430 | 0.450 |
| Text-only MiniLM | 0.485 | 0.488 | 0.483 |
| Emotion2Vec + Text | 0.520 | 0.528 | 0.517 |
| Continuous Mimi | 0.460 | 0.448 | 0.463 |

## Text-only baseline

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Negative | 0.412 | 0.438 | 0.424 | 64 |
| Neutral | 0.594 | 0.427 | 0.497 | 96 |
| Positive | 0.444 | 0.700 | 0.544 | 40 |

Confusion matrix (gold rows, predicted columns; negative / neutral / positive): `[[28, 21, 15], [35, 41, 20], [5, 7, 28]]`.

## Fusion modality check

| Diagnostic | Accuracy | Macro-F1 |
|---|---:|---:|
| Full fusion | 0.520 | 0.528 |
| Audio replaced with its training mean | 0.520 | 0.516 |
| Text replaced with its training mean | 0.445 | 0.444 |
| Audio branch permuted (100 shuffles, mean ± SD) | 0.418 ± 0.033 | 0.417 ± 0.033 |
| Text branch permuted (100 shuffles, mean ± SD) | 0.422 ± 0.027 | 0.418 ± 0.027 |

Both branches affect predictions; the diagnostics show no clear collapse to one modality. Replacing text with its training mean reduced performance more than replacing audio, while permuting either branch reduced both metrics.

## Short text sanity checks

These examples were encoded as text with MiniLM and evaluated by the text-only classifier. They are not audio tests or MELD test examples.

| Text | Prediction | Negative | Neutral | Positive |
|---|---|---:|---:|---:|
| “I absolutely love this!” | Positive | 7.2% | 10.6% | 82.2% |
| “This is terrible.” | Negative | 68.5% | 18.2% | 13.3% |
| “I'm so sad about this.” | Negative | 61.1% | 8.0% | 30.9% |
| “I lost my laptop last week, it's so sad.” | Positive | 43.8% | 7.4% | 48.8% |
| “Yes, I lost my laptop last week, it's so sad.” | Positive | 39.9% | 7.6% | 52.6% |
| “Okay, I understand.” | Neutral | 10.6% | 74.1% | 15.3% |

The laptop example shows that the text branch alone can produce the unexpected positive sentiment. It is outside MELD and has no gold label, so it does not establish a measured model error. Class-weighted training did not eliminate this behavior. The original sentiment checkpoints and test predictions were not modified during this diagnostic.

Machine-readable details are in [`artifacts/final/sentiment_diagnostic.json`](../artifacts/final/sentiment_diagnostic.json).
