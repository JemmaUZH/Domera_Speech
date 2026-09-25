# Speech Sentiment Analyzer ｜ Jiameng Zhang

## Approach and Design Motivation

Recent work on Multimodal Emotion Recognition (MER) shows that emotion is expressed through both language and speech, and these signals do not always agree. Inspired by emotion2vec, EMART, and Lindevelt et al. (EACL 2026), I compared audio-only, audio + text, and audio-native approaches.

I chose MELD as the main dataset because it provides audio, transcripts, sentiment labels, and fine-grained emotion labels from multi-speaker conversations. This allowed me to compare different modalities on the same data and evaluate both 3-class sentiment and 7-class emotion recognition.

### Emotion2Vec Audio-only

I used Emotion2Vec Base as the audio baseline because it is pretrained specifically for emotion representation and is relatively lightweight at around 90M parameters.

### Emotion2Vec + Text

I added a text branch to capture semantic information that may complement tone, prosody, and other acoustic cues. This design was motivated by Lindevelt et al. and EMART, which study the complementary roles of speech and text in emotion recognition.

### Continuous Mimi

Traditional Speech -> ASR -> Text pipelines can lose information such as intonation, pauses, and stress, while also adding ASR latency. Inspired by audio-native and full-duplex speech systems, I tested Continuous Mimi to see whether a continuous audio representation could retain useful emotional information with lower latency.

### GPT Realtime

I also tested GPT Realtime qualitatively as a larger audio-native reference, to see how an end-to-end audio model handles cases that are difficult for the smaller local models.

## Key Trade-offs

The main trade-off was accuracy versus latency.

On 3-class sentiment classification, Audio + Text achieved the best Macro-F1 (0.528), but required 0.847 s per utterance in my local benchmark. Continuous Mimi was less accurate (0.448 Macro-F1) but much faster (0.097 s) and does not require a separate ASR and text-encoding pipeline.

For a real-time conversational system, I would therefore prioritize Continuous Mimi as the deployment direction, while keeping Audio + Text as the stronger accuracy-oriented baseline.

## Evaluation

I evaluated the local models on a fixed MELD split of 1,000 training, 200 validation, and 200 test utterances. Because the classes are imbalanced, I used Macro-F1 as the main metric, together with accuracy and per-utterance latency.

| Model | Macro-F1 | Accuracy | Latency |
|---|---:|---:|---:|
| Emotion2Vec | 0.430 | 44.5% | 0.144 s |
| Emotion2Vec + Text | 0.528 | 52.0% | 0.847 s |
| Continuous Mimi | 0.448 | 46.0% | 0.097 s |

I also tested 7-class emotion recognition as a harder task. Audio + Text reached 0.262 Macro-F1, Continuous Mimi 0.250, and Emotion2Vec 0.179. Here, Continuous Mimi came much closer to Audio + Text while remaining substantially faster.

I then tested the representations on speaker-independent CREMA-D. After training lightweight classification heads on CREMA-D, their performance was very similar: Emotion2Vec reached 0.649, Audio + Text 0.643, and Continuous Mimi 0.644 Macro-F1. This suggests that the value of the text branch depends on the dataset and type of speech.

Finally, I built an interactive demo and tested the pipeline with my own recorded speech. This let me check how the models behave on real microphone input, including examples where the words and tone give different emotional signals.

## Known Limitations and Failure Modes

1. Limited evaluation data. The main evaluation uses a small subset of MELD, so the results may not fully represent real-world conversations.
2. Performance is still limited. The best sentiment model reached 52.0% accuracy / 0.528 Macro-F1. The harder 7-class emotion task reached only 39.0% accuracy / 0.262 Macro-F1, so there is still substantial room for improvement.
3. Neutral sentiment is difficult. Audio + Text classified 38 of 96 neutral examples as negative. The model therefore sometimes treats weak or ambiguous emotional signals as negative instead of neutral.
4. Sentiment and emotion outputs can disagree. They are produced by separately trained classifiers, with no constraint forcing them to be consistent. I also saw unexpected predictions when testing real audio in the demo. For 'I lost my laptop ... it's so sad,' the text-only sentiment branch assigned 48.8% to positive and 43.8% to negative, despite the clearly negative wording. This shows that individual branches can still make counter-intuitive predictions.

## What I Would Explore Next

1. Streaming inference. Inspired by EmoS, I would extend Continuous Mimi to process audio as it arrives and track emotion changes while the user is speaking, instead of waiting for a complete utterance.
2. Conversation context. Simply adding previous dialogue turns reduced performance in my experiment. Inspired by HiRoC's selective history routing, I would instead test speaker-aware and selective context, so the model only uses previous turns that are useful for the current prediction.
3. Audio-native foundation models. Due to API and resource constraints, I only tested GPT Realtime qualitatively on a few recordings. It correctly handled some examples that were difficult for the smaller models. With more resources, I would evaluate audio-native foundation models on the same benchmark and compare their accuracy, latency, and robustness with the local models.

## References

- Ma et al. (2024). emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation. Findings of ACL 2024.
- Lindevelt et al. (2026). The Correlation Between Emotion in Text and Speech Segments is Limited. Findings of EACL 2026.
- Seong et al. (2026). EMART: Emotion-Wheel-Guided Audio-Referred Text Representation. ACL 2026.
- Kim et al. (2026). Aligning Paralinguistic Understanding and Generation in Speech LLMs via Multi-Task Reinforcement Learning. EACL Industry 2026.
- Guo et al. (2026). EmoS: Fine-grained Streaming Emotional Understanding. ACL 2026.
- HiRoC (2026). Selective history routing and disagreement-aware calibration.
- Poria et al. (2019). MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations. ACL 2019.
