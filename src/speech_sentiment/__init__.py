"""MELD-only speech sentiment experiments."""

LABELS = ("negative", "neutral", "positive")
LABEL_TO_ID = {name: index for index, name in enumerate(LABELS)}
EMOTION_LABELS = ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise")
TASK_LABELS = {"sentiment": LABELS, "emotion": EMOTION_LABELS}
