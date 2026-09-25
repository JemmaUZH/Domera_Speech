from pathlib import Path

import pytest

from speech_sentiment import EMOTION_LABELS, LABELS
from speech_sentiment.realtime import (
    MAX_OUTPUT_TOKENS,
    PROMPT,
    RealtimeClassificationError,
    _response_failure_message,
    classify_audio,
    parse_classifications,
)


def test_incomplete_response_error_includes_server_reason():
    message = _response_failure_message({
        "status": "incomplete",
        "status_details": {"reason": "max_output_tokens"},
    })
    assert "incomplete" in message
    assert "max_output_tokens" in message


def test_realtime_output_budget_has_room_for_tool_completion():
    assert MAX_OUTPUT_TOKENS == 256


def test_prompt_requests_independent_exact_labels_and_json():
    assert "sentiment: exactly one of [negative, neutral, positive]" in PROMPT
    assert "emotion: exactly one of [anger, disgust, fear, joy, neutral, sadness, surprise]" in PROMPT
    assert "Do not mechanically map emotion to sentiment" in PROMPT
    assert "original speech audio directly for both tasks" in PROMPT
    assert "Return JSON only" in PROMPT


def test_parse_classifications_maps_to_project_label_orders():
    parsed = parse_classifications('{"sentiment":"positive","emotion":"sadness"}')
    assert parsed["3-class sentiment"] == {"label": 2, "label_text": "positive"}
    assert parsed["7-class emotion"] == {"label": 5, "label_text": "sadness"}
    assert tuple(LABELS) == ("negative", "neutral", "positive")
    assert tuple(EMOTION_LABELS) == ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise")


@pytest.mark.parametrize("raw", [
    "not json",
    '{"sentiment":"positive","emotion":"angry"}',
    '{"sentiment":"positive","emotion":"sadness","extra":"value"}',
    '{"sentiment":"sadness","emotion":"sadness"}',
])
def test_parse_classifications_rejects_invalid_or_extra_output(raw):
    with pytest.raises(RealtimeClassificationError):
        parse_classifications(raw)


def test_realtime_classifier_preserves_raw_json_and_returns_both_without_fake_probabilities(monkeypatch):
    raw = '{"sentiment":"negative","emotion":"sadness"}'

    async def fake_request(audio, api_key, prompt):
        assert audio == b"pcm"
        assert api_key == "secret"
        assert prompt == PROMPT
        return raw

    monkeypatch.setattr("speech_sentiment.realtime._pcm24k", lambda path: b"pcm")
    monkeypatch.setattr("speech_sentiment.realtime._request", fake_request)
    result = classify_audio(Path("sample.wav"), "secret")
    assert result["raw_response"] == raw
    assert result["predictions"]["3-class sentiment"]["label"] == 0
    assert result["predictions"]["7-class emotion"]["label"] == 5
    assert result["predictions"]["3-class sentiment"]["raw_response"] == raw
    assert result["predictions"]["7-class emotion"]["raw_response"] == raw
    assert result["predictions"]["3-class sentiment"]["probabilities"] is None
    assert result["predictions"]["7-class emotion"]["confidence"] is None
