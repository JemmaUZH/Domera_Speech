"""OpenAI Realtime audio-native, text-output classification for the demo."""

from __future__ import annotations

import asyncio
import base64
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from . import EMOTION_LABELS, LABELS

MODEL = "gpt-realtime-2.1"
MAX_OUTPUT_TOKENS = 256

PROMPT = """You are performing speech affect analysis.

Analyze the supplied speech audio using both what is said and how it is said, including lexical meaning, tone, prosody, pitch, intensity, rhythm, hesitation, and speaking style.

Return:
1. sentiment: exactly one of [negative, neutral, positive]
2. emotion: exactly one of [anger, disgust, fear, joy, neutral, sadness, surprise]

Treat sentiment and emotion as related but separate tasks. Do not mechanically map emotion to sentiment. Determine sentiment independently from the complete utterance rather than applying a fixed emotion-to-sentiment lookup. Analyze the original speech audio directly for both tasks.

Allowed sentiment labels: negative, neutral, positive.
Allowed emotion labels: anger, disgust, fear, joy, neutral, sadness, surprise.

Return JSON only, with exactly these two keys and one valid label in each field. For example:
{\"sentiment\":\"negative\",\"emotion\":\"sadness\"}

The first character of your entire response must be { and the last character must be }. Do not add a leading >, quotation mark, Markdown, code fence, or commentary."""
CLASSIFICATION_TOOL = {
    "type": "function",
    "name": "classify_speech_affect",
    "description": "Return the independently classified sentiment and emotion labels for the supplied speech audio.",
    "parameters": {
        "type": "object",
        "properties": {
            "sentiment": {"type": "string", "enum": list(LABELS)},
            "emotion": {"type": "string", "enum": list(EMOTION_LABELS)},
        },
        "required": ["sentiment", "emotion"],
        "additionalProperties": False,
    },
}


class RealtimeClassificationError(RuntimeError):
    """The model response could not be mapped safely to both project tasks."""


def _response_failure_message(response: dict) -> str:
    """Keep the Realtime completion reason visible instead of hiding it as a status."""
    status = response.get("status", "unknown")
    details = response.get("status_details")
    if not isinstance(details, dict):
        return f"GPT Realtime response ended with status {status!r}."

    reason = details.get("reason")
    error = details.get("error")
    parts = []
    if reason:
        parts.append(f"reason={reason!r}")
    if isinstance(error, dict):
        code = error.get("code")
        message = error.get("message")
        if code:
            parts.append(f"code={code!r}")
        if message:
            parts.append(f"message={message!r}")
    suffix = f" ({'; '.join(parts)})" if parts else f" (details={details!r})"
    return f"GPT Realtime response ended with status {status!r}{suffix}."


def parse_classifications(raw: str) -> dict[str, dict]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RealtimeClassificationError(f"GPT Realtime returned invalid JSON: {exc.msg}.") from exc
    if not isinstance(payload, dict) or set(payload) != {"sentiment", "emotion"}:
        raise RealtimeClassificationError("GPT Realtime JSON must contain exactly 'sentiment' and 'emotion'.")
    sentiment = payload["sentiment"]
    emotion = payload["emotion"]
    if not isinstance(sentiment, str) or sentiment not in LABELS:
        raise RealtimeClassificationError(f"GPT Realtime returned an invalid sentiment label: {sentiment!r}.")
    if not isinstance(emotion, str) or emotion not in EMOTION_LABELS:
        raise RealtimeClassificationError(f"GPT Realtime returned an invalid emotion label: {emotion!r}.")
    return {
        "3-class sentiment": {"label": LABELS.index(sentiment), "label_text": sentiment},
        "7-class emotion": {"label": EMOTION_LABELS.index(emotion), "label_text": emotion},
    }


def _pcm24k(audio_path: Path) -> bytes:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for OpenAI Realtime audio input")
    with tempfile.NamedTemporaryFile(suffix=".pcm") as output:
        result = subprocess.run(
            [ffmpeg, "-nostdin", "-y", "-v", "error", "-i", str(audio_path), "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "24000", output.name],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(f"Audio conversion failed: {result.stderr.strip()}")
        output.seek(0)
        return output.read()


async def _request(audio: bytes, api_key: str, prompt: str) -> str:
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError("Install the GPT Realtime dependency with: pip install -e '.[gpt-realtime]'") from exc

    url = f"wss://api.openai.com/v1/realtime?model={MODEL}"
    async with websockets.connect(
        url,
        additional_headers={"Authorization": f"Bearer {api_key}"},
        open_timeout=30,
        close_timeout=10,
        max_size=32 * 1024 * 1024,
    ) as socket:
        await socket.send(json.dumps({
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": prompt,
                "output_modalities": ["text"],
                "tools": [CLASSIFICATION_TOOL],
                "tool_choice": {"type": "function", "name": "classify_speech_affect"},
                "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}, "turn_detection": None}},
            },
        }))
        while True:
            event = json.loads(await asyncio.wait_for(socket.recv(), timeout=45))
            if event.get("type") == "error":
                raise RuntimeError(f"Realtime API error: {event.get('error', {}).get('message', event)}")
            if event.get("type") == "session.updated":
                break

        await socket.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(audio).decode("ascii")}))
        await socket.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await socket.send(json.dumps({
            "type": "response.create",
            "response": {
                "output_modalities": ["text"],
                # The tool arguments are short, but Realtime can spend output
                # budget before finishing the function call. Leave headroom so
                # normal classifications do not get truncated.
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "tool_choice": {"type": "function", "name": "classify_speech_affect"},
            },
        }))

        text_parts: list[str] = []
        while True:
            event = json.loads(await asyncio.wait_for(socket.recv(), timeout=90))
            kind = event.get("type")
            if kind == "error":
                raise RuntimeError(f"Realtime API error: {event.get('error', {}).get('message', event)}")
            if kind == "response.output_text.delta":
                text_parts.append(event.get("delta", ""))
            elif kind == "response.output_text.done" and not text_parts:
                text_parts.append(event.get("text", ""))
            elif kind == "response.done":
                response = event.get("response", {})
                if response.get("status") != "completed":
                    raise RealtimeClassificationError(_response_failure_message(response))
                for item in response.get("output", []):
                    if item.get("type") == "function_call" and item.get("name") == "classify_speech_affect":
                        return item.get("arguments", "")
                    for content in item.get("content", []):
                        if content.get("type") in ("text", "output_text"):
                            text_parts.append(content.get("text", ""))
                if text_parts:
                    return "".join(text_parts)
                raise RealtimeClassificationError("GPT Realtime completed without returning the required classification JSON.")


def classify_audio(audio_path: Path, api_key: str) -> dict:
    """Predict sentiment and emotion independently from one native audio request."""
    started = time.perf_counter()
    audio = _pcm24k(audio_path)
    raw = asyncio.run(_request(audio, api_key.strip(), PROMPT))
    latency = time.perf_counter() - started
    parsed = parse_classifications(raw)
    for task_result in parsed.values():
        task_result.update({
            "raw_response": raw,
            "confidence": None,
            "probabilities": None,
            "asr_transcript": None,
            "timings": {"end_to_end_seconds": latency},
        })
    return {"raw_response": raw, "predictions": parsed, "timings": {"end_to_end_seconds": latency}}
