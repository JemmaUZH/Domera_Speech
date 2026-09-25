"""Interactive MELD emotion and sentiment demo for the three trained systems."""

from __future__ import annotations

import os
import sys
import time
from functools import lru_cache
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from speech_sentiment import EMOTION_LABELS, LABELS  # noqa: E402
from speech_sentiment.inference import Predictor, normalized_wav, synchronize  # noqa: E402
from speech_sentiment.model import concatenate  # noqa: E402
from speech_sentiment.realtime import classify_audio  # noqa: E402


EMOTION_CHECKPOINT_DIR = Path(os.environ.get(
    "SPEECH_SENTIMENT_EMOTION_CHECKPOINT_DIR",
    os.environ.get("SPEECH_SENTIMENT_CHECKPOINT_DIR", ROOT / "artifacts" / "final" / "checkpoints" / "emotion"),
))
SENTIMENT_CHECKPOINT_DIR = Path(os.environ.get(
    "SPEECH_SENTIMENT_SENTIMENT_CHECKPOINT_DIR",
    ROOT / "artifacts" / "final" / "checkpoints" / "sentiment",
))
DEVICE = os.environ.get("SPEECH_SENTIMENT_DEVICE", "cpu").lower()
TASKS = {
    "7-class emotion": ("emotion", EMOTION_LABELS, EMOTION_CHECKPOINT_DIR),
    "3-class sentiment": ("sentiment", LABELS, SENTIMENT_CHECKPOINT_DIR),
}
SYSTEMS = {"Emotion2Vec audio-only": "audio", "Emotion2Vec + Text": "fusion", "Continuous Mimi": "mimi"}
TASK_CHOICES = list(TASKS)
COMPARE = "Compare all models"
GPT_REALTIME = "OpenAI GPT Realtime (audio-native)"
CHOICES = [*SYSTEMS, GPT_REALTIME, COMPARE]
COLORS = ["#4263eb", "#0f9d78", "#e07a2d", "#7547a8"]


def _resolve_device() -> str:
    if DEVICE == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SPEECH_SENTIMENT_DEVICE=cuda, but CUDA is unavailable")
    if DEVICE == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise RuntimeError("SPEECH_SENTIMENT_DEVICE=mps, but Apple MPS is unavailable")
    if DEVICE not in {"cpu", "cuda", "mps", "auto"}:
        raise RuntimeError("SPEECH_SENTIMENT_DEVICE must be cpu, cuda, mps, or auto")
    return DEVICE


@lru_cache(maxsize=6)
def _predictor(task_choice: str, name: str) -> Predictor:
    task, label_names, checkpoint_dir = TASKS[task_choice]
    expected_system = SYSTEMS[name]
    checkpoint_path = checkpoint_dir / f"{expected_system}.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing trained checkpoint: {checkpoint_path}")
    # The emotion predictor owns the shared feature encoders for this system;
    # the sentiment predictor only needs its independently trained head.
    predictor = Predictor(
        checkpoint_path,
        device=_resolve_device(),
        load_encoders=(task_choice == "7-class emotion"),
    )
    if predictor.task != task or predictor.system != expected_system:
        raise ValueError(f"{checkpoint_path} is not the expected {task} {expected_system} checkpoint")
    if predictor.label_names != label_names:
        raise ValueError(f"Unexpected {task} label order in {checkpoint_path}: {predictor.label_names}")
    return predictor


def _format_result(name: str, result: dict, latency: float, label_names: tuple[str, ...]) -> str:
    label = label_names[result["label"]]
    lines = [
        f"### {label.title()} · {result['confidence']:.1%} confidence",
        f"End-to-end inference latency: **{latency:.3f} s**",
    ]
    if result["asr_transcript"] is not None:
        transcript = result["asr_transcript"] or "(ASR returned an empty transcript.)"
        lines.append(f"\n**ASR transcript**\n\n> {transcript.replace(chr(10), ' ')}")
    if "asr_seconds" in result["timings"]:
        stages = result["timings"]
        lines.append(
            "\n_Component latency:_ "
            f"ASR {stages['asr_seconds']:.3f}s · "
            f"Emotion2Vec {stages['emotion2vec_seconds']:.3f}s · "
            f"Text encoder {stages['text_seconds']:.3f}s · "
            f"Fusion/head {stages['head_seconds']:.3f}s"
        )
    lines.append("\n**Class probabilities** are shown in the chart below.")
    return "\n".join(lines)


def _probability_plot(results: list[tuple[str, dict, float]], label_names: tuple[str, ...]):
    height = max(2.5, 2.15 * len(results))
    fig, axes = plt.subplots(len(results), 1, figsize=(9, height), squeeze=False)
    for index, (name, result, _) in enumerate(results):
        ax = axes[index, 0]
        ax.set_title(name, loc="left", fontsize=11, fontweight="semibold", pad=5)
        if result.get("probabilities") is None:
            ax.text(0.5, 0.5, "Label only · probability scores unavailable", ha="center", va="center", color="#666")
            ax.set_axis_off()
        else:
            probabilities = np.asarray(result["probabilities"], dtype=float)
            bars = ax.barh(label_names, probabilities, color=COLORS[index % len(COLORS)], height=0.66)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            ax.grid(axis="x", alpha=0.2)
            ax.set_axisbelow(True)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="y", length=0)
            ax.bar_label(bars, labels=[f"{p:.0%}" for p in probabilities], padding=3, fontsize=8)
    fig.tight_layout(h_pad=1.2)
    return fig


def _predict_both_tasks(system_name: str, audio_path: Path) -> tuple[dict[str, dict], str | None]:
    """Extract shared features once, then apply each task's existing trained head."""
    system = SYSTEMS[system_name]
    predictors = {choice: _predictor(choice, system_name) for choice in TASK_CHOICES}
    feature_predictor = predictors["7-class emotion"]
    vectors = {}
    timings = {}
    transcript = None

    if system in ("audio", "fusion"):
        start = time.perf_counter()
        vectors["audio"] = feature_predictor.audio.encode(audio_path)
        synchronize(feature_predictor.device)
        timings["emotion2vec_seconds"] = time.perf_counter() - start
    if system == "fusion":
        start = time.perf_counter()
        transcript = feature_predictor.asr.transcribe(audio_path)
        synchronize(feature_predictor.device)
        timings["asr_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        vectors["text"] = feature_predictor.text.encode(transcript)
        synchronize(feature_predictor.device)
        timings["text_seconds"] = time.perf_counter() - start
    if system == "mimi":
        start = time.perf_counter()
        vectors["mimi"] = feature_predictor.mimi.encode(audio_path)
        synchronize(feature_predictor.device)
        timings["mimi_seconds"] = time.perf_counter() - start

    feature = concatenate(vectors, system)
    shared_seconds = sum(timings.values())
    results = {}
    for choice, predictor in predictors.items():
        if feature.shape != predictor.mean.shape:
            raise ValueError(f"Expected feature shape {predictor.mean.shape}; got {feature.shape}")
        start = time.perf_counter()
        with torch.inference_mode():
            normalized = torch.from_numpy((feature - predictor.mean) / predictor.scale).to(predictor.device)[None]
            probabilities = torch.softmax(predictor.head(normalized), dim=-1)[0].cpu().numpy()
        synchronize(predictor.device)
        head_seconds = time.perf_counter() - start
        label = int(probabilities.argmax())
        results[choice] = {
            "label": label,
            "confidence": float(probabilities[label]),
            "probabilities": probabilities.tolist(),
            "asr_transcript": transcript,
            "timings": {**timings, "head_seconds": head_seconds, "end_to_end_seconds": shared_seconds + head_seconds},
        }
    return results, transcript


def _render_run_state(
    results_by_task: dict[str, list[tuple[str, dict, float]]],
    selected_model: str,
    model_names: list[str],
    has_api_key: bool,
    transcript: str | None,
    progress: str,
    *,
    completed: bool = False,
    error: str | None = None,
):
    table = []
    charts = {}
    task_specs = (("7-class emotion", EMOTION_LABELS), ("3-class sentiment", LABELS))
    for task, label_names in task_specs:
        task_results = results_by_task[task]
        charts[task] = _probability_plot(task_results, label_names) if task_results else None
        for name, result, latency in task_results:
            label = result.get("label_text", label_names[result["label"]] if result["label"] is not None else "Invalid response")
            confidence = f"{result['confidence']:.1%}" if result.get("confidence") is not None else "Unavailable"
            table.append([task, name, label, confidence, f"{latency:.3f} s"])

    if selected_model == COMPARE:
        expected_count = len(model_names) * len(task_specs)
        received_count = sum(len(rows) for rows in results_by_task.values())
        if completed:
            summary = f"Analysis complete: {received_count} task results from {len(model_names)} model(s)."
        else:
            summary = f"Results available: {received_count}/{expected_count}. New results will appear as each model finishes."
        if not has_api_key:
            summary += " GPT Realtime was skipped because no API key was entered."
        elif GPT_REALTIME in model_names:
            summary += " GPT Realtime returns labels without class probabilities."
    else:
        summary_parts = []
        for task, label_names in task_specs:
            task_results = results_by_task[task]
            if not task_results:
                summary_parts.append(f"## {task}\n_Analysis pending._")
                continue
            name, result, latency = task_results[0]
            if name == GPT_REALTIME:
                label = result.get("label_text", "invalid")
                summary_parts.append(f"## {task}: {label.title()}\nLatency: **{latency:.3f} s** · Confidence unavailable")
            else:
                summary_parts.append(f"## {task}\n{_format_result(name, result, latency, label_names)}")
        summary = "\n\n".join(summary_parts)

    gpt_rows = results_by_task["7-class emotion"]
    gpt_result = next((result for name, result, _ in gpt_rows if name == GPT_REALTIME), None)
    if gpt_result is not None:
        summary += f"\n\n**GPT Realtime raw JSON**\n\n```json\n{gpt_result['raw_response']}\n```"

    if error:
        summary += f"\n\n**{error}**"
    transcript_display = transcript or "Select Emotion2Vec + Text to view its ASR transcript."
    status = progress
    if selected_model == GPT_REALTIME or GPT_REALTIME in model_names:
        status += " · GPT Realtime API latency includes network and service overhead."
    elif selected_model == COMPARE:
        status += f" · Local inference device: {_resolve_device().upper()}."
    else:
        status += f" · Inference device: {_resolve_device().upper()}."
    return summary, table, charts["7-class emotion"], charts["3-class sentiment"], transcript_display, status


def _yield_run_state(*args, **kwargs):
    output = _render_run_state(*args, **kwargs)
    try:
        yield output
    finally:
        for figure in output[2:4]:
            if figure is not None:
                plt.close(figure)


def run_demo(audio_path: str | None, selected_model: str, api_key: str | None):
    if not audio_path:
        yield "Record or upload one audio utterance first.", [], None, None, "", ""
        return
    if selected_model not in CHOICES:
        yield "Select a model to run.", [], None, None, "", ""
        return
    if selected_model == GPT_REALTIME and not (api_key or "").strip():
        yield "Enter an OpenAI API key to use GPT Realtime.", [], None, None, "", ""
        return

    compare_models = selected_model == COMPARE
    has_api_key = bool((api_key or "").strip())
    model_names = [*SYSTEMS, *([GPT_REALTIME] if has_api_key else [])] if compare_models else [selected_model]
    results_by_task = {choice: [] for choice in TASK_CHOICES}
    fusion_transcript = None
    yield from _yield_run_state(results_by_task, selected_model, model_names, has_api_key, fusion_transcript, "Preparing audio…")
    try:
        normalization_start = time.perf_counter()
        with normalized_wav(Path(audio_path)) as wav_path:
            normalization_seconds = time.perf_counter() - normalization_start
            for name in model_names:
                if name == GPT_REALTIME:
                    yield from _yield_run_state(results_by_task, selected_model, model_names, has_api_key, fusion_transcript, f"Running {name} for both tasks…")
                    response = classify_audio(wav_path, api_key or "")
                    for choice, result in response["predictions"].items():
                        latency = normalization_seconds + result["timings"]["end_to_end_seconds"]
                        results_by_task[choice].append((name, result, latency))
                    yield from _yield_run_state(results_by_task, selected_model, model_names, has_api_key, fusion_transcript, f"Completed {name} for both tasks.")
                else:
                    yield from _yield_run_state(results_by_task, selected_model, model_names, has_api_key, fusion_transcript, f"Running {name} for both tasks…")
                    local_results, transcript = _predict_both_tasks(name, wav_path)
                    if name == "Emotion2Vec + Text":
                        fusion_transcript = transcript
                    for choice, result in local_results.items():
                        latency = normalization_seconds + result["timings"]["end_to_end_seconds"]
                        results_by_task[choice].append((name, result, latency))
                    yield from _yield_run_state(results_by_task, selected_model, model_names, has_api_key, fusion_transcript, f"Completed {name}.")
        yield from _yield_run_state(results_by_task, selected_model, model_names, has_api_key, fusion_transcript, "Analysis complete.", completed=True)
    except Exception as exc:
        yield from _yield_run_state(
            results_by_task,
            selected_model,
            model_names,
            has_api_key,
            fusion_transcript,
            "Analysis stopped after an error.",
            error=f"Inference failed: {type(exc).__name__}: {exc}",
        )


with gr.Blocks(title="Speech Emotion & Sentiment Analyzer", theme=gr.themes.Soft()) as app:
    gr.Markdown("# Speech Emotion & Sentiment Analyzer")
    with gr.Row():
        audio_input = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="Record or upload one utterance",
        )
        with gr.Column():
            model_choice = gr.Dropdown(choices=CHOICES, value=CHOICES[0], label="Model")
            api_key_input = gr.Textbox(
                label="OpenAI API key (for GPT Realtime)",
                placeholder="sk-…",
                type="password",
                info="Sent to the app backend for this request; the app does not store it.",
            )
            run_button = gr.Button("Analyze audio", variant="primary")
            device_note = gr.Markdown(f"Inference device: **{DEVICE.upper()}**" if DEVICE != "auto" else "Inference device: **auto-detect**")

    gr.Markdown("## Result")
    summary_output = gr.Markdown()
    comparison_output = gr.Dataframe(
        headers=["Task", "Model", "Prediction", "Confidence", "Latency"],
        datatype=["str", "str", "str", "str", "str"],
        interactive=False,
        label="Emotion and sentiment results",
    )
    with gr.Tabs():
        with gr.Tab("7-class emotion"):
            emotion_probability_output = gr.Plot(label="Emotion class probabilities")
        with gr.Tab("3-class sentiment"):
            sentiment_probability_output = gr.Plot(label="Sentiment class probabilities")
    transcript_output = gr.Textbox(label="ASR transcript used by Emotion2Vec + Text", interactive=False, lines=2)
    status_output = gr.Markdown()
    run_button.click(
        run_demo,
        inputs=[audio_input, model_choice, api_key_input],
        outputs=[summary_output, comparison_output, emotion_probability_output, sentiment_probability_output, transcript_output, status_output],
    )


if __name__ == "__main__":
    app.launch(
        server_name=os.environ.get("SPEECH_SENTIMENT_HOST", "127.0.0.1"),
        server_port=int(os.environ.get("SPEECH_SENTIMENT_PORT", "7860")),
        share=False,
    )
