"""Cellar Scanner -- Gradio app. Runs locally and as an HF Space.

Tab 1: label photo + optional spoken tasting note -> grounded profile +
pairings, cited, TTS read-back. Tab 2: Blind Taste Mode (note -> variety
guess via the locally fine-tuned LoRA model). See CELLAR_SCANNER_SPEC.md §5.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import gradio as gr

import audio
import vision
from cache import FileCache, SemanticCache
from guardrails import GuardrailError, Refusal, generate_validated
from scanner import WINE_CONFIDENCE_GATE, assemble_prompt, ground_recommendation
from schemas import Recommendation, TastingNote, WineCard

DATA_DIR = Path(__file__).resolve().parent / "data"
semantic_cache = SemanticCache(DATA_DIR / "rec_cache.db", similarity_threshold=0.93)
audio_cache = FileCache(DATA_DIR / "audio_cache")
ADAPTER_PATH = Path(__file__).resolve().parent / "training" / "adapters"
BASE_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"


def _wine_cache_key(card: WineCard) -> str:
    return f"{card.wine_name}:{card.variety}:{card.vintage}:{card.producer}"


def extract_wine(image) -> tuple[WineCard | None, str]:
    if image is None:
        return None, "Upload a wine label photo."

    from PIL import Image
    if not isinstance(image, Image.Image):
        image = Image.open(image)

    result = vision.extract(
        image, WineCard,
        task_prompt="producer, wine_name, vintage (4-digit year), variety (grape), "
                    "region, country, abv if visible",
        domain_description="a wine bottle label",
        min_confidence=0.6,
        # wine labels commonly carry French/Italian/Spanish/German/Portuguese
        # producer and region names -- widen past English-only OCR.
        ocr_langs="eng+fra+ita+spa+deu+por",
    )
    if isinstance(result, Refusal):
        return None, f"⚠ {result.user_message}"

    gate_refusal = WINE_CONFIDENCE_GATE.check(
        result.extraction_confidence,
        retry_message="I'm not confident I read that label correctly -- try a straight-on photo with the label filling the frame.",
    )
    if gate_refusal:
        return None, f"⚠ {gate_refusal.user_message}"

    return result, f"✓ {result.producer or ''} {result.wine_name or ''} ({result.vintage or '?'}), {result.variety or 'unknown variety'}"


def transcribe_note(audio_path: str | None) -> str:
    if audio_path is None:
        return ""
    try:
        return audio.transcribe(audio_path).text
    except audio.NoSpeechError:
        return ""


def build_recommendation(card: WineCard | None, note_text: str):
    if card is None:
        return "⚠ Scan a wine label first.", None, ""

    note = TastingNote(text=note_text) if note_text.strip() else None
    cache_key = _wine_cache_key(card)
    prompt, packed, retrieved_ids, retrieved_texts = assemble_prompt(card, note)

    cache_hit = semantic_cache.get(prompt, cache_key)
    if cache_hit:
        rec = Recommendation.model_validate_json(cache_hit.response)
        cached_note = f" (cached, {cache_hit.kind}, sim={cache_hit.similarity:.2f})"
        grounding_note = ""
    else:
        try:
            rec = generate_validated(prompt, Recommendation, max_retries=2, llm_kwargs={"tier": "smart"})
        except GuardrailError as e:
            return f"⚠ Couldn't produce a recommendation: {'; '.join(e.violations[:2])}", None, ""
        rec, report = ground_recommendation(rec, retrieved_texts)
        grounding_note = f" · grounding rate: {report.grounding_rate:.0%}"
        semantic_cache.put(prompt, cache_key, rec.model_dump_json())
        cached_note = ""

    citations_line = ", ".join(rec.citations) if rec.citations else "(none)"
    pairings_md = "\n".join(f"- **{p.dish}**: {p.why} [{', '.join(p.citation_ids)}]" for p in rec.pairings)
    summary = (
        f"{rec.profile}{cached_note}{grounding_note}\n\n### Pairings\n{pairings_md}\n\n"
        f"**Similar wines:** {', '.join(rec.similar_wines) if rec.similar_wines else '(none)'}\n\n"
        f"**Citations:** {citations_line}"
    )

    audio_path = audio.speak_cached(rec.profile, "en-US-AriaNeural", audio_cache)
    dev_panel = packed.report_markdown() + "\n\n**Cache stats:** " + str(semantic_cache.stats())
    return summary, str(audio_path), dev_panel


def blind_taste(note_text: str) -> str:
    if not note_text.strip():
        return "Enter or speak a tasting note first."
    if not ADAPTER_PATH.exists():
        return "⚠ Adapter not trained yet -- run training/lora_harness/train.sh first."

    prompt = (
        f"Blind tasting note (grape name masked as [grape]): {note_text}\n\n"
        "What grape variety is this? Answer with a ranked list:\n1. <variety>\n2. <variety>\n3. <variety>"
    )
    proc = subprocess.run(
        ["mlx_lm.generate", "--model", BASE_MODEL, "--adapter-path", str(ADAPTER_PATH),
         "--prompt", prompt, "--max-tokens", "60"],
        capture_output=True, text=True, timeout=60,
    )
    return proc.stdout.strip() or "(no output -- check mlx_lm is installed and the adapter is trained)"


with gr.Blocks(title="Cellar Scanner") as demo:
    gr.Markdown("# 🍷 Cellar Scanner")

    with gr.Tab("Scan a label"):
        wine_state = gr.State(None)
        with gr.Row():
            with gr.Column():
                label_image = gr.Image(type="pil", label="Wine label photo")
                extract_btn = gr.Button("Extract wine info")
                extract_status = gr.Markdown()
                note_audio = gr.Audio(sources=["microphone"], type="filepath", label="Speak a tasting note (optional)")
                note_text = gr.Textbox(label="Tasting note")
                rec_btn = gr.Button("Get recommendation", variant="primary")
            with gr.Column():
                rec_output = gr.Markdown()
                rec_audio = gr.Audio(label="Listen")
                with gr.Accordion("Dev panel", open=False):
                    dev_panel = gr.Markdown()

        extract_btn.click(extract_wine, inputs=[label_image], outputs=[wine_state, extract_status])
        note_audio.change(transcribe_note, inputs=[note_audio], outputs=[note_text])
        rec_btn.click(build_recommendation, inputs=[wine_state, note_text], outputs=[rec_output, rec_audio, dev_panel])

    with gr.Tab("Blind Taste Mode"):
        gr.Markdown(
            "Paste or speak a tasting note with the label hidden -- the **locally fine-tuned "
            "LoRA model** (not Claude) guesses the grape before you reveal it. "
            "On the hosted Space this runs via Gemini instead, since MLX only runs on Apple Silicon; "
            "the real local-model benchmark is in the writeup."
        )
        blind_audio = gr.Audio(sources=["microphone"], type="filepath", label="Speak your tasting note")
        blind_text = gr.Textbox(label="Tasting note (grape hidden)")
        blind_btn = gr.Button("Guess the grape")
        blind_output = gr.Markdown()

        blind_audio.change(transcribe_note, inputs=[blind_audio], outputs=[blind_text])
        blind_btn.click(blind_taste, inputs=[blind_text], outputs=[blind_output])

if __name__ == "__main__":
    demo.launch()
