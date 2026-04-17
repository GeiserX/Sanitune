"""Gradio-based web interface for Sanitune."""

from __future__ import annotations

import html
import json
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Keep temp dirs alive so Gradio can serve the files; cleaned up on next request
_previous_tmpdir: tempfile.TemporaryDirectory | None = None


def _parse_word_mappings(text: str) -> dict[str, str]:
    """Parse user-entered word mappings from 'word=replacement' lines or comma-separated pairs."""
    mappings: dict[str, str] = {}
    if not text.strip():
        return mappings

    # Support both newline-separated and comma-separated
    entries = text.replace("\n", ",").split(",")
    for entry in entries:
        entry = entry.strip()
        if "=" in entry:
            parts = entry.split("=", 1)
            word = parts[0].strip().lower()
            replacement = parts[1].strip()
            if word and replacement:
                mappings[word] = replacement
    return mappings


def _process_audio(
    audio_path: str,
    mode: str,
    language: str,
    synth_engine: str,
    target_words: str,
    replacement_mappings: str,
    use_auto_detection: bool,
    artist: str,
    title: str,
    bleep_freq: int,
    llm_provider: str,
    llm_api_key: str,
) -> tuple[str | None, str, str]:
    """Process an uploaded audio file and return (output_path, transcript, status).

    Returns:
        Tuple of (output_audio_path, transcript_html, status_message).
    """
    from sanitune.config import detect_device
    from sanitune.pipeline import process

    if not audio_path:
        return None, "", "No file uploaded."

    # Clean up previous request's temp dir
    global _previous_tmpdir
    if _previous_tmpdir is not None:
        _previous_tmpdir.cleanup()

    tmpdir = tempfile.TemporaryDirectory(prefix="sanitune_web_")
    _previous_tmpdir = tmpdir

    input_path = Path(audio_path)
    output_path = Path(tmpdir.name) / f"{input_path.stem}_clean.wav"

    device = detect_device("auto")

    # Parse manual word targets
    custom_words = [w.strip() for w in target_words.split(",") if w.strip()] if target_words else None

    # If auto-detection is off, we need at least manual words
    exclude_words = None
    if not use_auto_detection:
        if not custom_words:
            return None, "", "Auto-detection is off. Please enter words to target."
        # Disable built-in lists by excluding everything — the custom_words override will still apply
        # We pass a special flag to the pipeline
        exclude_words = ["*"]

    # Parse replacement mappings and write to a temp JSON if provided
    custom_mapping_path = None
    if mode == "replace" and replacement_mappings.strip():
        mappings = _parse_word_mappings(replacement_mappings)
        if mappings:
            mapping_file = Path(tmpdir.name) / "custom_mapping.json"
            mapping_file.write_text(json.dumps(mappings, ensure_ascii=False))
            custom_mapping_path = mapping_file
        elif not use_auto_detection:
            return None, "", "Replace mode requires word mappings (format: word=replacement)."

    # Configure AI suggestions if provided
    ai_api_key = llm_api_key.strip() if llm_api_key else None
    ai_provider = llm_provider if ai_api_key else None

    try:
        result = process(
            input_path,
            output_path,
            mode=mode,
            language=language,
            device=device,
            custom_words=custom_words,
            exclude_words=exclude_words,
            bleep_freq=bleep_freq,
            artist=artist.strip() or None,
            title=title.strip() or None,
            synth_engine=synth_engine,
            custom_mapping_path=custom_mapping_path,
            ai_provider=ai_provider,
            ai_api_key=ai_api_key,
        )
    except Exception:
        logger.exception("Processing failed")
        return None, "", "Processing failed due to an internal error. Check logs for details."

    # Build transcript HTML with flagged words highlighted
    flagged_indices = {fw.index for fw in result.flagged_words}
    flagged_terms = {fw.index: fw.matched_term for fw in result.flagged_words}

    html_parts = []
    for i, word in enumerate(result.transcription.words):
        timestamp = f"{word.start:.1f}s"
        safe_text = html.escape(word.text)
        if i in flagged_indices:
            safe_term = html.escape(flagged_terms[i])
            html_parts.append(
                f'<span style="background:#ff4444;color:white;padding:2px 4px;'
                f'border-radius:3px;cursor:help" title="Matched: {safe_term} @ {timestamp}">'
                f"{safe_text}</span>"
            )
        else:
            html_parts.append(f'<span title="{timestamp}">{safe_text}</span>')

    transcript_html = " ".join(html_parts)

    status = (
        f"Processed in {result.elapsed_seconds:.1f}s — "
        f"{len(result.flagged_words)} words flagged"
    )

    return str(output_path), transcript_html, status


def create_app():
    """Create and return the Gradio app."""
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Gradio is required for the web UI. Install with: pip install sanitune[web]")

    with gr.Blocks(
        title="Sanitune",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown("# Sanitune\nAI-powered song cleaning — upload a song and specify which words to clean.")

        with gr.Row():
            with gr.Column(scale=2):
                audio_input = gr.Audio(
                    label="Upload Song",
                    type="filepath",
                    sources=["upload"],
                )

                with gr.Row():
                    mode = gr.Dropdown(
                        choices=["mute", "bleep", "replace"],
                        value="mute",
                        label="Edit Mode",
                    )
                    language = gr.Dropdown(
                        choices=["en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh"],
                        value="en",
                        label="Language",
                    )

                gr.Markdown("### Words to Target")
                target_words = gr.Textbox(
                    label="Words to flag (comma-separated)",
                    placeholder="e.g. picha, joder, mierda",
                    lines=2,
                )

                use_auto_detection = gr.Checkbox(
                    label="Also use built-in profanity detection",
                    value=True,
                )

                replacement_mappings = gr.Textbox(
                    label="Replacement mappings (replace mode only, format: word=replacement)",
                    placeholder="e.g. picha=mano, joder=jolines, mierda=porqueria",
                    lines=3,
                    visible=True,
                )

                with gr.Row():
                    synth_engine = gr.Dropdown(
                        choices=["edge-tts", "bark"],
                        value="edge-tts",
                        label="Synth Engine (replace mode)",
                    )
                    bleep_freq = gr.Slider(
                        minimum=200, maximum=3000, value=1000, step=100,
                        label="Bleep Frequency (bleep mode)",
                    )

                with gr.Accordion("Lyrics Lookup (optional)", open=False):
                    artist = gr.Textbox(label="Artist", placeholder="e.g. Extremoduro")
                    title = gr.Textbox(label="Song Title")

                with gr.Accordion("AI Suggestions (optional)", open=False):
                    gr.Markdown(
                        "Provide your own LLM API key for context-aware replacement suggestions. "
                        "The key is used only for this session and never stored."
                    )
                    llm_provider = gr.Dropdown(
                        choices=["anthropic", "openai"],
                        value="anthropic",
                        label="LLM Provider",
                    )
                    llm_api_key = gr.Textbox(
                        label="API Key",
                        placeholder="sk-... or sk-ant-...",
                        type="password",
                    )

                process_btn = gr.Button("Process", variant="primary", size="lg")

            with gr.Column(scale=3):
                status_text = gr.Textbox(label="Status", interactive=False)
                transcript_html = gr.HTML(label="Transcript")
                audio_output = gr.Audio(label="Clean Version", type="filepath")

        process_btn.click(
            fn=_process_audio,
            inputs=[
                audio_input, mode, language, synth_engine,
                target_words, replacement_mappings, use_auto_detection,
                artist, title, bleep_freq,
                llm_provider, llm_api_key,
            ],
            outputs=[audio_output, transcript_html, status_text],
        )

    return app


def launch(host: str = "0.0.0.0", port: int = 7860, share: bool = False) -> None:
    """Launch the Gradio web UI."""
    app = create_app()
    app.launch(server_name=host, server_port=port, share=share)
