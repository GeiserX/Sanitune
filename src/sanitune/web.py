"""Gradio-based web interface for Sanitune."""

from __future__ import annotations

import html
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Keep temp dirs alive so Gradio can serve the files; cleaned up on next request
_previous_tmpdir: tempfile.TemporaryDirectory | None = None


def _process_audio(
    audio_path: str,
    mode: str,
    language: str,
    synth_engine: str,
    add_words: str,
    exclude_words: str,
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

    custom_words = [w.strip() for w in add_words.split(",") if w.strip()] if add_words else None
    excl_words = [w.strip() for w in exclude_words.split(",") if w.strip()] if exclude_words else None

    device = detect_device("auto")

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
            exclude_words=excl_words,
            bleep_freq=bleep_freq,
            artist=artist.strip() or None,
            title=title.strip() or None,
            synth_engine=synth_engine,
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
        gr.Markdown("# Sanitune\nAI-powered song cleaning — upload a song, detect profanity, and clean it.")

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
                    synth_engine = gr.Dropdown(
                        choices=["edge-tts", "bark"],
                        value="edge-tts",
                        label="Synth Engine (replace mode)",
                    )

                with gr.Accordion("Bleep Settings", open=False):
                    bleep_freq = gr.Slider(
                        minimum=200, maximum=3000, value=1000, step=100,
                        label="Bleep Frequency (Hz)",
                    )

                with gr.Accordion("Word Overrides", open=False):
                    add_words = gr.Textbox(
                        label="Additional Words to Flag",
                        placeholder="word1, word2, ...",
                    )
                    exclude_words = gr.Textbox(
                        label="Words to Exclude",
                        placeholder="word1, word2, ...",
                    )

                with gr.Accordion("Lyrics Lookup (optional)", open=False):
                    artist = gr.Textbox(label="Artist", placeholder="e.g. Bad Bunny")
                    title = gr.Textbox(label="Song Title", placeholder="e.g. Titi Me Pregunto")

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
                add_words, exclude_words, artist, title, bleep_freq,
                llm_provider, llm_api_key,
            ],
            outputs=[audio_output, transcript_html, status_text],
        )

    return app


def launch(host: str = "0.0.0.0", port: int = 7860, share: bool = False) -> None:
    """Launch the Gradio web UI."""
    app = create_app()
    app.launch(server_name=host, server_port=port, share=share)
