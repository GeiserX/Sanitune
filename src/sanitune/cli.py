"""CLI interface for Sanitune."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from sanitune import __version__
from sanitune.config import VALID_DEVICE_OPTIONS, VALID_MODES, Settings


@click.group()
@click.version_option(version=__version__, prog_name="sanitune")
def main() -> None:
    """Sanitune — AI-powered song cleaning."""


@main.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path), default=None, help="Output file path.")
@click.option(
    "-m",
    "--mode",
    type=click.Choice(sorted(VALID_MODES)),
    default=None,
    help="Editing mode. Defaults to SANITUNE_DEFAULT_MODE or 'mute'.",
)
@click.option("-l", "--language", default=None, help="Language code. Defaults to SANITUNE_LANGUAGE or 'en'.")
@click.option(
    "-d",
    "--device",
    type=click.Choice(sorted(VALID_DEVICE_OPTIONS)),
    default=None,
    help="Compute device. Defaults to SANITUNE_DEVICE or 'auto'.",
)
@click.option(
    "--bleep-freq",
    type=click.IntRange(min=1),
    default=None,
    help="Bleep tone frequency in Hz. Defaults to SANITUNE_BLEEP_FREQ or 1000.",
)
@click.option(
    "--max-file-size",
    type=click.IntRange(min=1),
    default=None,
    help="Maximum input file size in MB. Defaults to SANITUNE_MAX_FILE_SIZE or 200.",
)
@click.option("--model", "model_name", default="htdemucs_ft", help="Demucs model name.")
@click.option("--add-word", multiple=True, help="Additional words to flag (repeatable).")
@click.option("--exclude-word", multiple=True, help="Words to exclude from flagging (repeatable).")
@click.option("--artist", default=None, help="Artist name for lyrics lookup.")
@click.option("--title", default=None, help="Song title for lyrics lookup.")
@click.option("--genius-api-key", envvar="GENIUS_API_KEY", default=None, help="Genius API key for lyrics.")
@click.option(
    "--mapping",
    "custom_mapping",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Custom replacement word mapping JSON file (for replace mode).",
)
@click.option("--tts-voice", default=None, help="Override TTS voice name (for replace mode).")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
def process(
    input_file: Path,
    output: Path | None,
    mode: str | None,
    language: str | None,
    device: str | None,
    bleep_freq: int | None,
    max_file_size: int | None,
    model_name: str,
    add_word: tuple[str, ...],
    exclude_word: tuple[str, ...],
    artist: str | None,
    title: str | None,
    genius_api_key: str | None,
    custom_mapping: Path | None,
    tts_voice: str | None,
    verbose: bool,
) -> None:
    """Process an audio file to remove explicit content."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    try:
        settings = Settings.from_env()
    except ValueError as err:
        raise click.ClickException(str(err)) from err

    from sanitune.pipeline import process as run_pipeline

    result = run_pipeline(
        input_file,
        output,
        mode=settings.default_mode if mode is None else mode,
        language=settings.language if language is None else language,
        device=settings.device if device is None else device,
        custom_words=list(add_word) if add_word else None,
        exclude_words=list(exclude_word) if exclude_word else None,
        bleep_freq=settings.bleep_freq if bleep_freq is None else bleep_freq,
        model_name=model_name,
        max_file_size_mb=settings.max_file_size_mb if max_file_size is None else max_file_size,
        artist=artist,
        title=title,
        genius_api_key=genius_api_key,
        custom_mapping_path=custom_mapping,
        tts_voice=tts_voice,
    )

    click.echo(f"Output: {result.output_path}")
    click.echo(f"Flagged words: {len(result.flagged_words)}")
    if result.flagged_words:
        for fw in result.flagged_words:
            click.echo(f"  [{fw.word.start:.2f}s-{fw.word.end:.2f}s] {fw.word.text} (matched: {fw.matched_term})")
    click.echo(f"Time: {result.elapsed_seconds:.1f}s")
