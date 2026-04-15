"""CLI interface for Sanitune."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from sanitune import __version__


@click.group()
@click.version_option(version=__version__, prog_name="sanitune")
def main() -> None:
    """Sanitune — AI-powered song cleaning."""


@main.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path), default=None, help="Output file path.")
@click.option("-m", "--mode", type=click.Choice(["mute", "bleep"]), default="mute", help="Editing mode.")
@click.option("-l", "--language", default="en", help="Language code (en, es, ...).")
@click.option("-d", "--device", type=click.Choice(["auto", "cpu", "cuda", "mps"]), default="auto", help="Compute device.")
@click.option("--bleep-freq", type=int, default=1000, help="Bleep tone frequency in Hz.")
@click.option("--model", "model_name", default="htdemucs_ft", help="Demucs model name.")
@click.option("--add-word", multiple=True, help="Additional words to flag (repeatable).")
@click.option("--exclude-word", multiple=True, help="Words to exclude from flagging (repeatable).")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
def process(
    input_file: Path,
    output: Path | None,
    mode: str,
    language: str,
    device: str,
    bleep_freq: int,
    model_name: str,
    add_word: tuple[str, ...],
    exclude_word: tuple[str, ...],
    verbose: bool,
) -> None:
    """Process an audio file to remove explicit content."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    from sanitune.pipeline import process as run_pipeline

    result = run_pipeline(
        input_file,
        output,
        mode=mode,
        language=language,
        device=device,
        custom_words=list(add_word) if add_word else None,
        exclude_words=list(exclude_word) if exclude_word else None,
        bleep_freq=bleep_freq,
        model_name=model_name,
    )

    click.echo(f"Output: {result.output_path}")
    click.echo(f"Flagged words: {len(result.flagged_words)}")
    if result.flagged_words:
        for fw in result.flagged_words:
            click.echo(f"  [{fw.word.start:.2f}s-{fw.word.end:.2f}s] {fw.word.text} (matched: {fw.matched_term})")
    click.echo(f"Time: {result.elapsed_seconds:.1f}s")
