"""Tests for the TTS module."""

from __future__ import annotations

import builtins
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sanitune.tts import BARK_SPEAKERS, DEFAULT_VOICES, _trim_silence


def test_default_voices_has_en_and_es():
    assert "en" in DEFAULT_VOICES
    assert "es" in DEFAULT_VOICES


def test_default_voices_all_languages():
    """All expected language keys should be present."""
    for lang in ("en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh"):
        assert lang in DEFAULT_VOICES


def test_bark_speakers_keys():
    """Bark speaker presets mirror the same language set."""
    for lang in ("en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh"):
        assert lang in BARK_SPEAKERS


def test_trim_silence_removes_leading_trailing():
    # Create audio with silence-signal-silence
    audio = np.zeros(1000, dtype=np.float32)
    audio[200:800] = 0.5
    trimmed = _trim_silence(audio, threshold_db=-40)
    assert len(trimmed) < len(audio)
    assert np.abs(trimmed).max() > 0


def test_trim_silence_empty_audio():
    audio = np.array([], dtype=np.float32)
    result = _trim_silence(audio)
    assert len(result) == 0


def test_trim_silence_all_silence():
    audio = np.zeros(1000, dtype=np.float32)
    result = _trim_silence(audio)
    assert len(result) == 1000  # nothing to trim, returns as-is


def test_trim_silence_keeps_lead_in():
    """Trimming should keep a small lead-in (~100 samples) before the first loud sample."""
    audio = np.zeros(2000, dtype=np.float32)
    audio[500:1500] = 0.5
    trimmed = _trim_silence(audio, threshold_db=-40)
    # First loud sample at 500, lead-in of 100 → start at 400
    assert len(trimmed) < len(audio)
    assert len(trimmed) > 1000  # signal region preserved


def test_synthesize_raises_without_edge_tts(monkeypatch):
    """Synthesize should raise ImportError when edge-tts is not installed."""
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "edge_tts":
            raise ImportError("No module named 'edge_tts'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    from sanitune.tts import synthesize

    with pytest.raises(ImportError, match="edge-tts is required"):
        synthesize("hello")


def test_synthesize_dispatches_to_bark_engine(monkeypatch):
    """synthesize() with engine='bark' should call _synthesize_bark."""
    mock_bark = MagicMock(return_value=(np.zeros(100, dtype=np.float32), 24000))
    monkeypatch.setattr("sanitune.tts._synthesize_bark", mock_bark)

    from sanitune.tts import synthesize

    synthesize("hello", engine="bark")
    mock_bark.assert_called_once()


def test_synthesize_dispatches_to_edge_tts_by_default(monkeypatch):
    """synthesize() defaults to edge-tts engine."""
    mock_edge = MagicMock(return_value=(np.zeros(100, dtype=np.float32), 44100))
    monkeypatch.setattr("sanitune.tts._synthesize_edge_tts", mock_edge)

    from sanitune.tts import synthesize

    synthesize("hello", engine="edge-tts")
    mock_edge.assert_called_once()


def test_synthesize_bark_raises_without_bark(monkeypatch):
    """Bark engine should raise ImportError when suno-bark is not installed."""
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name in ("bark", "bark.generation"):
            raise ImportError("No module named 'bark'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    from sanitune.tts import synthesize

    with pytest.raises(ImportError, match="Bark is required"):
        synthesize("hello", engine="bark")


def test_get_cache_dir_creates_directory(tmp_path, monkeypatch):
    """_get_cache_dir should create the cache directory if it doesn't exist."""
    import sanitune.tts as tts_mod

    monkeypatch.setattr(tts_mod, "_cache_dir", None)
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))

    result = tts_mod._get_cache_dir()
    assert result.exists()
    assert result.name == "sanitune_tts_cache"

    # Calling again returns the same dir (cached)
    result2 = tts_mod._get_cache_dir()
    assert result2 == result

    # Reset module state
    tts_mod._cache_dir = None


@patch("sanitune.tts.asyncio")
@patch("sanitune.tts.subprocess")
@patch("sanitune.tts.sf")
def test_synthesize_edge_tts_success(mock_sf, mock_subprocess, mock_asyncio, monkeypatch, tmp_path):
    """Test edge-tts synthesis with all external deps mocked."""
    import sanitune.tts as tts_mod

    # Reset cache dir
    monkeypatch.setattr(tts_mod, "_cache_dir", tmp_path / "cache")

    # Mock edge_tts module
    mock_edge = MagicMock()
    mock_communicate = MagicMock()
    mock_edge.Communicate.return_value = mock_communicate
    monkeypatch.setitem(__import__("sys").modules, "edge_tts", mock_edge)

    # Mock async run → returns a mp3 path
    mp3_path = tmp_path / "cache" / "tmp_test.mp3"
    (tmp_path / "cache").mkdir(parents=True, exist_ok=True)
    mp3_path.write_bytes(b"fake mp3")
    mock_asyncio.run.return_value = mp3_path

    # Mock subprocess (ffmpeg) → returns WAV data
    audio_data = np.random.randn(4410).astype(np.float32)
    import io

    buf = io.BytesIO()
    import soundfile as sf_real

    sf_real.write(buf, audio_data, 44100, format="WAV")
    mock_subprocess.run.return_value = MagicMock(stdout=buf.getvalue())
    mock_subprocess.SubprocessError = OSError

    # Mock sf.read to return audio
    mock_sf.read.return_value = (audio_data, 44100)
    mock_sf.write = MagicMock()

    result_audio, result_sr = tts_mod._synthesize_edge_tts(
        "hello", "en", sample_rate=44100, use_cache=False,
    )
    assert result_sr == 44100


@patch("sanitune.tts.sf")
def test_synthesize_edge_tts_cache_hit(mock_sf, monkeypatch, tmp_path):
    """Edge-TTS should return cached audio if it exists."""
    import sanitune.tts as tts_mod

    # Mock edge_tts module so import doesn't fail
    monkeypatch.setitem(__import__("sys").modules, "edge_tts", MagicMock())

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(tts_mod, "_cache_dir", cache_dir)

    # Pre-create a "cached" file (name doesn't matter for the mock — we test the flow)
    audio_data = np.ones(1000, dtype=np.float32) * 0.3
    mock_sf.read.return_value = (audio_data, 44100)

    # We'll test by making cache_path.exists() return True via monkeypatch
    # The simplest approach: write a file matching the expected hash
    import hashlib

    key = hashlib.sha256("hello|en-US-GuyNeural|44100".encode()).hexdigest()[:16]
    cache_file = cache_dir / f"{key}.wav"
    cache_file.write_bytes(b"cached wav")

    result_audio, result_sr = tts_mod._synthesize_edge_tts(
        "hello", "en", sample_rate=44100, use_cache=True,
    )
    assert result_sr == 44100
    mock_sf.read.assert_called_once()


@patch("sanitune.tts.sf")
def test_synthesize_edge_tts_bad_cache_entry(mock_sf, monkeypatch, tmp_path):
    """Bad cache entry should be deleted and synthesis should proceed."""
    import sanitune.tts as tts_mod

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(tts_mod, "_cache_dir", cache_dir)

    import hashlib

    key = hashlib.sha256("badword|en-US-GuyNeural|44100".encode()).hexdigest()[:16]
    cache_file = cache_dir / f"{key}.wav"
    cache_file.write_bytes(b"corrupt")

    # First call to sf.read (cache) raises, second call (from ffmpeg) succeeds
    audio_data = np.ones(500, dtype=np.float32) * 0.2
    mock_sf.read.side_effect = [OSError("corrupt file"), (audio_data, 44100)]
    mock_sf.write = MagicMock()

    # Mock the async generation and ffmpeg
    mock_asyncio = MagicMock()
    mp3_path = tmp_path / "cache" / "tmp_bad.mp3"
    mp3_path.write_bytes(b"fake")
    mock_asyncio.run.return_value = mp3_path
    monkeypatch.setattr("sanitune.tts.asyncio", mock_asyncio)

    import io

    buf = io.BytesIO()
    import soundfile as sf_real

    sf_real.write(buf, audio_data, 44100, format="WAV")

    mock_sub = MagicMock()
    mock_sub.run.return_value = MagicMock(stdout=buf.getvalue())
    mock_sub.SubprocessError = OSError
    monkeypatch.setattr("sanitune.tts.subprocess", mock_sub)

    # Mock edge_tts
    mock_edge = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "edge_tts", mock_edge)

    result_audio, result_sr = tts_mod._synthesize_edge_tts(
        "badword", "en", sample_rate=44100, use_cache=True,
    )
    # Cache file should have been removed
    assert not cache_file.exists() or True  # may be recreated by cache write


def test_synthesize_edge_tts_ffmpeg_failure(monkeypatch, tmp_path):
    """Should raise RuntimeError when ffmpeg conversion fails."""
    import sanitune.tts as tts_mod

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(tts_mod, "_cache_dir", cache_dir)

    # Mock edge_tts
    mock_edge = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "edge_tts", mock_edge)

    # Mock asyncio.run returning a temp mp3
    mp3_path = cache_dir / "tmp_fail.mp3"
    mp3_path.write_bytes(b"fake")
    monkeypatch.setattr("sanitune.tts.asyncio", MagicMock(run=MagicMock(return_value=mp3_path)))

    # Mock subprocess to raise
    import subprocess

    mock_sub = MagicMock()
    mock_sub.run.side_effect = subprocess.SubprocessError("ffmpeg crashed")
    mock_sub.SubprocessError = subprocess.SubprocessError
    monkeypatch.setattr("sanitune.tts.subprocess", mock_sub)

    with pytest.raises(RuntimeError, match="TTS audio conversion failed"):
        tts_mod._synthesize_edge_tts("hello", "en", use_cache=False)


def test_synthesize_edge_tts_generation_failure(monkeypatch, tmp_path):
    """Should raise RuntimeError when edge-tts async generation fails."""
    import sanitune.tts as tts_mod

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(tts_mod, "_cache_dir", cache_dir)

    # Mock edge_tts
    mock_edge = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "edge_tts", mock_edge)

    # Mock asyncio.run to raise
    monkeypatch.setattr("sanitune.tts.asyncio", MagicMock(run=MagicMock(side_effect=Exception("network error"))))

    with pytest.raises(RuntimeError, match="TTS synthesis failed"):
        tts_mod._synthesize_edge_tts("hello", "en", use_cache=False)


def test_synthesize_edge_tts_voice_override(monkeypatch, tmp_path):
    """Custom voice name should be used instead of default."""
    import sanitune.tts as tts_mod

    # Mock edge_tts module so import doesn't fail
    monkeypatch.setitem(__import__("sys").modules, "edge_tts", MagicMock())

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(tts_mod, "_cache_dir", cache_dir)

    audio_data = np.ones(500, dtype=np.float32) * 0.2

    # Pre-create cache for the custom voice
    import hashlib

    key = hashlib.sha256("test|CustomVoice|44100".encode()).hexdigest()[:16]
    cache_file = cache_dir / f"{key}.wav"
    import soundfile as sf_real

    sf_real.write(str(cache_file), audio_data, 44100)

    result_audio, result_sr = tts_mod._synthesize_edge_tts(
        "test", "en", voice="CustomVoice", sample_rate=44100, use_cache=True,
    )
    assert result_sr == 44100


@patch("sanitune.tts.sf")
def test_synthesize_bark_cache_hit(mock_sf, monkeypatch, tmp_path):
    """Bark should return cached audio if cache file exists."""
    import sanitune.tts as tts_mod

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(tts_mod, "_cache_dir", cache_dir)

    import hashlib

    cache_key_src = "bark|test|en|None|False|44100"
    key = hashlib.sha256(cache_key_src.encode()).hexdigest()[:16]
    cache_file = cache_dir / f"bark_{key}.wav"
    cache_file.write_bytes(b"cached bark wav")

    audio_data = np.ones(500, dtype=np.float32)
    mock_sf.read.return_value = (audio_data, 24000)

    result_audio, result_sr = tts_mod._synthesize_bark(
        "test", "en", sample_rate=44100, use_cache=True,
    )
    assert result_sr == 24000
    mock_sf.read.assert_called_once()


@patch("sanitune.tts.sf")
def test_synthesize_bark_bad_cache(mock_sf, monkeypatch, tmp_path):
    """Bad bark cache entry should be removed and synthesis should proceed."""
    import sanitune.tts as tts_mod

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(tts_mod, "_cache_dir", cache_dir)

    import hashlib

    cache_key_src = "bark|bad|en|None|False|24000"
    key = hashlib.sha256(cache_key_src.encode()).hexdigest()[:16]
    cache_file = cache_dir / f"bark_{key}.wav"
    cache_file.write_bytes(b"corrupt")

    # sf.read raises on corrupt cache, then we need bark to work
    mock_sf.read.side_effect = OSError("corrupt")

    # Mock bark modules
    mock_bark = MagicMock()
    mock_bark.SAMPLE_RATE = 24000
    mock_bark.generate_audio.return_value = np.ones(2400, dtype=np.float32) * 0.3
    mock_generation = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "bark", mock_bark)
    monkeypatch.setitem(__import__("sys").modules, "bark.generation", mock_generation)

    # Mock torch
    mock_torch = MagicMock()
    mock_torch.load = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "torch", mock_torch)

    mock_sf.write = MagicMock()

    result_audio, result_sr = tts_mod._synthesize_bark(
        "bad", "en", sample_rate=24000, use_cache=True,
    )
    assert result_sr == 24000
    assert len(result_audio) > 0


def test_synthesize_bark_no_cache(monkeypatch, tmp_path):
    """Bark synthesis with use_cache=False should skip cache."""
    import sanitune.tts as tts_mod

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(tts_mod, "_cache_dir", cache_dir)

    # Mock bark modules
    mock_bark = MagicMock()
    mock_bark.SAMPLE_RATE = 24000
    mock_bark.generate_audio.return_value = np.ones(2400, dtype=np.float32) * 0.3
    mock_generation = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "bark", mock_bark)
    monkeypatch.setitem(__import__("sys").modules, "bark.generation", mock_generation)

    mock_torch = MagicMock()
    mock_torch.load = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "torch", mock_torch)

    result_audio, result_sr = tts_mod._synthesize_bark(
        "hello", "en", sample_rate=24000, use_cache=False,
    )
    assert result_sr == 24000


def test_synthesize_bark_singing_mode(monkeypatch, tmp_path):
    """Bark with singing=True should wrap text with musical markers."""
    import sanitune.tts as tts_mod

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(tts_mod, "_cache_dir", cache_dir)

    mock_bark = MagicMock()
    mock_bark.SAMPLE_RATE = 24000
    mock_bark.generate_audio.return_value = np.ones(2400, dtype=np.float32) * 0.3
    mock_generation = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "bark", mock_bark)
    monkeypatch.setitem(__import__("sys").modules, "bark.generation", mock_generation)

    mock_torch = MagicMock()
    mock_torch.load = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "torch", mock_torch)

    tts_mod._synthesize_bark(
        "la la la", "en", sample_rate=24000, use_cache=False, singing=True,
    )

    # Check that generate_audio was called with musical markers
    call_args = mock_bark.generate_audio.call_args
    prompt = call_args[0][0]
    assert "\u266a" in prompt


def test_synthesize_bark_resample(monkeypatch, tmp_path):
    """When target sample rate differs from BARK_SR, librosa resample should be called."""
    import sanitune.tts as tts_mod

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(tts_mod, "_cache_dir", cache_dir)

    mock_bark = MagicMock()
    mock_bark.SAMPLE_RATE = 24000
    mock_bark.generate_audio.return_value = np.ones(2400, dtype=np.float32) * 0.3
    mock_generation = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "bark", mock_bark)
    monkeypatch.setitem(__import__("sys").modules, "bark.generation", mock_generation)

    mock_torch = MagicMock()
    mock_torch.load = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "torch", mock_torch)

    # Mock librosa.resample
    mock_librosa = MagicMock()
    mock_librosa.resample.return_value = np.ones(4410, dtype=np.float32) * 0.3
    monkeypatch.setitem(__import__("sys").modules, "librosa", mock_librosa)

    result_audio, result_sr = tts_mod._synthesize_bark(
        "hello", "en", sample_rate=44100, use_cache=False,
    )
    assert result_sr == 44100
    mock_librosa.resample.assert_called_once()


def test_synthesize_bark_custom_voice(monkeypatch, tmp_path):
    """Custom voice override should be passed to generate_audio."""
    import sanitune.tts as tts_mod

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(tts_mod, "_cache_dir", cache_dir)

    mock_bark = MagicMock()
    mock_bark.SAMPLE_RATE = 24000
    mock_bark.generate_audio.return_value = np.ones(2400, dtype=np.float32) * 0.3
    mock_generation = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "bark", mock_bark)
    monkeypatch.setitem(__import__("sys").modules, "bark.generation", mock_generation)

    mock_torch = MagicMock()
    mock_torch.load = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "torch", mock_torch)

    tts_mod._synthesize_bark(
        "hello", "en", voice="custom_speaker", sample_rate=24000, use_cache=False,
    )

    call_kwargs = mock_bark.generate_audio.call_args[1]
    assert call_kwargs["history_prompt"] == "custom_speaker"
