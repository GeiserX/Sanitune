"""Kits.ai API client — singing voice conversion via cloud API."""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

BASE_URL = "https://arpeggi.io/api/kits/v1"
POLL_INTERVAL = 5
MAX_POLL_SECONDS = 300


def convert_voice(
    source_audio: np.ndarray,
    sample_rate: int,
    voice_model_id: int,
    api_key: str,
    *,
    conversion_strength: float = 0.5,
    pitch_shift: int = 0,
    model_volume_mix: float = 0.5,
) -> np.ndarray:
    """Convert audio to a target voice using Kits.ai API.

    Args:
        source_audio: Mono float32 audio to convert.
        sample_rate: Sample rate of the source audio.
        voice_model_id: Kits.ai voice model ID (from app.kits.ai).
        api_key: Kits.ai API key.
        conversion_strength: How strongly to apply the model's accent (0-1).
        pitch_shift: Semitone shift (-24 to 24).
        model_volume_mix: Blend input volume toward model volume (0-1).

    Returns:
        Converted mono float32 audio array.

    Raises:
        ImportError: If requests is not installed.
        RuntimeError: If conversion fails or times out.
    """
    try:
        import requests
    except ImportError:
        raise ImportError("requests is required for Kits.ai integration. Install with: pip install requests")

    headers = {"Authorization": f"Bearer {api_key}"}

    # Write source audio to a temp WAV file for upload
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        sf.write(str(tmp_path), source_audio, sample_rate)

    try:
        # Submit conversion job
        logger.info("Submitting voice conversion to Kits.ai (model=%d)...", voice_model_id)
        with open(tmp_path, "rb") as f:
            resp = requests.post(
                f"{BASE_URL}/voice-conversions",
                headers=headers,
                files={"soundFile": ("source.wav", f, "audio/wav")},
                data={
                    "voiceModelId": voice_model_id,
                    "conversionStrength": conversion_strength,
                    "pitchShift": pitch_shift,
                    "modelVolumeMix": model_volume_mix,
                },
                timeout=60,
            )

        if resp.status_code == 401:
            raise RuntimeError("Kits.ai API authentication failed — check your API key")
        if resp.status_code == 429:
            raise RuntimeError("Kits.ai rate limit exceeded (1 request/minute) — try again later")
        if resp.status_code != 200:
            raise RuntimeError(f"Kits.ai API error {resp.status_code}: {resp.text[:200]}")

        job = resp.json()
        job_id = job["id"]
        logger.info("Kits.ai job %d submitted, polling for completion...", job_id)

        # Poll until complete
        elapsed = 0.0
        while elapsed < MAX_POLL_SECONDS:
            time.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL

            poll_resp = requests.get(
                f"{BASE_URL}/voice-conversions/{job_id}",
                headers=headers,
                timeout=30,
            )
            if poll_resp.status_code != 200:
                raise RuntimeError(f"Kits.ai poll error {poll_resp.status_code}: {poll_resp.text[:200]}")

            job = poll_resp.json()
            status = job.get("status", "unknown")

            if status == "success":
                output_url = job.get("outputFileUrl")
                if not output_url:
                    raise RuntimeError("Kits.ai job succeeded but no output URL returned")
                logger.info("Kits.ai conversion complete (%.0fs)", elapsed)
                break
            elif status == "error":
                raise RuntimeError(f"Kits.ai conversion failed: {job}")
            elif status == "cancelled":
                raise RuntimeError("Kits.ai conversion was cancelled")

            logger.debug("Kits.ai job %d status: %s (%.0fs elapsed)", job_id, status, elapsed)
        else:
            raise RuntimeError(f"Kits.ai conversion timed out after {MAX_POLL_SECONDS}s")

        # Download the converted audio
        dl_resp = requests.get(output_url, timeout=60)
        if dl_resp.status_code != 200:
            raise RuntimeError(f"Failed to download Kits.ai output: HTTP {dl_resp.status_code}")

        # Parse the downloaded WAV
        import io
        data, sr = sf.read(io.BytesIO(dl_resp.content), dtype="float32")

        # Ensure mono
        if data.ndim > 1:
            data = data.mean(axis=1)

        # Resample if needed
        if sr != sample_rate:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=sample_rate).astype(np.float32)

        return data.astype(np.float32)

    finally:
        tmp_path.unlink(missing_ok=True)


def list_voice_models(api_key: str, *, my_models: bool = True) -> list[dict]:
    """List available voice models from Kits.ai.

    Args:
        api_key: Kits.ai API key.
        my_models: If True, only return your own models.

    Returns:
        List of voice model dicts with 'id', 'title', 'isUsable' fields.
    """
    import requests

    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"perPage": 50, "myModels": str(my_models).lower()}

    resp = requests.get(
        f"{BASE_URL}/voice-models",
        headers=headers,
        params=params,
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Kits.ai API error {resp.status_code}: {resp.text[:200]}")

    return resp.json()
