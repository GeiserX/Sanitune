"""Tests for the Kits.ai API client module."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from sanitune.kits_client import BASE_URL, convert_voice, list_voice_models


def _make_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    """Helper to create WAV bytes from numpy array."""
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


class TestConvertVoice:
    @patch("sanitune.kits_client.time.sleep")
    @patch("requests.get")
    @patch("requests.post")
    def test_successful_conversion(self, mock_post, mock_get, mock_sleep):
        sr = 16000
        source = np.random.randn(sr).astype(np.float32) * 0.1

        # Mock submit response
        submit_resp = MagicMock()
        submit_resp.status_code = 200
        submit_resp.json.return_value = {"id": 42}
        mock_post.return_value = submit_resp

        # Mock poll response (success)
        out_audio = np.random.randn(sr).astype(np.float32) * 0.1
        wav_bytes = _make_wav_bytes(out_audio, sr)

        poll_resp = MagicMock()
        poll_resp.status_code = 200
        poll_resp.json.return_value = {"status": "success", "outputFileUrl": "https://example.com/out.wav"}

        # Mock download response
        dl_resp = MagicMock()
        dl_resp.status_code = 200
        dl_resp.content = wav_bytes

        mock_get.side_effect = [poll_resp, dl_resp]

        result = convert_voice(source, sr, voice_model_id=1, api_key="test-key")
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    @patch("requests.post")
    def test_auth_failure(self, mock_post):
        sr = 16000
        source = np.zeros(sr, dtype=np.float32)

        resp = MagicMock()
        resp.status_code = 401
        mock_post.return_value = resp

        with pytest.raises(RuntimeError, match="authentication failed"):
            convert_voice(source, sr, voice_model_id=1, api_key="bad-key")

    @patch("requests.post")
    def test_rate_limit(self, mock_post):
        sr = 16000
        source = np.zeros(sr, dtype=np.float32)

        resp = MagicMock()
        resp.status_code = 429
        mock_post.return_value = resp

        with pytest.raises(RuntimeError, match="rate limit"):
            convert_voice(source, sr, voice_model_id=1, api_key="key")

    @patch("requests.post")
    def test_generic_api_error(self, mock_post):
        sr = 16000
        source = np.zeros(sr, dtype=np.float32)

        resp = MagicMock()
        resp.status_code = 500
        resp.text = "Server error"
        mock_post.return_value = resp

        with pytest.raises(RuntimeError, match="API error 500"):
            convert_voice(source, sr, voice_model_id=1, api_key="key")

    @patch("sanitune.kits_client.MAX_POLL_SECONDS", 0)
    @patch("requests.get")
    @patch("requests.post")
    def test_conversion_timeout(self, mock_post, mock_get):
        sr = 16000
        source = np.zeros(sr, dtype=np.float32)

        submit_resp = MagicMock()
        submit_resp.status_code = 200
        submit_resp.json.return_value = {"id": 1}
        mock_post.return_value = submit_resp

        with pytest.raises(RuntimeError, match="timed out"):
            convert_voice(source, sr, voice_model_id=1, api_key="key")

    @patch("sanitune.kits_client.time.sleep")
    @patch("requests.get")
    @patch("requests.post")
    def test_conversion_error_status(self, mock_post, mock_get, mock_sleep):
        sr = 16000
        source = np.zeros(sr, dtype=np.float32)

        submit_resp = MagicMock()
        submit_resp.status_code = 200
        submit_resp.json.return_value = {"id": 1}
        mock_post.return_value = submit_resp

        poll_resp = MagicMock()
        poll_resp.status_code = 200
        poll_resp.json.return_value = {"status": "error"}
        mock_get.return_value = poll_resp

        with pytest.raises(RuntimeError, match="conversion failed"):
            convert_voice(source, sr, voice_model_id=1, api_key="key")

    @patch("sanitune.kits_client.time.sleep")
    @patch("requests.get")
    @patch("requests.post")
    def test_conversion_cancelled(self, mock_post, mock_get, mock_sleep):
        sr = 16000
        source = np.zeros(sr, dtype=np.float32)

        submit_resp = MagicMock()
        submit_resp.status_code = 200
        submit_resp.json.return_value = {"id": 1}
        mock_post.return_value = submit_resp

        poll_resp = MagicMock()
        poll_resp.status_code = 200
        poll_resp.json.return_value = {"status": "cancelled"}
        mock_get.return_value = poll_resp

        with pytest.raises(RuntimeError, match="cancelled"):
            convert_voice(source, sr, voice_model_id=1, api_key="key")

    @patch("sanitune.kits_client.time.sleep")
    @patch("requests.get")
    @patch("requests.post")
    def test_poll_error_status(self, mock_post, mock_get, mock_sleep):
        sr = 16000
        source = np.zeros(sr, dtype=np.float32)

        submit_resp = MagicMock()
        submit_resp.status_code = 200
        submit_resp.json.return_value = {"id": 1}
        mock_post.return_value = submit_resp

        poll_resp = MagicMock()
        poll_resp.status_code = 500
        poll_resp.text = "Error"
        mock_get.return_value = poll_resp

        with pytest.raises(RuntimeError, match="poll error"):
            convert_voice(source, sr, voice_model_id=1, api_key="key")

    @patch("sanitune.kits_client.time.sleep")
    @patch("requests.get")
    @patch("requests.post")
    def test_download_failure(self, mock_post, mock_get, mock_sleep):
        sr = 16000
        source = np.zeros(sr, dtype=np.float32)

        submit_resp = MagicMock()
        submit_resp.status_code = 200
        submit_resp.json.return_value = {"id": 1}
        mock_post.return_value = submit_resp

        poll_resp = MagicMock()
        poll_resp.status_code = 200
        poll_resp.json.return_value = {"status": "success", "outputFileUrl": "https://example.com/out.wav"}

        dl_resp = MagicMock()
        dl_resp.status_code = 404
        mock_get.side_effect = [poll_resp, dl_resp]

        with pytest.raises(RuntimeError, match="Failed to download"):
            convert_voice(source, sr, voice_model_id=1, api_key="key")

    @patch("sanitune.kits_client.time.sleep")
    @patch("requests.get")
    @patch("requests.post")
    def test_success_no_output_url(self, mock_post, mock_get, mock_sleep):
        sr = 16000
        source = np.zeros(sr, dtype=np.float32)

        submit_resp = MagicMock()
        submit_resp.status_code = 200
        submit_resp.json.return_value = {"id": 1}
        mock_post.return_value = submit_resp

        poll_resp = MagicMock()
        poll_resp.status_code = 200
        poll_resp.json.return_value = {"status": "success"}
        mock_get.return_value = poll_resp

        with pytest.raises(RuntimeError, match="no output URL"):
            convert_voice(source, sr, voice_model_id=1, api_key="key")

    @patch("sanitune.kits_client.time.sleep")
    @patch("requests.get")
    @patch("requests.post")
    def test_stereo_output_converted_to_mono(self, mock_post, mock_get, mock_sleep):
        sr = 16000
        source = np.zeros(sr, dtype=np.float32)

        submit_resp = MagicMock()
        submit_resp.status_code = 200
        submit_resp.json.return_value = {"id": 1}
        mock_post.return_value = submit_resp

        stereo_audio = np.random.randn(sr, 2).astype(np.float32) * 0.1
        wav_bytes = _make_wav_bytes(stereo_audio, sr)

        poll_resp = MagicMock()
        poll_resp.status_code = 200
        poll_resp.json.return_value = {"status": "success", "outputFileUrl": "https://example.com/out.wav"}

        dl_resp = MagicMock()
        dl_resp.status_code = 200
        dl_resp.content = wav_bytes
        mock_get.side_effect = [poll_resp, dl_resp]

        result = convert_voice(source, sr, voice_model_id=1, api_key="key")
        assert result.ndim == 1  # Should be mono


class TestListVoiceModels:
    @patch("requests.get")
    def test_list_models(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"id": 1, "title": "Model1"}]
        mock_get.return_value = mock_resp

        result = list_voice_models("test-key")
        assert len(result) == 1
        assert result[0]["title"] == "Model1"

    @patch("requests.get")
    def test_list_models_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"
        mock_get.return_value = mock_resp

        with pytest.raises(RuntimeError, match="API error 401"):
            list_voice_models("bad-key")
