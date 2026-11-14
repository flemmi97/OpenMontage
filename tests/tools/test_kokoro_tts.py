"""Tests for the offline Kokoro TTS provider."""

from __future__ import annotations

import pytest

from tools.audio.kokoro_tts import KokoroTTS
from tools.base_tool import ToolStatus


def test_contract_fields():
    tool = KokoroTTS()
    assert tool.capability == "tts"
    assert tool.provider == "kokoro"
    assert tool.estimate_cost({"text": "hi"}) == 0.0
    assert "kokoro-tts" in tool.agent_skills
    assert tool.input_schema["required"] == ["text"]


def test_unavailable_without_model_files(tmp_path, monkeypatch):
    monkeypatch.setenv("KOKORO_MODEL_DIR", str(tmp_path))
    tool = KokoroTTS()
    assert tool.get_status() == ToolStatus.UNAVAILABLE
    result = tool.execute({"text": "hello"})
    assert not result.success
    assert "not available" in result.error


@pytest.mark.skipif(
    KokoroTTS().get_status() != ToolStatus.AVAILABLE,
    reason="Kokoro model files not installed",
)
def test_generates_wav_and_rejects_unknown_voice(tmp_path):
    tool = KokoroTTS()
    out = tmp_path / "speech.wav"
    result = tool.execute({"text": "Hello there.", "voice": "am_michael", "output_path": str(out)})
    assert result.success, result.error
    assert out.stat().st_size > 1000
    assert result.data["lang"] == "en-us"
    assert result.data["audio_duration_seconds"] > 0.3

    bad = tool.execute({"text": "Hi", "voice": "zz_nobody", "output_path": str(tmp_path / "x.wav")})
    assert not bad.success
    assert "Unknown Kokoro voice" in bad.error
