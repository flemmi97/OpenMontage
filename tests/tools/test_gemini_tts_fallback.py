"""Gemini TTS: generateContent fallback when the Interactions API cannot see a TTS model."""

from __future__ import annotations

import io
import wave

from tools.audio.gemini_tts import GeminiTTS


def test_prompt_separates_director_notes_from_transcript():
    prompt = GeminiTTS.build_generate_content_prompt({"text": "Hello there.", "style": "excited"})
    assert "### DIRECTOR'S NOTES\nexcited" in prompt
    assert prompt.rstrip().endswith("### TRANSCRIPT\nHello there.")
    assert "do not read them aloud" in prompt
    assert GeminiTTS.build_generate_content_prompt({"text": "Plain."}) == "Plain."


def test_prompt_for_multi_speaker_turns():
    prompt = GeminiTTS.build_generate_content_prompt(
        {"turns": [{"speaker": "A", "text": "Hi", "style": "warm"}, {"speaker": "B", "text": "Yo"}]}
    )
    assert "A (warm): Hi" in prompt and "B: Yo" in prompt


def test_falls_back_to_generate_content_on_not_found(tmp_path, monkeypatch):
    tool = GeminiTTS()

    class Interactions:
        def create(self, **kwargs):
            raise RuntimeError("Error code: 404 - Model 'gemini-3.8-flash-tts' not found")

    class Client:
        interactions = Interactions()

    monkeypatch.setattr("tools.google_credentials.get_genai_client", lambda **kw: Client())
    pcm = b"\x00\x01" * 2400
    monkeypatch.setattr(GeminiTTS, "_generate_content_fallback", lambda self, client, inputs: _wav(pcm))
    out = tmp_path / "x.wav"
    result = tool.execute({"text": "Hi", "voice_id": "Kore", "output_path": str(out)})
    assert result.success, result.error
    assert result.data["api_path"] == "generate_content_fallback"
    with wave.open(str(out)) as w:
        assert w.getframerate() == 24000 and w.getnframes() == 2400


def _wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(pcm)
    return buf.getvalue()
