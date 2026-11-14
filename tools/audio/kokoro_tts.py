"""Kokoro local text-to-speech provider tool (ONNX runtime, fully offline).

Kokoro-82M is an open-weight (Apache-2.0) TTS model whose voices sound far
more natural than Piper's while still running on CPU. This tool uses the
``kokoro-onnx`` package, which needs no PyTorch.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)

_MODEL_FILE = "kokoro-v1.0.onnx"
_VOICES_FILE = "voices-v1.0.bin"
_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"

# Voice-name prefix -> espeak language code used for phonemization.
_LANG_BY_PREFIX = {
    "a": "en-us",
    "b": "en-gb",
    "e": "es",
    "f": "fr-fr",
    "h": "hi",
    "i": "it",
    "j": "ja",
    "p": "pt-br",
    "z": "cmn",
}


class KokoroTTS(BaseTool):
    name = "kokoro_tts"
    version = "0.1.0"
    tier = ToolTier.VOICE
    capability = "tts"
    provider = "kokoro"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.LOCAL

    dependencies = ["python:kokoro_onnx", "python:soundfile"]
    install_instructions = (
        "Install Kokoro TTS (offline, no API key):\n"
        "  pip install kokoro-onnx soundfile\n"
        "Then download the model files (~350 MB) into ~/.kokoro "
        "(or set KOKORO_MODEL_DIR):\n"
        f"  mkdir -p ~/.kokoro && cd ~/.kokoro\n"
        f"  curl -LO {_RELEASE}/{_MODEL_FILE}\n"
        f"  curl -LO {_RELEASE}/{_VOICES_FILE}"
    )
    agent_skills = ["kokoro-tts", "text-to-speech"]

    capabilities = [
        "text_to_speech",
        "offline_generation",
    ]
    supports = {
        "voice_cloning": False,
        "multilingual": True,
        "offline": True,
        "native_audio": True,
    }
    best_for = [
        "natural-sounding free offline narration",
        "english voiceover for short-form social video (TikTok, Reels, Shorts)",
        "privacy-sensitive local-only workflows",
    ]
    not_good_for = [
        "voice clone matching",
        "german narration (no German voices)",
        "fine emotion control",
    ]

    input_schema = {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {"type": "string"},
            "voice": {
                "type": "string",
                "default": "af_heart",
                "description": (
                    "Kokoro voice name, e.g. af_heart, af_bella, am_michael, "
                    "am_fenrir, bf_emma, bm_george. Prefix: a=US English, "
                    "b=UK English; f=female, m=male."
                ),
            },
            "voice_id": {"type": "string", "description": "Alias for voice."},
            "speed": {"type": "number", "default": 1.0, "minimum": 0.5, "maximum": 2.0},
            "lang": {
                "type": "string",
                "description": "espeak language code. Defaults from the voice prefix (a->en-us, b->en-gb).",
            },
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=2, ram_mb=1500, vram_mb=0, disk_mb=400, network_required=False
    )
    retry_policy = RetryPolicy(max_retries=1, retryable_errors=[])
    idempotency_key_fields = ["text", "voice", "speed", "lang"]
    side_effects = ["writes audio file to output_path"]
    user_visible_verification = ["Listen to generated audio for intelligibility and pacing"]

    _engine = None  # cached Kokoro instance (model load takes a few seconds)

    @staticmethod
    def _model_dir() -> Path:
        return Path(os.environ.get("KOKORO_MODEL_DIR", Path.home() / ".kokoro"))

    def _model_paths(self) -> tuple[Path, Path]:
        d = self._model_dir()
        return d / _MODEL_FILE, d / _VOICES_FILE

    def get_status(self) -> ToolStatus:
        try:
            import kokoro_onnx  # noqa: F401
            import soundfile  # noqa: F401
        except ImportError:
            return ToolStatus.UNAVAILABLE
        model, voices = self._model_paths()
        if model.is_file() and voices.is_file():
            return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def _get_engine(self):
        if KokoroTTS._engine is None:
            from kokoro_onnx import Kokoro

            model, voices = self._model_paths()
            KokoroTTS._engine = Kokoro(str(model), str(voices))
        return KokoroTTS._engine

    def list_voices(self) -> list[str]:
        return sorted(self._get_engine().get_voices())

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        if self.get_status() != ToolStatus.AVAILABLE:
            return ToolResult(success=False, error="Kokoro TTS not available. " + self.install_instructions)

        start = time.time()
        try:
            result = self._generate(inputs)
        except Exception as exc:
            return ToolResult(success=False, error=f"Kokoro TTS generation failed: {exc}")

        result.duration_seconds = round(time.time() - start, 2)
        return result

    def _generate(self, inputs: dict[str, Any]) -> ToolResult:
        import soundfile as sf

        text = inputs["text"]
        voice = inputs.get("voice") or inputs.get("voice_id") or "af_heart"
        speed = float(inputs.get("speed", 1.0))
        lang = inputs.get("lang") or _LANG_BY_PREFIX.get(voice[:1], "en-us")

        engine = self._get_engine()
        available = engine.get_voices()
        if voice not in available:
            return ToolResult(
                success=False,
                error=f"Unknown Kokoro voice '{voice}'. Available: {', '.join(sorted(available))}",
            )

        output_path = Path(inputs.get("output_path", "tts_output.wav"))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        samples, sample_rate = engine.create(text, voice=voice, speed=speed, lang=lang)

        if output_path.suffix.lower() == ".wav":
            sf.write(str(output_path), samples, sample_rate)
            fmt = "wav"
        else:
            if not shutil.which("ffmpeg"):
                return ToolResult(success=False, error="ffmpeg is required for non-WAV output")
            tmp_wav = output_path.with_suffix(".tmp.wav")
            sf.write(str(tmp_wav), samples, sample_rate)
            proc = subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-y", "-i", str(tmp_wav), str(output_path)],
                capture_output=True,
                text=True,
            )
            tmp_wav.unlink(missing_ok=True)
            if proc.returncode != 0:
                return ToolResult(success=False, error=f"ffmpeg conversion failed: {proc.stderr}")
            fmt = output_path.suffix.lstrip(".").lower()

        audio_seconds = round(len(samples) / sample_rate, 3)
        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "model": "kokoro-v1.0",
                "voice": voice,
                "lang": lang,
                "speed": speed,
                "text_length": len(text),
                "audio_duration_seconds": audio_seconds,
                "output": str(output_path),
                "format": fmt,
            },
            artifacts=[str(output_path)],
            model="kokoro-v1.0",
            cost_usd=0.0,
        )
