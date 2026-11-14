---
name: kokoro-tts
description: Generate natural-sounding narration fully offline with Kokoro-82M (ONNX). Use for free English voiceover, especially short-form social video (TikTok, Reels, Shorts), when no paid TTS key is configured or the user wants local-only audio.
---

# Kokoro TTS (offline)

Tool: `kokoro_tts` (route through `tts_selector`, `preferred_provider: "kokoro"`).
No API key, no cost, no network at generation time. Runs on CPU at roughly 2–5x real time.

## Setup

```bash
pip install kokoro-onnx soundfile
mkdir -p ~/.kokoro && cd ~/.kokoro
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

Override the folder with `KOKORO_MODEL_DIR`.

## Voices

Names are `<lang><gender>_<name>`: `a` = US English, `b` = UK English, `f` = female, `m` = male.
The language for phonemization is derived from the prefix automatically.

| Voice | Character | Good for |
|---|---|---|
| `af_heart` | warm, clear US female (best overall) | default narration, lifestyle, tips |
| `af_bella` | bright, energetic US female | hooks, upbeat listicles |
| `am_michael` | calm, confident US male | explainers, storytelling |
| `am_fenrir` | stronger, punchier US male | hype, facts, motivation |
| `bf_emma` | polished UK female | premium, educational |
| `bm_george` | deep UK male | documentary, mystery |

`KokoroTTS().list_voices()` returns the full installed list (also es, fr, it, ja, pt, hi, zh voices).
There are no German voices — use Piper for German.

## Parameters

- `voice` — see above (default `af_heart`). `voice_id` is accepted as an alias.
- `speed` — 0.5–2.0. Short-form social: **1.05–1.15** keeps energy up without slurring. Explainers: 1.0.
- `output_path` — `.wav` writes directly; `.mp3`/other extensions convert via ffmpeg.

## Writing text that sounds good

- Punctuation drives prosody: commas add short pauses, periods add longer ones, `!` lifts energy, `?` rises.
- Use `...` or a new sentence for a deliberate beat before a reveal.
- Spell out awkward tokens: numbers that should be read a certain way ("ten times" not "10x"), acronyms ("A I" or "AI" both work; test).
- Keep sentences short (≤ 15 words) for short-form video — it also makes caption chunks cleaner.
- Generate per scene/section rather than one giant block so timing can be adjusted per cut.

## Captions

Kokoro returns no word timestamps. For word-level captions, run the generated narration through the
`transcriber` tool (faster-whisper, offline) to get word timings, then burn captions in composition.
