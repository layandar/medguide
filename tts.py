"""Text-to-speech with Kokoro (runs locally via ONNX, no API key).

Model files (download once, see README):
  data/kokoro-v1.0.onnx
  data/voices-v1.0.bin

Kokoro covers English, Japanese, Chinese, French, Italian, Portuguese, Spanish
and Hindi -- not Arabic. For other languages, route to Piper or MMS-TTS here.
"""

import io
import threading
from pathlib import Path

import soundfile as sf

DATA = Path(__file__).parent / "data"
MODEL_PATH = DATA / "kokoro-v1.0.onnx"
VOICES_PATH = DATA / "voices-v1.0.bin"

_kokoro = None
_lock = threading.Lock()

# language code -> (kokoro lang, voice). Add Piper/MMS routes for other languages later.
VOICES = {
    "en": ("en-us", "af_heart"),
    "en-gb": ("en-gb", "bf_emma"),
    "es": ("es", "ef_dora"),
    "fr": ("fr-fr", "ff_siwis"),
}


def available() -> bool:
    return MODEL_PATH.exists() and VOICES_PATH.exists()


def _load():
    global _kokoro
    with _lock:
        if _kokoro is None:
            from kokoro_onnx import Kokoro  # imported lazily so the app starts without the model

            _kokoro = Kokoro(str(MODEL_PATH), str(VOICES_PATH))
    return _kokoro


def synthesize_wav(text: str, lang: str = "en", speed: float = 0.9) -> bytes:
    """Return WAV bytes. Slightly slower than normal speed helps comprehension."""
    if lang not in VOICES:
        raise ValueError(f"Kokoro has no voice for '{lang}'. Add Piper/MMS routing for this language.")
    k_lang, voice = VOICES[lang]
    kokoro = _load()
    samples, sample_rate = kokoro.create(text, voice=voice, speed=speed, lang=k_lang)
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV")
    return buf.getvalue()
