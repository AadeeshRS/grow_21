"""Optional neural text-to-speech for the game module's spoken instructions.

Uses Piper (offline, open-source neural TTS) when `piper-tts` is installed and a voice model is
available; otherwise reports itself unavailable and the app falls back to the device voice.
Voice model: GROW21_TTS_VOICE (path to a Piper .onnx file), default voices/en_US-lessac-high.onnx.
"""

from __future__ import annotations

import io
import os
import threading
import wave
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from ..paths import PACKAGE_ROOT

DEFAULT_VOICE = PACKAGE_ROOT / "voices" / "en_US-lessac-high.onnx"
MAX_CHARS = 300


class NeuralVoice:
    def __init__(self, model_path: Optional[Path] = None, cache_size: int = 256):
        self.model_path = Path(model_path or os.environ.get("GROW21_TTS_VOICE") or DEFAULT_VOICE)
        self._voice = None
        self._lock = threading.Lock()
        self._cache: "OrderedDict[str, bytes]" = OrderedDict()
        self._cache_size = cache_size

    @property
    def available(self) -> bool:
        if not self.model_path.exists():
            return False
        try:
            import piper  # noqa: F401
        except ImportError:
            return False
        return True

    def synthesize(self, text: str) -> bytes:
        """Returns a WAV file for `text`. Raises RuntimeError when no voice is available."""
        text = " ".join(text.split())[:MAX_CHARS]
        with self._lock:
            if text in self._cache:
                self._cache.move_to_end(text)
                return self._cache[text]
            if self._voice is None:
                if not self.available:
                    raise RuntimeError("No neural voice installed")
                from piper import PiperVoice
                self._voice = PiperVoice.load(str(self.model_path))
            from piper import SynthesisConfig
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                # Slightly slower than default speech, for children who process speech more slowly.
                self._voice.synthesize_wav(text, w, syn_config=SynthesisConfig(length_scale=1.12, noise_scale=0.6, noise_w_scale=0.75))
            audio = buf.getvalue()
            self._cache[text] = audio
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
            return audio
