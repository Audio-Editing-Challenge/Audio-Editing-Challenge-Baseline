"""Small audio helpers shared by the DSP tools, clients, and router.

Kept dependency-light (soundfile + numpy; librosa only where needed) and
import-safe: no model loads or global side effects at import time.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf


def audio_info(path: str) -> tuple[float, int, int]:
    """Return (duration_seconds, sample_rate, channels) without loading samples."""
    info = sf.info(path)
    return float(info.duration), int(info.samplerate), int(info.channels)


def duration_seconds(path: str) -> float:
    return audio_info(path)[0]


def load(path: str) -> tuple[np.ndarray, int]:
    """Load audio as float32. Returns (y, sr) with y shape (n,) mono or (channels, n)."""
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        # soundfile gives (n, channels); transpose to (channels, n) for librosa axis=-1
        y = y.T
    return y, sr


def save(path: str, y: np.ndarray, sr: int) -> str:
    """Write float audio to `path`. Accepts (n,) or (channels, n); writes 16-bit PCM WAV."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        y = y.T  # back to (n, channels) for soundfile
    sf.write(path, y, sr, subtype="PCM_16")
    return path


def peak_limit(y: np.ndarray, ceiling: float = 0.999) -> np.ndarray:
    """Scale the whole signal down so its peak <= ceiling (no hard clipping)."""
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > ceiling:
        y = y * (ceiling / peak)
    return y
