"""Deterministic signal-processing edit tools: speed, volume, pitch.

Pure DSP (librosa + numpy), no learned model. Each function takes an input path
and an output path, validates its parameter, writes the result, and returns the
output path. Import-safe: no global side effects.
"""

from __future__ import annotations

import librosa
import numpy as np

from agent.audio_utils import load, save, peak_limit


def _apply_per_channel(y: np.ndarray, fn) -> np.ndarray:
    """Apply a mono effect `fn(mono)->mono` to (n,) or (channels, n) audio."""
    if y.ndim == 1:
        return fn(y)
    return np.stack([fn(y[c]) for c in range(y.shape[0])], axis=0)


def speed(in_path: str, out_path: str, factor: float) -> str:
    """Change tempo by `factor` (pitch-preserving time-stretch).

    factor > 1 -> faster / shorter; factor < 1 -> slower / longer.
    """
    factor = float(factor)
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError(
            f"speed factor must be a positive finite number, got {factor!r}"
        )
    y, sr = load(in_path)
    stretched = _apply_per_channel(
        y, lambda m: librosa.effects.time_stretch(m, rate=factor)
    )
    return save(out_path, stretched, sr)


def volume(in_path: str, out_path: str, gain_db: float) -> str:
    """Change loudness by `gain_db` decibels (duration/pitch preserved, clipping-safe)."""
    gain_db = float(gain_db)
    if not np.isfinite(gain_db):
        raise ValueError(f"gain_db must be a finite number, got {gain_db!r}")
    y, sr = load(in_path)
    scaled = y * (10.0 ** (gain_db / 20.0))
    scaled = peak_limit(scaled)  # only attenuates if a boost would clip
    return save(out_path, scaled, sr)


def pitch(in_path: str, out_path: str, semitones: float) -> str:
    """Shift pitch by `semitones` (duration preserved)."""
    semitones = float(semitones)
    if not np.isfinite(semitones):
        raise ValueError(f"semitones must be a finite number, got {semitones!r}")
    y, sr = load(in_path)
    shifted = _apply_per_channel(
        y, lambda m: librosa.effects.pitch_shift(m, sr=sr, n_steps=semitones)
    )
    return save(out_path, shifted, sr)
