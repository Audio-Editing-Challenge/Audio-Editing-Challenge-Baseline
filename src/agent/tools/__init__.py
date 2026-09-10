"""Signal-processing tools + their OpenAI-style tool schemas.

The router manages the "current working audio" path, so the schemas expose ONLY
the edit parameters (not input/output paths). `DSP_TOOLS` maps a tool name to a
callable `(in_path, out_path, **params) -> out_path`.
"""

from __future__ import annotations

from agent.tools import dsp

SPEED_SCHEMA = {
    "type": "function",
    "function": {
        "name": "speed",
        "description": (
            "Change the tempo/playback speed of the current audio by a factor "
            "(pitch preserved). factor>1 makes it faster/shorter, factor<1 "
            "slower/longer. Use for 'speed up', 'slow down', 'x1.5 faster' edits."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "factor": {
                    "type": "number",
                    "description": "Speed multiplier, e.g. 1.5 = 1.5x faster, 0.75 = slower. Must be > 0.",
                }
            },
            "required": ["factor"],
        },
    },
}

VOLUME_SCHEMA = {
    "type": "function",
    "function": {
        "name": "volume",
        "description": (
            "Change the loudness of the current audio by a gain in decibels "
            "(duration/pitch unchanged, clipping-safe). Positive = louder, "
            "negative = quieter. Use for 'increase/lower the volume', 'make it louder'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "gain_db": {
                    "type": "number",
                    "description": "Gain in dB, e.g. +6 louder, -6 quieter.",
                }
            },
            "required": ["gain_db"],
        },
    },
}

PITCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "pitch",
        "description": (
            "Shift the pitch of the current audio by a number of semitones "
            "(duration unchanged). Positive = higher, negative = lower. Use for "
            "'raise/lower the pitch', 'up 2 semitones', 'make voice deeper'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "semitones": {
                    "type": "number",
                    "description": "Pitch shift in semitones, e.g. +2 higher, -3 lower.",
                }
            },
            "required": ["semitones"],
        },
    },
}

DSP_TOOL_SCHEMAS = [SPEED_SCHEMA, VOLUME_SCHEMA, PITCH_SCHEMA]

# name -> callable(in_path, out_path, **params) -> out_path
DSP_TOOLS = {
    "speed": dsp.speed,
    "volume": dsp.volume,
    "pitch": dsp.pitch,
}

__all__ = [
    "dsp",
    "SPEED_SCHEMA",
    "VOLUME_SCHEMA",
    "PITCH_SCHEMA",
    "DSP_TOOL_SCHEMAS",
    "DSP_TOOLS",
]
