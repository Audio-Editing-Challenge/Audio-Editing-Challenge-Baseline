"""Agent-track baseline package for the Audio Editing Challenge.

The router (`agent.router.Router`) turns one MMAE sample into an edited audio clip
by orchestrating five tools through an OpenAI-compatible LLM:

- `speed` / `volume` / `pitch` — local DSP (`agent.tools.dsp`)
- `separate`                   — SAM-Audio-Large HTTP service (`agent.clients.sam_audio_client`)
- `generative_edit`            — AuK + Prompt Enhancer HTTP service (`agent.clients.auk_client`)
"""

__version__ = "0.1.0"
