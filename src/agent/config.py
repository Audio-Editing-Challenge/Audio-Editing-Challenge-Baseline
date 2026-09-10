"""Central configuration for the agent-track baseline.

Everything the router and the runner need to reach the LLM and the two model
services lives here, sourced from environment variables with sensible defaults so
the exact endpoint / model string is a one-line override (never hardcoded).

Compliance note: the DEFAULT router LLM is DeepSeek's HOSTED API. The Audio
Editing Challenge Track-2 rule #1 forbids "commercial or hosted APIs" for a
final/audited submission. Because the client is OpenAI-compatible, pointing
`AGENT_LLM_BASE_URL` at a locally-served open-weights LLM (e.g. Qwen via vLLM)
is a drop-in compliant swap — no code change. See the README compliance note.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any


# DeepSeek's official API model id is `deepseek-v4-flash` (the product name
# "deepseek-flash" is NOT accepted by the chat endpoint). Override via env if
# your account exposes a different id.
DEFAULT_LLM_MODEL = "deepseek-v4-flash"
DEFAULT_LLM_BASE_URL = "https://api.deepseek.com"


@dataclass
class LLMConfig:
    """OpenAI-compatible chat LLM backing the router."""

    model: str = field(
        default_factory=lambda: os.environ.get("AGENT_LLM_MODEL", DEFAULT_LLM_MODEL)
    )
    base_url: str = field(
        default_factory=lambda: os.environ.get(
            "AGENT_LLM_BASE_URL", DEFAULT_LLM_BASE_URL
        )
    )
    api_key_env: str = field(
        default_factory=lambda: os.environ.get(
            "AGENT_LLM_API_KEY_ENV", "DEEPSEEK_API_KEY"
        )
    )
    # DeepSeek-V4: drive the router in NON-thinking mode. It is cheaper/faster and
    # avoids the "echo reasoning_content back every turn or get HTTP 400" rule that
    # applies when tools are present in thinking mode.
    disable_thinking: bool = field(
        default_factory=lambda: os.environ.get("AGENT_LLM_DISABLE_THINKING", "1") != "0"
    )
    temperature: float = field(
        default_factory=lambda: float(os.environ.get("AGENT_LLM_TEMPERATURE", "0.0"))
    )
    # "auto" lets the model call `finish` to end; the router also has a keyword
    # fallback if the model under-triggers tools.
    tool_choice: str = field(
        default_factory=lambda: os.environ.get("AGENT_LLM_TOOL_CHOICE", "auto")
    )
    timeout: float = field(
        default_factory=lambda: float(os.environ.get("AGENT_LLM_TIMEOUT", "120"))
    )
    max_retries: int = field(
        default_factory=lambda: int(os.environ.get("AGENT_LLM_MAX_RETRIES", "2"))
    )

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env)


@dataclass
class ServiceConfig:
    """URLs of the two model microservices the router calls over HTTP."""

    sam_audio_url: str = field(
        default_factory=lambda: os.environ.get("SAM_AUDIO_URL", "http://127.0.0.1:8305")
    )
    auk_url: str = field(
        default_factory=lambda: os.environ.get("AUK_URL", "http://127.0.0.1:8310")
    )
    http_timeout: float = field(
        default_factory=lambda: float(os.environ.get("AGENT_SERVICE_TIMEOUT", "600"))
    )


@dataclass
class RouterConfig:
    """Loop / behaviour knobs for the router."""

    max_tool_calls: int = field(
        default_factory=lambda: int(os.environ.get("AGENT_MAX_TOOL_CALLS", "6"))
    )
    seed: int = field(default_factory=lambda: int(os.environ.get("AGENT_SEED", "44")))
    llm: LLMConfig = field(default_factory=LLMConfig)
    services: ServiceConfig = field(default_factory=ServiceConfig)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # never serialize the actual key; just record which env var holds it
        d["llm"].pop("api_key", None)
        return d
