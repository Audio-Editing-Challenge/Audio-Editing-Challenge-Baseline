"""LLM router: turn one MMAE sample into an edited audio clip via a tool-calling loop.

The router advertises five edit tools (speed / volume / pitch / separate /
generative_edit) plus an internal `finish` control tool to an OpenAI-compatible
LLM, then executes whatever the model calls, chaining edits on a single
"current audio" path. It is defensive by design: per-sample failures never abort
a run, and there is a deterministic keyword fallback when the LLM under-triggers.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from agent.audio_utils import audio_info
from agent.config import RouterConfig
from agent.clients.sam_audio_client import SAMAudioClient, SEPARATE_SCHEMA
from agent.clients.auk_client import AuKClient, GENERATIVE_EDIT_SCHEMA
from agent.tools import DSP_TOOLS, DSP_TOOL_SCHEMAS

FINISH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "Call this when the edit is complete and the current audio is the final result. Takes no arguments.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

SYSTEM_PROMPT = (
    "You are an audio-editing router. You are given ONE natural-language edit instruction "
    "and a 'current audio' clip. Achieve the instruction by calling the provided tools, then "
    "call `finish`. Rules:\n"
    "- Each tool operates on the CURRENT audio and replaces it with its result; chain tools "
    "for multi-step edits (e.g. separate, then change volume).\n"
    "- Use `speed`/`volume`/`pitch` for deterministic signal edits.\n"
    "- Use `separate` to isolate (stem='target') or remove (stem='residual') a specific sound.\n"
    "- Use `generative_edit` for anything that creates/replaces content or changes length "
    "(add, remove-by-resynthesis, replace, denoise, emotion/style change).\n"
    "- Prefer the simplest tool that satisfies the instruction. Do not over-edit.\n"
    "- When the current audio already satisfies the instruction, call `finish` immediately."
)


@dataclass
class RouteResult:
    output_path: str | None
    changed: bool
    trace: dict[str, Any] = field(default_factory=dict)


class Router:
    def __init__(
        self,
        config: RouterConfig | None = None,
        sam_client: Any | None = None,
        auk_client: Any | None = None,
        llm_client: Any | None = None,
    ):
        self.cfg = config or RouterConfig()
        self._sam = sam_client
        self._auk = auk_client
        self._llm = llm_client

    # ------------------------------------------------------------------ lazy deps
    @property
    def sam(self):
        if self._sam is None:
            self._sam = SAMAudioClient(
                self.cfg.services.sam_audio_url, timeout=self.cfg.services.http_timeout
            )
        return self._sam

    @property
    def auk(self):
        if self._auk is None:
            self._auk = AuKClient(
                self.cfg.services.auk_url, timeout=self.cfg.services.http_timeout
            )
        return self._auk

    @property
    def llm(self):
        if self._llm is None:
            from openai import OpenAI  # lazy: only needed for the real backend

            key = self.cfg.llm.api_key
            if not key:
                raise RuntimeError(
                    f"LLM API key env '{self.cfg.llm.api_key_env}' is not set; "
                    "export it or inject a fake llm_client for testing."
                )
            self._llm = OpenAI(
                base_url=self.cfg.llm.base_url,
                api_key=key,
                timeout=self.cfg.llm.timeout,
                max_retries=self.cfg.llm.max_retries,
            )
        return self._llm

    def tools(self) -> list[dict]:
        return [
            *DSP_TOOL_SCHEMAS,
            SEPARATE_SCHEMA,
            GENERATIVE_EDIT_SCHEMA,
            FINISH_SCHEMA,
        ]

    # ------------------------------------------------------------------ MMAE parse
    @staticmethod
    def parse_sample(
        sample: dict, wav_root: str
    ) -> tuple[str, str | None, list[str], str | None]:
        """Return (instruction, primary_input_abs_path, all_input_abs_paths, reduction_note)."""
        texts, audios = [], []
        for m in sample.get("messages", []):
            if m.get("role") != "user":
                continue
            content = m.get("content")
            if not isinstance(content, list):
                continue
            for c in content:
                if not isinstance(c, dict):
                    continue
                if c.get("type") == "text" and c.get("text"):
                    texts.append(c["text"].strip())
                elif c.get("type") == "audio":
                    url = c.get("audio_url") or c.get("audio")
                    if url:
                        audios.append(url)
        instruction = " ".join(t for t in texts if t)

        def _abs(u: str) -> str:
            return u if os.path.isabs(u) else os.path.abspath(os.path.join(wav_root, u))

        abs_audios = [_abs(u) for u in audios]
        reduction = None
        n_user_turns = sum(
            1 for m in sample.get("messages", []) if m.get("role") == "user"
        )
        if len(abs_audios) > 1 or n_user_turns > 1:
            reduction = (
                f"reduced: {len(abs_audios)} input audio(s), {n_user_turns} user turn(s) "
                "-> use primary audio + combined instruction"
            )
        primary = abs_audios[0] if abs_audios else None
        return instruction, primary, abs_audios, reduction

    # ------------------------------------------------------------------ main entry
    def route(self, sample: dict, wav_root: str, work_dir: str) -> RouteResult:
        sid = sample.get("id", "unknown")
        instruction, primary, all_audios, reduction = self.parse_sample(
            sample, wav_root
        )
        trace: dict[str, Any] = {
            "id": sid,
            "instruction": instruction,
            "input_audio": primary,
            "num_input_audios": len(all_audios),
            "reduction": reduction,
            "steps": [],
            "fallback_used": None,
            "budget_hit": False,
            "error": None,
        }

        if not instruction:
            trace["error"] = "empty instruction"
            return RouteResult(output_path=primary, changed=False, trace=trace)
        if not primary or not os.path.isfile(primary):
            trace["error"] = f"input audio not found: {primary}"
            return RouteResult(output_path=None, changed=False, trace=trace)

        os.makedirs(work_dir, exist_ok=True)
        state = {"current": primary, "changed": False, "step": 0}

        try:
            self._run_llm_loop(instruction, primary, work_dir, state, trace)
        except Exception as exc:  # LLM/backend failure -> keyword fallback
            trace["steps"].append({"stage": "llm_loop_error", "error": repr(exc)})

        # If the LLM produced no successful edit, fall back to a deterministic classifier.
        if not state["changed"]:
            self._keyword_fallback(instruction, work_dir, state, trace)

        return RouteResult(
            output_path=state["current"], changed=state["changed"], trace=trace
        )

    # ------------------------------------------------------------------ LLM loop
    def _run_llm_loop(self, instruction, primary, work_dir, state, trace) -> None:
        dur, sr, ch = audio_info(primary)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Instruction: {instruction}\n"
                    f"Current audio: duration={dur:.2f}s, sample_rate={sr}, channels={ch}.\n"
                    "Apply the instruction with the tools, then call finish."
                ),
            },
        ]

        extra_body = {}
        if self.cfg.llm.disable_thinking and "deepseek" in self.cfg.llm.base_url:
            extra_body = {"thinking": {"type": "disabled"}}

        for _ in range(self.cfg.max_tool_calls):
            resp = self.llm.chat.completions.create(
                model=self.cfg.llm.model,
                messages=messages,
                tools=self.tools(),
                tool_choice=self.cfg.llm.tool_choice,
                temperature=self.cfg.llm.temperature,
                extra_body=extra_body or None,
            )
            msg = resp.choices[0].message
            tool_calls = list(getattr(msg, "tool_calls", None) or [])

            # DeepSeek-V4 quirk: tool call sometimes serialized into content.
            if not tool_calls and getattr(msg, "content", None):
                parsed = self._parse_content_toolcall(msg.content)
                if parsed:
                    trace["steps"].append(
                        {
                            "stage": "recovered_content_toolcall",
                            "calls": [p["name"] for p in parsed],
                        }
                    )
                    # Execute the recovered batch ONCE and stop. We cannot append a
                    # protocol-valid tool result (there is no tool_call_id for a
                    # content-serialized call), so with temperature=0 the model would
                    # otherwise re-emit and re-apply the same edit every round until
                    # the budget is hit. Applying once is the safe deterministic choice.
                    self._dispatch_parsed(parsed, work_dir, state, trace)
                    return
                # plain text with no tool call -> model considers itself done
                trace["steps"].append(
                    {"stage": "llm_text_stop", "content": msg.content[:200]}
                )
                return

            if not tool_calls:
                trace["steps"].append({"stage": "no_tool_call"})
                return

            # append the assistant turn (with its tool_calls) before tool results
            messages.append(
                msg.model_dump()
                if hasattr(msg, "model_dump")
                else {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [tc.model_dump() for tc in tool_calls],
                }
            )
            finish_requested = False
            for tc in tool_calls:
                name = tc.function.name
                if name == "finish":
                    finish_requested = True
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": "ok: finishing",
                        }
                    )
                    continue
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}
                result_msg = self._exec_tool(name, args, work_dir, state, trace)
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result_msg}
                )
            # `finish` is terminal, but only AFTER executing every real edit in the batch
            if finish_requested:
                trace["steps"].append({"stage": "finish"})
                return
        else:
            trace["budget_hit"] = True

    def _dispatch_parsed(self, parsed: list[dict], work_dir, state, trace) -> bool:
        """Execute recovered tool calls (edits first, then finish). Returns True if `finish` seen."""
        finish = False
        for p in parsed:
            if p["name"] == "finish":
                finish = True
                continue
            self._exec_tool(p["name"], p.get("args", {}), work_dir, state, trace)
        if finish:
            trace["steps"].append({"stage": "finish"})
        return finish

    # ------------------------------------------------------------------ execution
    def _exec_tool(self, name, args, work_dir, state, trace) -> str:
        state["step"] += 1
        step = state["step"]
        current = state["current"]
        out = os.path.join(work_dir, f"step{step}_{name}.wav")
        entry = {"stage": "tool", "step": step, "tool": name, "args": args}
        try:
            if name in DSP_TOOLS:
                DSP_TOOLS[name](current, out, **args)
            elif name == "separate":
                self.sam.separate(
                    current,
                    out,
                    description=args.get("description", ""),
                    stem=args.get("stem", "target"),
                )
            elif name == "generative_edit":
                res = self.auk.edit(
                    current,
                    args.get("instruction", ""),
                    out,
                    gen_seconds=args.get("gen_seconds"),
                    seed=self.cfg.seed,
                )
                entry["enhanced_instruction"] = res.enhanced_instruction
                entry["gen_seconds"] = res.gen_seconds
                entry["task_type"] = res.task_type
            else:
                entry["error"] = f"unknown tool {name!r}"
                trace["steps"].append(entry)
                return f"error: unknown tool {name!r}"

            state["current"] = out
            state["changed"] = True
            entry["output"] = out
            trace["steps"].append(entry)
            return f"ok: wrote {os.path.basename(out)}"
        except (
            Exception
        ) as exc:  # per-tool isolation: record and let the loop/fallback continue
            entry["error"] = repr(exc)
            trace["steps"].append(entry)
            return f"error: {exc}"

    # ------------------------------------------------------------------ fallbacks
    @staticmethod
    def _parse_content_toolcall(content: str) -> list[dict]:
        """Best-effort recovery of tool call(s) embedded in message content.

        Scans for balanced JSON objects with json.raw_decode so it handles BOTH
        code-fenced JSON and raw inline JSON with a NESTED `arguments` object
        (a plain `[^{}]` regex cannot cross the inner braces).
        """
        out: list[dict] = []
        decoder = json.JSONDecoder()
        i, n = 0, len(content)
        while i < n:
            if content[i] != "{":
                i += 1
                continue
            try:
                obj, end = decoder.raw_decode(content, i)
            except ValueError:
                i += 1
                continue
            i = max(end, i + 1)
            if not isinstance(obj, dict):
                continue
            name = obj.get("name")
            if not name:
                continue
            args = (
                obj.get("arguments") or obj.get("args") or obj.get("parameters") or {}
            )
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            if not isinstance(args, dict):
                args = {}
            out.append({"name": name, "args": args})
        return out

    def _keyword_fallback(self, instruction, work_dir, state, trace) -> None:
        name, args = self._classify(instruction)
        trace["fallback_used"] = {"tool": name, "args": args}
        self._exec_tool(name, args, work_dir, state, trace)

    @staticmethod
    def _classify(instruction: str) -> tuple[str, dict]:
        text = instruction.lower()

        def _num(default):
            m = re.search(r"(\d+(?:\.\d+)?)", text)
            return float(m.group(1)) if m else default

        has_pct = "%" in text

        if any(k in text for k in ("faster", "speed up", "quicker", "tempo up")):
            f = _num(1.5)
            if has_pct:  # "50% faster" -> 1.5
                f = 1.0 + f / 100.0
            if not (1.0 < f <= 4.0):  # clamp implausible bare numbers (e.g. "50")
                f = 1.5
            return "speed", {"factor": f}
        if any(k in text for k in ("slower", "slow down", "tempo down")):
            f = _num(0.75)
            if has_pct:  # "20% slower" -> 0.8
                f = max(0.05, 1.0 - f / 100.0)
            if not (0.25 <= f < 1.0):
                f = 0.75
            return "speed", {"factor": f}
        if any(
            k in text
            for k in ("louder", "increase the volume", "raise the volume", "amplify")
        ):
            return "volume", {"gain_db": 6.0}
        if any(
            k in text
            for k in (
                "quieter",
                "lower the volume",
                "decrease the volume",
                "softer",
                "attenuate",
            )
        ):
            return "volume", {"gain_db": -6.0}
        if (
            "pitch" in text
            or "semitone" in text
            or "deeper" in text
            or "higher voice" in text
        ):
            n = min(abs(_num(2.0)), 12.0)  # clamp to +-1 octave
            sign = (
                -1.0 if ("lower" in text or "down" in text or "deeper" in text) else 1.0
            )
            return "pitch", {"semitones": sign * n}
        if any(
            k in text for k in ("remove", "delete", "get rid", "without the", "erase")
        ):
            return "separate", {"description": instruction, "stem": "residual"}
        if any(
            k in text
            for k in ("extract", "isolate", "keep only", "separate", "only the")
        ):
            return "separate", {"description": instruction, "stem": "target"}
        # most general fallback: let AuK handle it
        return "generative_edit", {"instruction": instruction}
