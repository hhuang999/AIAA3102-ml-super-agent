"""The single exit point for all LLM calls.

Responsibilities: choose the parameter group for a role, send the request to the
endpoint, retry API errors, validate structured answers, and append one
line per request to ``logs/llm_calls.jsonl``.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from ..errors import LLMUnavailableError, StructuredOutputError
from ..utils import append_jsonl, new_id, now_iso
from .jsonparse import parse_json_object
from .schema import Schema, describe, validate

# Role name -> (parameter group, call kind).
ROLES: dict[str, tuple[str, str]] = {
    "desc_cleaner": ("judgement", "text"),
    "direction_judge": ("judgement", "structured"),
    "draft_writer.data_prep": ("generation", "text"),
    "draft_writer.model_def": ("generation", "text"),
    "draft_writer.train_eval": ("generation", "text"),
    "draft_integrator": ("generation", "text"),
    "planner.free": ("generation", "text"),
    "planner.structured": ("generation", "structured"),
    "patch_coder": ("generation", "text"),
    "script_coder": ("generation", "text"),
    "repair_coder": ("generation", "text"),
    "reviewer": ("judgement", "structured"),
    "result_reader": ("judgement", "structured"),
    "leak_checker": ("judgement", "structured"),
    "format_fixer": ("judgement", "structured"),
}

ABORT_POLL_SECONDS = 0.5
# A reply cut off by the output limit while thinking is retried with the next lower reasoning effort.
LOWER_EFFORT = {"max": "high", "high": "low"}
NON_RETRYABLE_STATUS = frozenset({400, 401, 403, 404, 422})
# Roles that assess an already finished run; they may keep working into the finalize reserve.
ASSESSMENT_ROLES = frozenset({"result_reader", "leak_checker", "format_fixer"})


@dataclass
class TransportResponse:
    content: str
    finish_reason: str | None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None


class TransientAPIError(Exception):
    """Network error, timeout, server error or rate limit: worth retrying."""


class FatalAPIError(Exception):
    """A request the endpoint will never accept (bad key, bad parameters)."""


class Transport(Protocol):
    def __call__(self, request: Mapping[str, Any]) -> TransportResponse: ...


@dataclass
class TextResult:
    content: str
    truncated: bool


class OpenAITransport:
    """Sends chat completion requests to an OpenAI-compatible endpoint."""

    def __init__(self, base_url: str, api_key: str, timeout: float) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)

    def __call__(self, request: Mapping[str, Any]) -> TransportResponse:
        import openai

        kwargs: dict[str, Any] = {
            "model": request["model"],
            "messages": request["messages"],
            "max_tokens": request["max_tokens"],
            "temperature": request["temperature"],
        }
        extra: dict[str, Any] = {"thinking": {"type": "enabled" if request["thinking"] else "disabled"}}
        if request["thinking"] and request.get("reasoning_effort"):
            extra["reasoning_effort"] = request["reasoning_effort"]
        kwargs["extra_body"] = extra
        if request.get("json_mode"):
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError) as exc:
            raise TransientAPIError(f"{type(exc).__name__}: {exc}") from exc
        except openai.APIStatusError as exc:
            if exc.status_code in NON_RETRYABLE_STATUS:
                raise FatalAPIError(f"HTTP {exc.status_code}: {exc.message}") from exc
            raise TransientAPIError(f"HTTP {exc.status_code}: {exc.message}") from exc
        except openai.APIError as exc:
            raise TransientAPIError(f"{type(exc).__name__}: {exc}") from exc
        if not resp.choices:
            raise TransientAPIError("empty response (no choices)")
        choice = resp.choices[0]
        content = choice.message.content or ""
        usage = resp.usage
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        reasoning_tokens = None
        details = getattr(usage, "completion_tokens_details", None) if usage else None
        if details is not None:
            reasoning_tokens = getattr(details, "reasoning_tokens", None)
        return TransportResponse(
            content=content,
            finish_reason=choice.finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
        )


def call_cost(group: Mapping[str, Any], prompt_tokens: int | None, completion_tokens: int | None) -> float | None:
    """Cost of one request from the per-million-token prices of its parameter group. None without usage data."""
    if prompt_tokens is None and completion_tokens is None:
        return None
    return ((prompt_tokens or 0) * float(group.get("input_price_per_million", 0.0))
            + (completion_tokens or 0) * float(group.get("output_price_per_million", 0.0))) / 1e6


def abort_decision(role: str, seconds_left: float, reserve: float, interrupted: bool, hard_stop: bool) -> bool:
    """Whether a call for ``role`` must be abandoned now.

    Roles that assess an already finished run may keep working until half of the finalize
    reserve is used; all other (generating) roles stop when the reserve begins.
    """
    if role in ASSESSMENT_ROLES:
        return interrupted or seconds_left <= reserve / 2.0
    return hard_stop or seconds_left <= reserve


class LLMGateway:
    """All LLM traffic goes through an instance of this class."""

    def __init__(
        self,
        llm_cfg: Mapping[str, Any],
        calls_log_path: str | Path | None,
        transport: Transport,
        should_abort: Callable[[str], bool] | None = None,
        prompt_log_dir: Callable[[str], Path | None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cfg = llm_cfg
        self.calls_log_path = Path(calls_log_path) if calls_log_path else None
        self.transport = transport
        self.should_abort = should_abort or (lambda role: False)
        self.prompt_log_dir = prompt_log_dir
        self._sleep = sleep
        self._lock = threading.Lock()
        self.totals: dict[str, float] = {"calls": 0, "failed_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}

    # ------------------------------------------------------------------ helpers
    def _group(self, role: str) -> Mapping[str, Any]:
        group = ROLES.get(role, ("generation", "text"))[0]
        return self.cfg[group]

    def _cost(self, group: Mapping[str, Any], pt: int | None, ct: int | None) -> float | None:
        return call_cost(group, pt, ct)

    def _log(self, row: dict[str, Any]) -> None:
        with self._lock:
            self.totals["calls"] += 1
            if not row["ok"]:
                self.totals["failed_calls"] += 1
            self.totals["prompt_tokens"] += row.get("prompt_tokens") or 0
            self.totals["completion_tokens"] += row.get("completion_tokens") or 0
            self.totals["cost"] += row.get("cost") or 0.0
        if self.calls_log_path is not None:
            append_jsonl(self.calls_log_path, row)

    def _log_prompt(self, node_id: str | None, role: str, attempt: int, messages: list[dict[str, str]]) -> None:
        if not node_id or self.prompt_log_dir is None:
            return
        try:
            d = self.prompt_log_dir(node_id)
            if d is None:
                return
            append_jsonl(Path(d) / "prompts.jsonl", {"ts": now_iso(), "agent": role, "attempt": attempt, "messages": messages})
        except OSError:
            pass

    def _call_transport(self, request: Mapping[str, Any]) -> TransportResponse:
        """Run the transport in a helper thread so a hard deadline can abandon it."""
        result: dict[str, Any] = {}
        done = threading.Event()

        def work() -> None:
            try:
                result["value"] = self.transport(request)
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                result["error"] = exc
            finally:
                done.set()

        t = threading.Thread(target=work, daemon=True)
        t.start()
        while not done.wait(ABORT_POLL_SECONDS):
            if self.should_abort(str(request.get("role", ""))):
                raise LLMUnavailableError("abandoned: deadline reached during generation")
        if "error" in result:
            raise result["error"]
        return result["value"]

    def _request_once(
        self,
        role: str,
        messages: list[dict[str, str]],
        json_mode: bool,
        node_id: str | None,
        call_id: str,
        attempt_counter: list[int],
        effort: str | None = None,
    ) -> TransportResponse:
        """Send one logical request, retrying API errors. Logs one line per attempt."""
        group = self._group(role)
        thinking = str(group.get("thinking", "disabled")) == "enabled"
        effort = effort or group.get("reasoning_effort")
        request = {
            "role": role,
            "model": group["model"],
            "messages": messages,
            "max_tokens": int(group["max_output_tokens"]),
            "temperature": float(group["temperature"]),
            "thinking": thinking,
            "reasoning_effort": effort,
            "json_mode": json_mode,
        }
        max_retries = int(self.cfg.get("api_max_retries", 20))
        delay = float(self.cfg.get("api_retry_delay_seconds", 3))
        api_failures = 0
        while True:
            if self.should_abort(role):
                raise LLMUnavailableError("deadline reached")
            attempt_counter[0] += 1
            attempt = attempt_counter[0]
            self._log_prompt(node_id, role, attempt, messages)
            t0 = time.monotonic()
            base_row = {
                "ts": now_iso(),
                "call_id": call_id,
                "attempt": attempt,
                "agent": role,
                "node_id": node_id,
                "model": group["model"],
                "thinking": thinking,
                "reasoning_effort": effort if thinking else None,
            }
            try:
                resp = self._call_transport(request)
            except (TransientAPIError, FatalAPIError) as exc:
                self._log({**base_row, "prompt_tokens": None, "completion_tokens": None, "reasoning_tokens": None,
                           "latency_secs": round(time.monotonic() - t0, 3), "ok": False, "error": str(exc)[:300],
                           "cost": None})
                if isinstance(exc, FatalAPIError):
                    raise LLMUnavailableError(str(exc)) from exc
                api_failures += 1
                if api_failures > max_retries:
                    raise LLMUnavailableError(f"LLM unavailable after {max_retries} retries: {exc}") from exc
                waited = 0.0
                while waited < delay:
                    if self.should_abort(role):
                        raise LLMUnavailableError("deadline reached while waiting to retry") from exc
                    step = min(ABORT_POLL_SECONDS, delay - waited)
                    self._sleep(step)
                    waited += step
                continue
            except LLMUnavailableError as exc:
                self._log({**base_row, "prompt_tokens": None, "completion_tokens": None, "reasoning_tokens": None,
                           "latency_secs": round(time.monotonic() - t0, 3), "ok": False, "error": str(exc)[:300],
                           "cost": None})
                raise
            resp_row = {
                **base_row,
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
                "reasoning_tokens": resp.reasoning_tokens,
                "latency_secs": round(time.monotonic() - t0, 3),
                "cost": self._cost(group, resp.prompt_tokens, resp.completion_tokens),
            }
            resp._log_row = resp_row  # type: ignore[attr-defined]
            return resp

    # ------------------------------------------------------------------ public API
    def _lower_effort(self, role: str, current: str | None) -> str | None:
        """The next lower reasoning effort for a thinking role, or None."""
        group = self._group(role)
        if not self.cfg.get("lower_effort_on_truncation", True) or str(group.get("thinking", "disabled")) != "enabled":
            return None
        return LOWER_EFFORT.get(str(current or group.get("reasoning_effort")))

    def text(self, role: str, system: str, user: str, node_id: str | None = None) -> TextResult:
        """Plain text call. A reply cut off by the length limit is returned with ``truncated``.

        An empty reply cut off by the limit (the reasoning used the whole output budget) is retried
        once with a lower reasoning effort.
        """
        call_id = new_id()
        counter = [0]
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        effort: str | None = None
        while True:
            resp = self._request_once(role, messages, False, node_id, call_id, counter, effort)
            truncated = resp.finish_reason == "length"
            row = resp._log_row  # type: ignore[attr-defined]
            ok = bool(resp.content.strip())
            lower = self._lower_effort(role, effort) if (not ok and truncated) else None
            error = None if ok else ("empty reply; retrying with reasoning effort " + lower if lower else "empty reply")
            self._log({**row, "ok": ok, "error": error})
            if lower is None:
                return TextResult(content=resp.content, truncated=truncated)
            effort = lower

    def structured(
        self,
        role: str,
        system: str,
        user: str,
        schema: Schema,
        node_id: str | None = None,
        extra_check: Callable[[dict[str, Any], bool], tuple[dict[str, Any], list[str]]] | None = None,
        max_attempts: int | None = None,
    ) -> dict[str, Any]:
        """Structured call. ``extra_check(obj, is_last_attempt)`` may normalise and add errors."""
        call_id = new_id()
        counter = [0]
        attempts = int(max_attempts or self.cfg.get("json_max_attempts", 3))
        base_user = user if "json" in user.lower() else user + "\n\nAnswer in JSON."
        base_user = base_user + "\n\n" + describe(schema)
        feedback = ""
        last_errors: list[str] = []
        effort: str | None = None
        for i in range(attempts):
            is_last = i == attempts - 1
            content = base_user if not feedback else base_user + "\n\n" + feedback
            messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
            resp = self._request_once(role, messages, True, node_id, call_id, counter, effort)
            row = resp._log_row  # type: ignore[attr-defined]
            errors: list[str] = []
            obj: dict[str, Any] | None = None
            if resp.finish_reason == "length":
                errors.append("the reply was cut off by the output length limit; answer more briefly")
                effort = self._lower_effort(role, effort) or effort
            else:
                parsed = parse_json_object(resp.content)
                if parsed is None:
                    errors.append("the reply did not contain a parseable JSON object")
                else:
                    obj, errors = validate(parsed, schema)
                    if not errors and extra_check is not None:
                        obj, errors = extra_check(obj, is_last)
            if not errors and obj is not None:
                self._log({**row, "ok": True, "error": None})
                return obj
            last_errors = errors
            self._log({**row, "ok": False, "error": "; ".join(errors)[:300]})
            feedback = (
                "Your previous answer was rejected for these reasons:\n- "
                + "\n- ".join(errors)
                + "\nPlease answer again with a corrected JSON object."
            )
        raise StructuredOutputError(f"structured output failed for role {role}: {'; '.join(last_errors)}")


def make_gateway(
    llm_cfg: Mapping[str, Any],
    api_key: str,
    calls_log_path: str | Path | None,
    should_abort: Callable[[str], bool] | None = None,
    prompt_log_dir: Callable[[str], Path | None] | None = None,
) -> LLMGateway:
    transport = OpenAITransport(llm_cfg["base_url"], api_key, float(llm_cfg.get("request_timeout_seconds", 1200)))
    return LLMGateway(llm_cfg, calls_log_path, transport, should_abort=should_abort, prompt_log_dir=prompt_log_dir)


__all__ = [
    "ROLES",
    "LLMGateway",
    "OpenAITransport",
    "abort_decision",
    "call_cost",
    "TextResult",
    "TransportResponse",
    "TransientAPIError",
    "FatalAPIError",
    "make_gateway",
]
