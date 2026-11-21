"""The single exit point for all LLM calls.

Responsibilities: choose the parameter group for a role, send the request to the
endpoint, retry API errors, validate structured answers, and append one
line per request to ``logs/llm_calls.jsonl``.
"""

from __future__ import annotations

import json
import math
import random
import threading
from dataclasses import asdict
from .ledger import CancelledCall, Ledger, DeferredCall, PermanentCall, fingerprint
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
# A confirmed reasoning-only exhaustion can recover with a lower effort (down to disabling thinking).
LOWER_EFFORT = {"max": "high", "high": "low", "low": "off"}
NON_RETRYABLE_STATUS = frozenset({400, 401, 402, 403, 404, 422})
NO_BILLING_HTTP_STATUS = frozenset({400, 401, 402, 403, 404, 422, 429, 503})
DEFAULT_MAX_CONTEXT_TOKENS = 1_000_000
# Roles that assess an already finished run; they may keep working into the finalize reserve.
ASSESSMENT_ROLES = frozenset({"result_reader", "leak_checker", "format_fixer"})


@dataclass
class TransportResponse:
    content: str
    finish_reason: str | None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None
    provider_id: str | None = None
    returned_model: str | None = None
    ttft_s: float | None = None
    first_content_s: float | None = None
    first_reasoning_s: float | None = None
    transport_complete: bool = True
    uncached_tokens: int | None = None


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
    finish_reason: str | None = None
    body_limited: bool = False


def _token_count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _usage_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    return dict(value) if hasattr(value, "model_dump") else {}


class _SSEErrorFrame(ValueError):
    """Explicit error payload received inside an HTTP 200 SSE stream."""


def _backoff_delay(base_delay: float, streak: int, retry_after: float = 0.0, cap: float = 60.0) -> float:
    if retry_after > 0 and math.isfinite(retry_after):
        return float(retry_after)
    if base_delay <= 0 or not math.isfinite(base_delay):
        return 0.0
    exp = min(cap, base_delay * (2 ** min(max(0, streak), 6)))
    return exp * random.uniform(0.75, 1.25)


def _is_reasoning_exhausted(resp: TransportResponse, thinking: bool) -> bool:
    """True when thinking consumed the output budget with no visible content."""
    return (
        not resp.content.strip()
        and resp.finish_reason == "length"
        and thinking
        and (
            resp.reasoning_tokens is None
            or resp.completion_tokens is None
            or resp.reasoning_tokens > 0
        )
    )


class OpenAITransport:
    """Sends chat completion requests to an OpenAI-compatible endpoint."""

    def __init__(self, base_url: str, api_key: str, timeout: float) -> None:
        import httpx
        from openai import OpenAI

        self._base_url = str(base_url)
        timeout_cfg = httpx.Timeout(
            timeout,
            connect=min(float(timeout), 10.0),
            pool=min(float(timeout), 30.0),
        )
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_cfg, max_retries=0)

    def close(self) -> None:
        self._client.close()

    def __call__(self, request: Mapping[str, Any]) -> TransportResponse:
        import openai
        import httpx

        kwargs: dict[str, Any] = {
            "model": request["model"],
            "messages": request["messages"],
            "max_tokens": request["max_tokens"],
        }
        if not request["thinking"]:
            kwargs["temperature"] = request["temperature"]
        is_deepseek = "deepseek" in self._base_url.lower() or "deepseek" in str(request["model"]).lower()
        extra: dict[str, Any] = {}
        if request["thinking"] or is_deepseek:
            extra["thinking"] = {"type": "enabled" if request["thinking"] else "disabled"}
        if request["thinking"] and request.get("reasoning_effort"):
            extra["reasoning_effort"] = request["reasoning_effort"]
        if extra:
            kwargs["extra_body"] = extra
        if request.get("json_mode"):
            kwargs["response_format"] = {"type": "json_object"}
        stream = request.get("stream", False)
        text, reason, usage, rid, model, first = [], None, {}, None, None, None
        start = time.monotonic()
        first_content = None
        first_reasoning = None
        on_stream_open = request.get("on_stream_open")
        try:
            if stream:
                kwargs["stream"] = True
                if request.get("stream_include_usage", True):
                    kwargs["stream_options"] = {"include_usage": True}
                done = False
                with self._client.chat.completions.with_streaming_response.create(**kwargs) as response:
                    if on_stream_open is not None:
                        on_stream_open(response.close)
                    try:
                        for line in response.iter_lines():
                            if request.get("cancel") and request["cancel"].is_set():
                                break
                            if not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                done = True
                                break
                            chunk = json.loads(data)
                            if not isinstance(chunk, dict):
                                raise ValueError("malformed SSE frame")
                            if chunk.get("error"):
                                raise _SSEErrorFrame(f"error SSE frame: {chunk['error']}")
                            if first is None:
                                first = time.monotonic() - start
                            rid = chunk.get("id") or rid
                            model = chunk.get("model") or model
                            new_usage = chunk.get("usage")
                            choices = chunk.get("choices", [])
                            if new_usage is not None and not isinstance(new_usage, dict) or not isinstance(choices, list):
                                raise ValueError("invalid SSE usage or choices")
                            usage = new_usage or usage
                            for ch in choices:
                                if not isinstance(ch, dict) or not isinstance(ch.get("delta", {}), dict):
                                    raise ValueError("invalid SSE choice")
                                delta = ch.get("delta", {})
                                if first_reasoning is None and delta.get("reasoning_content"):
                                    first_reasoning = time.monotonic() - start
                                fragment = delta.get("content")
                                if fragment is not None and not isinstance(fragment, str):
                                    raise ValueError("invalid SSE content")
                                if fragment and first_content is None:
                                    first_content = time.monotonic() - start
                                if fragment:
                                    text.append(fragment)
                                reason = ch.get("finish_reason") or reason
                    finally:
                        if on_stream_open is not None:
                            on_stream_open(None)
                if not done or not reason:
                    reason = "interrupted"
            else:
                from openai.types.chat import ChatCompletion

                resp = self._client.chat.completions.create(**kwargs)
                if not isinstance(resp, ChatCompletion) or not isinstance(resp.choices, list):
                    raise ValueError("invalid chat completion")
                rid, model = resp.id, resp.model
                usage = _usage_dict(resp.usage)
                if resp.choices:
                    ch = resp.choices[0]
                    fragment = getattr(getattr(ch, "message", None), "content", None)
                    if fragment is not None and not isinstance(fragment, str):
                        raise ValueError("invalid chat completion content")
                    text.append(fragment or "")
                    reason = getattr(ch, "finish_reason", None)
        except openai.APIStatusError as exc:
            error = FatalAPIError if exc.status_code in NON_RETRYABLE_STATUS else TransientAPIError
            err = error(f"HTTP {exc.status_code}: {exc.message}")
            err.status = exc.status_code
            hint = exc.response.headers.get("retry-after")
            if not hint and isinstance(exc.body, dict):
                hint = exc.body.get("retry_after")
            if not hint:
                try:
                    raw_body = exc.response.json()
                    if isinstance(raw_body, dict):
                        hint = raw_body.get("retry_after") or (
                            raw_body.get("error").get("retry_after")
                            if isinstance(raw_body.get("error"), dict)
                            else None
                        )
                except Exception:  # noqa: BLE001
                    hint = None
            try:
                delay = float(hint or 0)
                err.retry_after = max(0.0, delay) if math.isfinite(delay) else 0.0
            except (ValueError, TypeError):
                from email.utils import parsedate_to_datetime
                try:
                    err.retry_after = max(0.0, parsedate_to_datetime(hint).timestamp() - time.time())
                except (ValueError, TypeError, OverflowError):
                    err.retry_after = 0.0
            raise err from exc
        except (openai.APIError, httpx.TransportError, ValueError, OSError) as exc:
            if isinstance(exc, _SSEErrorFrame) or (not text and not usage):
                err = TransientAPIError(f"{type(exc).__name__}: {str(exc)[:200]}")
                cause = exc.__cause__ if isinstance(exc, openai.APIConnectionError) else exc
                if isinstance(cause, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)):
                    err.not_sent = True
                raise err from exc
            reason = "interrupted"
        prompt = _token_count(usage.get("prompt_tokens"))
        completion = _token_count(usage.get("completion_tokens"))
        reasoning = _token_count(_usage_dict(usage.get("completion_tokens_details")).get("reasoning_tokens"))
        cached = _token_count(usage.get("prompt_cache_hit_tokens"))
        if cached is None:
            cached = _token_count(_usage_dict(usage.get("prompt_tokens_details")).get("cached_tokens"))
        if cached is not None and prompt is not None and cached > prompt:
            cached = None
        return TransportResponse("".join(text), reason, prompt, completion, reasoning, cached, rid, model, first, first_content,
                                 first_reasoning)


def call_cost(group: Mapping[str, Any], prompt_tokens: int | None, completion_tokens: int | None) -> float | None:
    """Cost of one request from the per-million-token prices of its parameter group. None without usage data."""
    if prompt_tokens is None or completion_tokens is None:
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
        *,
        run_id: str | None = None,
    ) -> None:
        self.cfg = llm_cfg
        self.calls_log_path = Path(calls_log_path) if calls_log_path else None
        self.transport = transport
        self.should_abort = should_abort or (lambda role: False)
        self.prompt_log_dir = prompt_log_dir
        self._sleep = sleep
        self.ledger = (
            Ledger(
                self.calls_log_path.with_name("operations.sqlite"),
                int(llm_cfg.get("operation_extra_sends", 2)),
                run_id=run_id,
                rate_limit_retries=int(llm_cfg.get("operation_rate_limit_retries", 8)),
            )
            if self.calls_log_path
            else None
        )
        self.formal = run_id is not None
        self.cooldown_until = 0.0
        self._cooldown_streak = 0
        self.endpoint_failure = None
        self._lock = threading.Lock()
        self.totals: dict[str, float] = {"calls": 0, "failed_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}

    # ------------------------------------------------------------------ helpers
    def _group(self, role: str) -> Mapping[str, Any]:
        group = ROLES.get(role, ("generation", "text"))[0]
        policy=self.cfg.get("role_policy",{}).get(role,{})
        return {**self.cfg[group],**policy}

    def _cost(self, group: Mapping[str, Any], pt: int | None, ct: int | None) -> float | None:
        return call_cost(group, pt, ct)

    def _receipt_cost(self, group, resp):
        cost = self._cost(group,resp.prompt_tokens,resp.completion_tokens)
        if cost is not None and resp.cached_tokens is not None:
            cost -= resp.cached_tokens*(group["input_price_per_million"]-group.get("cached_input_price_per_million",group["input_price_per_million"]))/1e6
        return cost

    def _record_cooldown(self, status: int, retry_after: float = 0.0) -> tuple[float, float]:
        with self._lock:
            self._cooldown_streak += 1
            streak = self._cooldown_streak
            if retry_after > 0 and math.isfinite(retry_after):
                cool = float(retry_after)
            elif status == 524:
                cool = 120.0
            else:
                base = float(self.cfg.get("api_retry_delay_seconds", 3))
                cool = _backoff_delay(base if base > 0 else 3.0, streak - 1, cap=60.0)
            self.cooldown_until = max(self.cooldown_until, time.monotonic() + cool)
            return self.cooldown_until, cool

    def _reset_cooldown_streak(self) -> None:
        with self._lock:
            self._cooldown_streak = 0

    def _get_route_state(self) -> tuple[str | None, float]:
        with self._lock:
            return self.endpoint_failure, self.cooldown_until

    def _set_endpoint_failure(self, message: str) -> None:
        with self._lock:
            self.endpoint_failure = message

    def _log(self, row: dict[str, Any]) -> None:
        with self._lock:
            self.totals["calls"] += 1
            if not self.formal and row.get("cost") is None:
                self.totals["unknown_cost_attempts"]=self.totals.get("unknown_cost_attempts",0)+1
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

    def _call_transport(self, request: Mapping[str, Any], on_late=None) -> TransportResponse:
        """Run the transport in a helper thread so a hard deadline can abandon it."""
        result: dict[str, Any] = {}
        done = threading.Event()
        guard = threading.Lock()
        cancel = threading.Event()
        closer: list[Callable[[], None] | None] = [None]

        def on_stream_open(close_fn: Callable[[], None] | None) -> None:
            should_close = False
            with guard:
                closer[0] = close_fn
                should_close = cancel.is_set() and close_fn is not None
            if should_close and close_fn is not None:
                try:
                    close_fn()
                except Exception:  # noqa: BLE001
                    pass

        def work() -> None:
            try:
                result["value"] = self.transport(request)
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                result["error"] = exc
            finally:
                with guard:
                    closer[0] = None
                    late = cancel.is_set()
                    done.set()
                if late and on_late is not None and "value" in result:
                    on_late(result["value"])

        request = {**request, "cancel": cancel, "on_stream_open": on_stream_open}
        t = threading.Thread(target=work, daemon=True)
        t.start()
        while not done.wait(ABORT_POLL_SECONDS):
            if self.should_abort(str(request.get("role", ""))):
                close_fn = None
                with guard:
                    if done.is_set():
                        break
                    cancel.set()
                    close_fn = closer[0]
                if close_fn is not None:
                    try:
                        close_fn()
                    except Exception:  # noqa: BLE001
                        pass
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
        expanded: bool = False,
    ) -> TransportResponse:
        """Send one logical request, retrying API errors. Logs one line per attempt."""
        group = self._group(role)
        thinking = effort != "off" if effort is not None else group.get("thinking") == "enabled"
        effort = effort or group.get("reasoning_effort")
        request = {
            "role": role,
            "model": group["model"],
            "messages": messages,
            "max_tokens": self.output_limit(role, expanded),
            "temperature": float(group["temperature"]),
            "thinking": thinking,
            "reasoning_effort": effort if thinking else None,
            "json_mode": json_mode,
        }
        request["stream"] = bool(self.cfg.get("stream",False))
        if self.formal:
            return self._formal_request_once(request, node_id, call_id, attempt_counter)
        max_ctx = int(group.get("max_context_tokens", self.cfg.get("max_context_tokens", DEFAULT_MAX_CONTEXT_TOKENS)))
        prompt_bytes = sum(len(m["content"].encode()) for m in request["messages"])
        if max_ctx > 0 and prompt_bytes + request["max_tokens"] + 256 > max_ctx:
            raise PermanentCall("required_context_too_large")
        max_retries = max(0, int(self.cfg.get("api_max_retries", 1)))
        delay = float(self.cfg.get("api_retry_delay_seconds", 3))
        api_failures = 0
        while True:
            if self.should_abort(role):
                raise LLMUnavailableError("deadline reached")
            failure, cooldown = self._get_route_state()
            if failure:
                raise LLMUnavailableError(failure)
            if time.monotonic() < cooldown:
                raise DeferredCall("endpoint cooling down", cooldown)
            sig = fingerprint(request)
            # UTF-8 bytes give a conservative input allowance for these text prompts.
            reserve = call_cost(group, sum(len(m["content"].encode())+32 for m in request["messages"]),request["max_tokens"])
            attempt_id=self.ledger.begin(node_id,role,call_id,sig,reserve) if self.ledger else new_id()
            attempt_counter[0] += 1
            attempt = attempt_counter[0]
            self._log_prompt(node_id, role, attempt, request["messages"])
            t0 = time.monotonic()
            base_row = {
                "ts": now_iso(), "attempt_id":attempt_id, "operation_id":node_id, "fingerprint":sig,
                "endpoint":self.cfg.get("base_url"),"currency":self.cfg.get("currency"),"max_tokens":request["max_tokens"],"stream":request["stream"],
                "json_mode": json_mode, "temperature": request["temperature"],
                "prompt_chars": sum(len(m["content"]) for m in messages),
                "call_id": call_id,
                "attempt": attempt,
                "agent": role,
                "node_id": node_id,
                "model": group["model"],
                "thinking": thinking,
                "reasoning_effort": effort if thinking else None,
            }
            abandoned_logged = threading.Event()
            try:
                def reconcile(late):
                    if not self.ledger or late.prompt_tokens is None or late.completion_tokens is None:
                        return
                    abandoned_logged.wait()
                    charge = self._receipt_cost(group,late)
                    row = dict(attempt_id=attempt_id, prompt_tokens=late.prompt_tokens,completion_tokens=late.completion_tokens,
                               reasoning_tokens=late.reasoning_tokens, cached_tokens=late.cached_tokens,
                               cost=charge,provider_id=late.provider_id)
                    self.ledger.finish(attempt_id,charge,row,"late_usage")
                    with self._lock:
                        self.totals["unknown_cost_attempts"] -= 1
                        self.totals["prompt_tokens"] += late.prompt_tokens
                        self.totals["completion_tokens"] += late.completion_tokens
                        self.totals["cost"] += charge
                    append_jsonl(self.calls_log_path.with_name("late_usage.jsonl"),row)
                resp = self._call_transport(request,on_late=reconcile)
            except (TransientAPIError, FatalAPIError) as exc:
                status = getattr(exc, "status", None)
                not_sent = bool(getattr(exc, "not_sent", False))
                if not_sent:
                    l_status, l_cost = "not_sent", 0.0
                elif status == 429:
                    l_status, l_cost = "rate_limited", 0.0
                elif status == 503:
                    l_status, l_cost = "overloaded", 0.0
                elif status in NO_BILLING_HTTP_STATUS:
                    l_status, l_cost = "http_rejected", 0.0
                else:
                    l_status, l_cost = "transport_error", None
                if self.ledger:
                    self.ledger.finish(attempt_id, l_cost, None, l_status)
                self._log({**base_row, "prompt_tokens": None, "completion_tokens": None, "reasoning_tokens": None,
                           "latency_secs": round(time.monotonic() - t0, 3), "ok": False, "error": str(exc)[:300],
                           "cost": None, "http_status": status})
                if isinstance(exc, FatalAPIError):
                    if status in (401, 402, 403):
                        self._set_endpoint_failure(str(exc))
                    raise PermanentCall(str(exc)) from exc
                if status in (524, 429):
                    resume_at, cool = self._record_cooldown(status, getattr(exc, "retry_after", 0.0))
                    raise DeferredCall(f"HTTP {status}; cool down {cool:g}s", resume_at) from exc
                api_failures += 1
                if api_failures > max_retries:
                    if status == 503:
                        resume_at, cool = self._record_cooldown(503, getattr(exc, "retry_after", 0.0))
                        raise DeferredCall(f"HTTP 503; cool down {cool:g}s", resume_at) from exc
                    raise LLMUnavailableError(f"LLM unavailable after {max_retries} retries: {exc}") from exc
                wait_s = _backoff_delay(delay, api_failures - 1, getattr(exc, "retry_after", 0.0))
                waited = 0.0
                while waited < wait_s:
                    if self.should_abort(role):
                        raise LLMUnavailableError("deadline reached while waiting to retry") from exc
                    step = min(ABORT_POLL_SECONDS, wait_s - waited)
                    self._sleep(step)
                    waited += step
                continue
            except LLMUnavailableError as exc:
                try:
                    if self.ledger:
                        self.ledger.finish(attempt_id,None,None,"abandoned")
                    self._log({**base_row, "prompt_tokens": None, "completion_tokens": None, "reasoning_tokens": None,
                               "latency_secs": round(time.monotonic() - t0, 3), "ok": False, "error": str(exc)[:300],
                               "cost": None})
                finally:
                    abandoned_logged.set()
                raise
            except Exception as exc:
                if self.ledger:
                    self.ledger.finish(attempt_id,None,None,"transport_error")
                self._log({**base_row, "latency_secs": round(time.monotonic() - t0, 3),
                           "ok": False, "error": type(exc).__name__, "cost": None})
                raise
            self._reset_cooldown_streak()
            resp_row = {
                **base_row,
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
                "reasoning_tokens": resp.reasoning_tokens,
                "latency_secs": round(time.monotonic() - t0, 3),
            }
            hit=resp.cached_tokens
            resp_row["cost"] = self._receipt_cost(group,resp)
            resp_row.update(cached_tokens=hit,finish_reason=resp.finish_reason,provider_id=resp.provider_id,
                            returned_model=resp.returned_model,ttft_s=resp.ttft_s,
                            first_content_s=resp.first_content_s,first_reasoning_s=resp.first_reasoning_s)
            if self.ledger:
                self.ledger.finish(attempt_id,resp_row["cost"],{k:v for k,v in resp_row.items() if "token" in k},"received")
                if node_id:
                    self.ledger.put(node_id,role+".receipt",attempt_id,asdict(resp))
            resp._log_row = resp_row  # type: ignore[attr-defined]
            return resp

    def _formal_request_once(self, request, operation, call_id, counter):
        role = request["role"]
        group = self._group(role)
        ledger = self.ledger
        assert ledger is not None
        signature = fingerprint(request)
        prompt_bytes = sum(len(m["content"].encode()) for m in request["messages"])
        max_ctx = int(group.get("max_context_tokens", self.cfg.get("max_context_tokens", DEFAULT_MAX_CONTEXT_TOKENS)))
        # UTF-8 byte length is an upper bound for these text prompts.
        if max_ctx > 0 and prompt_bytes + request["max_tokens"] + 256 > max_ctx:
            raise PermanentCall("required_context_too_large")
        max_retries = max(0, int(self.cfg.get("api_max_retries", 1)))
        delay = float(self.cfg.get("api_retry_delay_seconds", 3))
        api_failures = 0
        while True:
            if self.should_abort(role):
                raise CancelledCall("cancelled")
            failure, cooldown = self._get_route_state()
            if failure:
                raise PermanentCall(failure)
            if time.monotonic() < cooldown:
                raise DeferredCall("endpoint_cooldown", cooldown)
            attempt = None
            attempt_num = 0
            t0 = time.monotonic()

            def begin():
                nonlocal attempt, attempt_num
                if self.should_abort(role):
                    raise CancelledCall("cancelled")
                ep_fail, ep_cool = self._get_route_state()
                if ep_fail:
                    raise PermanentCall(ep_fail)
                if time.monotonic() < ep_cool:
                    raise DeferredCall("endpoint_cooldown", ep_cool)
                attempt = ledger.begin_request(operation, role, call_id, signature, request)
                counter[0] += 1
                attempt_num = counter[0]
                self._log_prompt(operation, role, attempt_num, request["messages"])
                return attempt

            def make_row():
                return {
                    "ts": now_iso(), "attempt_id": attempt, "operation_id": operation, "fingerprint": signature,
                    "endpoint": self.cfg.get("base_url"), "currency": self.cfg.get("currency"),
                    "stream": request["stream"], "json_mode": request["json_mode"],
                    "temperature": request["temperature"],
                    "prompt_chars": sum(len(m["content"]) for m in request["messages"]),
                    "prompt_bytes": prompt_bytes, "call_id": call_id, "attempt": attempt_num or counter[0],
                    "agent": role, "node_id": operation, "model": request["model"],
                    "thinking": request["thinking"], "reasoning_effort": request["reasoning_effort"],
                    "max_tokens": request["max_tokens"],
                    "latency_secs": round(time.monotonic() - t0, 3),
                }

            try:
                resp = self.transport.invoke(request, lambda: self.should_abort(role), begin, ledger.receive_usage, ledger.finish_request)
            except FatalAPIError as exc:
                status = getattr(exc, "status", None)
                self._log({**make_row(), "prompt_tokens": None, "completion_tokens": None, "reasoning_tokens": None,
                           "ok": False, "error": str(exc)[:300], "cost": None, "http_status": status})
                if status in (401, 402, 403) or str(exc) == "missing_api_credential":
                    self._set_endpoint_failure(str(exc))
                raise PermanentCall(str(exc)) from exc
            except TransientAPIError as exc:
                status = getattr(exc, "status", None)
                self._log({**make_row(), "prompt_tokens": None, "completion_tokens": None, "reasoning_tokens": None,
                           "ok": False, "error": str(exc)[:300], "cost": None, "http_status": status})
                if status in (429, 524):
                    resume_at, _cool = self._record_cooldown(status, getattr(exc, "retry_after", 0.0))
                    raise DeferredCall(f"http_{status}", resume_at) from exc
                if api_failures < max_retries:
                    api_failures += 1
                    wait_s = _backoff_delay(delay, api_failures - 1, getattr(exc, "retry_after", 0.0))
                    deadline = time.monotonic() + wait_s
                    while time.monotonic() < deadline:
                        if self.should_abort(role):
                            raise CancelledCall("cancelled") from exc
                        self._sleep(min(.1, deadline - time.monotonic()))
                    continue
                if status == 503:
                    resume_at, _cool = self._record_cooldown(503, getattr(exc, "retry_after", 0.0))
                    raise DeferredCall("http_503", resume_at) from exc
                raise LLMUnavailableError(f"LLM unavailable after {max_retries} retries: {exc}") from exc
            except CancelledCall:
                if attempt is not None:
                    self._log({**make_row(), "prompt_tokens": None, "completion_tokens": None, "reasoning_tokens": None,
                               "ok": False, "error": "cancelled", "cost": None})
                raise
            self._reset_cooldown_streak()
            if not resp.transport_complete:
                resp.finish_reason = "model_idle_timeout" if resp.finish_reason == "model_idle_timeout" else "no_transport_terminator"
            if resp.finish_reason in ("content_filter", "tool_calls"):
                self._log({**make_row(), "prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens,
                           "reasoning_tokens": resp.reasoning_tokens, "cached_tokens": resp.cached_tokens,
                           "finish_reason": resp.finish_reason, "provider_id": resp.provider_id,
                           "returned_model": resp.returned_model, "ttft_s": resp.ttft_s,
                           "first_content_s": resp.first_content_s, "first_reasoning_s": resp.first_reasoning_s,
                           "ok": False, "error": resp.finish_reason, "cost": None})
                raise PermanentCall(resp.finish_reason)
            resp._log_row = {
                **make_row(),
                "prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens,
                "reasoning_tokens": resp.reasoning_tokens, "cached_tokens": resp.cached_tokens,
                "finish_reason": resp.finish_reason, "provider_id": resp.provider_id,
                "returned_model": resp.returned_model, "ttft_s": resp.ttft_s,
                "first_content_s": resp.first_content_s, "first_reasoning_s": resp.first_reasoning_s,
                "cost": None,
            }
            return resp

    # ------------------------------------------------------------------ public API
    def output_limit(self, role, expanded=False):
        group = self._group(role)
        initial = int(group["max_output_tokens"])
        return max(initial, int(group.get("recovery_max_output_tokens", initial))) if expanded else initial

    def _lower_effort(self, role: str, current: str | None) -> str | None:
        """The next lower reasoning effort for a thinking role, or None."""
        group = self._group(role)
        if not self.cfg.get("lower_effort_on_truncation", True):
            return None
        if current == "off":
            return None
        if current is None and group.get("thinking") != "enabled":
            return None
        return LOWER_EFFORT.get(str(current or group.get("reasoning_effort")))

    def text(self, role: str, system: str, user: str, node_id: str | None = None, *, expanded=False) -> TextResult:
        """Plain text call. A reply cut off by the length limit is returned with ``truncated``.

        An empty reply cut off by the limit (the reasoning used the whole output budget) is retried
        with progressively lower reasoning effort down to disabling thinking.
        """
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        ledger = self.ledger if node_id else None
        state_key = fingerprint([messages, self.output_limit(role, expanded)])
        stage = f"{role}.text_retry"
        saved = ledger.get(node_id, stage, state_key) if ledger else None
        call_id = saved["call_id"] if saved else new_id()
        counter = [ledger.sends(call_id) if saved and ledger else 0]
        effort = saved["effort"] if saved else None
        while True:
            resp = self._request_once(role, messages, False, node_id, call_id, counter, effort, expanded)
            truncated = resp.finish_reason not in ("stop",)
            row = resp._log_row  # type: ignore[attr-defined]
            ok = bool(resp.content.strip())
            lower = None
            if _is_reasoning_exhausted(resp, bool(row["thinking"])) and self.cfg.get("lower_effort_on_truncation", True):
                if role in ("patch_coder", "repair_coder") and effort is None:
                    lower = "off"
                else:
                    lower = self._lower_effort(role, effort)
            error = None if ok else ("empty_response; retrying with reasoning effort " + lower if lower else "empty_response")
            self._log({**row, "ok": ok and not truncated, "has_content": ok, "error": error or (resp.finish_reason if truncated else None)})
            if lower is None:
                if effort is not None and ledger:
                    ledger.put(node_id, stage, state_key, None)
                body_limited = resp.finish_reason == "length" and ok
                return TextResult(resp.content, truncated, resp.finish_reason, body_limited)
            effort = lower
            if ledger:
                ledger.put(node_id, stage, state_key, {"call_id": call_id, "effort": effort})

    def structured(
        self,
        role: str,
        system: str,
        user: str,
        schema: Schema,
        node_id: str | None = None,
        extra_check: Callable[[dict[str, Any], bool], tuple[dict[str, Any], list[str]]] | None = None,
        max_attempts: int | None = None,
        initial_effort: str | None = None,
        expanded: bool = False,
        resume_key: str | None = None,
    ) -> dict[str, Any]:
        """Structured call. ``extra_check(obj, is_last_attempt)`` may normalise and add errors."""
        attempts = int(max_attempts or self.cfg.get("json_max_attempts", 3))
        schema_description = describe(schema)
        base_user = user if "json" in user.lower() else user + "\n\nAnswer in JSON."
        base_user = base_user + "\n\n" + schema_description
        ledger = self.ledger if node_id else None
        state_key = fingerprint([resume_key or [system, user], schema_description, initial_effort, attempts,
                                 self.output_limit(role, expanded)])
        stage = f"{role}.json_retry"
        saved = ledger.get(node_id, stage, state_key) if ledger else None
        call_id = saved["call_id"] if saved else new_id()
        counter = [ledger.sends(call_id) if saved and ledger else 0]
        feedback = saved["feedback"] if saved else ""
        last_errors: list[str] = []
        effort = saved["effort"] if saved else initial_effort
        current_expanded = saved.get("expanded", expanded) if saved else expanded
        for i in range(saved["index"] if saved else 0, attempts):
            is_last = i == attempts - 1
            content = base_user if not feedback else base_user + "\n\n" + feedback
            messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
            resp = self._request_once(role, messages, True, node_id, call_id, counter, effort, current_expanded)
            row = resp._log_row  # type: ignore[attr-defined]
            errors: list[str] = []
            obj: dict[str, Any] | None = None
            if resp.finish_reason == "length":
                if _is_reasoning_exhausted(resp, bool(row["thinking"])):
                    errors.append("the response spent its output budget before writing JSON; focus on the requested decision and return the object")
                    lower = self._lower_effort(role, effort)
                    if lower is not None:
                        effort = lower
                    elif not current_expanded and self.output_limit(role, True) > self.output_limit(role, False):
                        current_expanded = True
                else:
                    errors.append("previous JSON was truncated; return a complete object with every required field")
                    if not current_expanded and self.output_limit(role, True) > self.output_limit(role, False):
                        current_expanded = True
            elif resp.finish_reason != "stop":
                errors.append(f"incomplete response ({resp.finish_reason}); return a complete JSON object")
            elif not resp.content.strip():
                errors.append("blank_stop: return a nonempty JSON object")
            else:
                if self.formal:
                    from .jsonparse import parse_complete_object

                    parsed = parse_complete_object(resp.content)
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
                if i and ledger:
                    ledger.put(node_id, stage, state_key, None)
                return obj
            last_errors = errors
            rejected = resp.content[:2400]
            self._log({**row, "ok": False, "error": "; ".join(errors)[:300]})
            prior = f"\nPrior response (preserve its findings):\n{rejected}" if rejected.strip() else ""
            feedback = (
                "Your previous answer was rejected for these reasons:\n- "
                + "\n- ".join(errors)
                + prior + "\nReturn the corrected JSON only."
            )
            if not is_last and ledger:
                ledger.put(node_id, stage, state_key, {"call_id": call_id, "index": i + 1,
                                                       "feedback": feedback, "effort": effort,
                                                       "expanded": current_expanded})
        if attempts > 1 and ledger:
            ledger.put(node_id, stage, state_key, None)
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
