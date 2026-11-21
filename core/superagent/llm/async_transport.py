"""Cancellable DeepSeek chat transport for synchronous candidate workers."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import math
import threading
import time
from email.utils import parsedate_to_datetime

import httpx
import openai

from .gateway import NON_RETRYABLE_STATUS, FatalAPIError, TransientAPIError, TransportResponse, _token_count
from .ledger import CancelledCall

log = logging.getLogger("superagent")

NO_BILLING_HTTP_STATUS = frozenset({400, 401, 402, 403, 404, 422, 429, 503})


class _InvalidFrameError(ValueError):
    """Malformed SSE or completion payload from the remote transport."""


class _ModelIdleError(Exception):
    """Stream went idle before producing any model output or usage."""


def _usage(raw, provider_id, model):
    if not isinstance(raw, dict):
        raise _InvalidFrameError("usage must be an object")
    details = raw.get("prompt_tokens_details") or {}
    completion = raw.get("completion_tokens_details") or {}
    if not isinstance(details, dict) or not isinstance(completion, dict):
        raise _InvalidFrameError("usage details must be objects")
    prompt = _token_count(raw.get("prompt_tokens"))
    cached = _token_count(raw.get("prompt_cache_hit_tokens"))
    if cached is None:
        cached = _token_count(details.get("cached_tokens"))
    if cached is not None and prompt is not None and cached > prompt:
        cached = None
    uncached = _token_count(raw.get("prompt_cache_miss_tokens"))
    if uncached is not None and prompt is not None and uncached > prompt:
        uncached = None
    return {
        "provider_id": provider_id,
        "model": model,
        "input_tokens": prompt,
        "cached_input_tokens": cached,
        "uncached_input_tokens": uncached,
        "output_tokens": _token_count(raw.get("completion_tokens")),
        "reasoning_tokens": _token_count(completion.get("reasoning_tokens")),
    }


def retry_after(value):
    try:
        delay = float(value)
        return max(0.0, delay) if math.isfinite(delay) else 0.0
    except (ValueError, TypeError):
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
        except (ValueError, TypeError, OverflowError):
            return 0.0


class AsyncDeepSeekTransport:
    def __init__(self, base_url, key, limits, slots):
        self.key, self.limits = key, limits
        self._closed = False
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="deepseek-io", daemon=True)
        self.thread.start()
        self._db_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="deepseek-ledger"
        )
        self.slots = asyncio.Semaphore(slots)
        self.client = openai.AsyncOpenAI(
            base_url=base_url, api_key=key or "unused-no-credentials", max_retries=0,
            timeout=httpx.Timeout(limits["read_idle_seconds"], connect=limits["connect_seconds"], pool=limits["pool_seconds"]),
        )

    def close(self):
        if self._closed:
            return
        self._closed = True

        async def _shutdown():
            current = asyncio.current_task()
            pending = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            await self.client.close()

        try:
            if not self.loop.is_closed():
                asyncio.run_coroutine_threadsafe(_shutdown(), self.loop).result(timeout=5)
        except Exception as exc:  # noqa: BLE001
            log.warning("DeepSeek async transport close incomplete: %s", exc)
        try:
            if not self.loop.is_closed():
                self.loop.call_soon_threadsafe(self.loop.stop)
        except RuntimeError:
            pass
        self.thread.join(5)
        if not self.loop.is_running() and not self.loop.is_closed():
            self.loop.close()
        self._db_executor.shutdown(wait=True)

    def invoke(self, request, cancel, begin, receive, finish):
        if cancel():
            raise CancelledCall("cancelled")
        if not self.key:
            raise FatalAPIError("missing_api_credential")
        closed = threading.Event()

        async def work():
            try:
                return await self._attempt(request, begin, receive, finish)
            finally:
                closed.set()

        future = asyncio.run_coroutine_threadsafe(work(), self.loop)
        cancelled = False
        while True:
            try:
                return future.result(timeout=.05)
            except concurrent.futures.TimeoutError:
                if cancel() and not cancelled:
                    cancelled = True
                    future.cancel()
                    if not closed.wait(5):
                        log.warning("DeepSeek stream did not finish closing within 5s after cancellation")
                    raise CancelledCall("cancelled")

    async def _attempt(self, request, begin, receive, finish):
        start = time.monotonic()
        role = request["role"]
        attempt = None
        attempt_box: list[str] = []
        begin_fut = None
        reason = None
        phase = "pool_timeout"
        status = "transport_error"
        timing = {}
        loop = asyncio.get_running_loop()
        cap = self.limits[
            "plan_review_attempt_seconds" if role in ("reviewer", "planner.structured", "planner.free", "leak_checker")
            else "code_attempt_seconds" if "coder" in role or role.startswith("draft_")
            else "short_attempt_seconds"
        ]

        def _run_begin():
            aid = begin()
            attempt_box.append(aid)
            return aid

        async def _record_usage(usage_payload):
            await loop.run_in_executor(self._db_executor, receive, attempt, usage_payload)

        try:
            async with asyncio.timeout(cap):
                await asyncio.wait_for(self.slots.acquire(), timeout=self.limits["pool_seconds"])
                try:
                    # Run SQLite registration off the event loop while retaining attempt_id across cancellation.
                    begin_fut = loop.run_in_executor(self._db_executor, _run_begin)
                    attempt = await asyncio.shield(begin_fut)
                    phase = "absolute_attempt_timeout"
                    timing["pool_seconds"] = time.monotonic() - start
                    reply = await self._chat(request, _record_usage, timing, cap)
                    reason = reply.finish_reason
                    status = ("received" if reply.transport_complete else
                              "model_idle_timeout" if reason == "model_idle_timeout" else "no_transport_terminator")
                    return reply
                finally:
                    self.slots.release()
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        except _ModelIdleError as exc:
            status = "model_idle_timeout"
            reason = "model_idle_timeout"
            raise TransientAPIError("model_idle_timeout") from exc
        except TimeoutError as exc:
            status = phase
            raise TransientAPIError(phase) from exc
        except openai.APIStatusError as exc:
            timing["http_status"] = exc.status_code
            if exc.status_code == 429:
                status = "rate_limited"
            elif exc.status_code == 503:
                status = "overloaded"
            elif exc.status_code in NO_BILLING_HTTP_STATUS:
                status = "http_rejected"
            else:
                status = "http_error"
            error = FatalAPIError if exc.status_code in NON_RETRYABLE_STATUS else TransientAPIError
            msg = getattr(exc, "message", None) or str(exc)
            failure = error(f"HTTP {exc.status_code}: {str(msg)[:300]}")
            failure.status = exc.status_code
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
            failure.retry_after = retry_after(hint)
            raise failure from exc
        except (openai.APITimeoutError, openai.APIConnectionError, httpx.TransportError) as exc:
            category = exc.__cause__ if isinstance(exc, openai.APIConnectionError) else exc
            reason = {
                httpx.ReadTimeout: "read_idle_timeout", httpx.ConnectTimeout: "connect_timeout",
                httpx.PoolTimeout: "pool_timeout",
            }.get(type(category), "network_error")
            status = "not_sent" if isinstance(category, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)) else reason
            raise TransientAPIError(reason) from exc
        except openai.APIError as exc:
            status = "api_error"
            raise TransientAPIError(f"{type(exc).__name__}: {str(exc)[:200]}") from exc
        except _InvalidFrameError as exc:
            status = "invalid_transport_frame"
            raise TransientAPIError(f"invalid_transport_frame: {str(exc)[:200]}") from exc
        finally:
            if attempt is None and begin_fut is not None:
                try:
                    await begin_fut
                except BaseException:  # noqa: BLE001
                    pass
                if attempt_box:
                    attempt = attempt_box[0]
            if attempt:
                try:
                    await loop.run_in_executor(
                        self._db_executor,
                        lambda: finish(attempt, status, reason, time.monotonic() - start, **timing),
                    )
                except Exception:  # noqa: BLE001
                    log.exception("failed to record finish_request for attempt %s", attempt)

    async def _chat(self, request, receive, timing, cap):
        start = time.monotonic()
        options = {
            "model": request["model"], "messages": request["messages"], "max_tokens": request["max_tokens"],
            "extra_body": {"thinking": {"type": "enabled" if request["thinking"] else "disabled"}},
        }
        if request["thinking"]:
            if request.get("reasoning_effort"):
                options["extra_body"]["reasoning_effort"] = request["reasoning_effort"]
        else:
            options["temperature"] = request["temperature"]
        if request["json_mode"]:
            options["response_format"] = {"type": "json_object"}
        usage = {}
        provider_id = model = reason = None
        text = []
        done = False
        if not request["stream"]:
            req_timeout = httpx.Timeout(
                cap, connect=self.limits["connect_seconds"], pool=self.limits["pool_seconds"]
            )
            reply = await self.client.chat.completions.create(**options, timeout=req_timeout)
            provider_id, model = reply.id, reply.model
            if reply.usage is not None:
                usage = _usage(reply.usage.model_dump(), provider_id, model)
                await receive(usage)
            if reply.choices:
                reason = reply.choices[0].finish_reason
                fragment = getattr(getattr(reply.choices[0], "message", None), "content", None)
                if fragment is not None and not isinstance(fragment, str):
                    raise _InvalidFrameError("invalid chat completion content")
                text.append(fragment or "")
            done = True
        else:
            idle = self.limits["model_idle_seconds"]
            try:
                async with asyncio.timeout(idle) as model_timer:
                    async with self.client.chat.completions.with_streaming_response.create(
                        **options, stream=True, stream_options={"include_usage": True},
                    ) as response:
                        timing["first_headers_seconds"] = time.monotonic() - start
                        async for line in response.iter_lines():
                            now = time.monotonic() - start
                            timing.setdefault("first_byte_seconds", now)
                            if not line.startswith("data:"):
                                continue
                            value = line[5:].strip()
                            if value == "[DONE]":
                                done = True
                                break
                            timing.setdefault("first_data_seconds", now)
                            try:
                                chunk = json.loads(value)
                            except ValueError as exc:
                                raise _InvalidFrameError(f"malformed SSE JSON: {exc}") from exc
                            if not isinstance(chunk, dict) or chunk.get("error"):
                                err_info = chunk.get("error") if isinstance(chunk, dict) else "non-object frame"
                                raise _InvalidFrameError(f"error SSE frame: {err_info}")
                            provider_id = chunk.get("id") or provider_id
                            model = chunk.get("model") or model
                            if chunk.get("usage") is not None:
                                usage = _usage(chunk["usage"], provider_id, model)
                                await receive(usage)
                            choices = chunk.get("choices", [])
                            if not isinstance(choices, list):
                                raise _InvalidFrameError("choices must be an array")
                            for choice in choices:
                                if not isinstance(choice, dict) or not isinstance(choice.get("delta", {}), dict):
                                    raise _InvalidFrameError("invalid choice delta")
                                delta = choice.get("delta", {})
                                reasoning = delta.get("reasoning_content")
                                content = delta.get("content")
                                if reasoning is not None and not isinstance(reasoning, str):
                                    raise _InvalidFrameError("invalid reasoning delta")
                                if content is not None and not isinstance(content, str):
                                    raise _InvalidFrameError("invalid content delta")
                                if reasoning:
                                    timing.setdefault("first_reasoning_seconds", now)
                                if content:
                                    timing.setdefault("first_content_seconds", now)
                                    text.append(content)
                                if (reasoning and reasoning.strip()) or (content and content.strip()) or choice.get("finish_reason"):
                                    timing["last_model_seconds"] = now
                                    model_timer.reschedule(asyncio.get_running_loop().time() + idle)
                                reason = choice.get("finish_reason") or reason
            except TimeoutError as exc:
                if not text and not usage:
                    raise _ModelIdleError("model_idle_timeout") from exc
                return TransportResponse(
                    "".join(text), "model_idle_timeout", usage.get("input_tokens"), usage.get("output_tokens"),
                    usage.get("reasoning_tokens"), usage.get("cached_input_tokens"), provider_id, model,
                    timing.get("first_data_seconds"), timing.get("first_content_seconds"),
                    timing.get("first_reasoning_seconds"), transport_complete=False,
                    uncached_tokens=usage.get("uncached_input_tokens"),
                )
        return TransportResponse(
            "".join(text), reason, usage.get("input_tokens"), usage.get("output_tokens"),
            usage.get("reasoning_tokens"), usage.get("cached_input_tokens"), provider_id, model,
            timing.get("first_data_seconds"), timing.get("first_content_seconds"), timing.get("first_reasoning_seconds"),
            transport_complete=bool(done and reason),
            uncached_tokens=usage.get("uncached_input_tokens"),
        )
