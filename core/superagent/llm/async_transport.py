"""Cancellable DeepSeek chat transport for synchronous candidate workers."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import math
import threading
import time
from email.utils import parsedate_to_datetime

import httpx
import openai

from .gateway import NON_RETRYABLE_STATUS, FatalAPIError, TransientAPIError, TransportResponse, _token_count
from .ledger import CancelledCall


class _InvalidFrameError(ValueError):
    """Invalid or error SSE/JSON payload frame from transport."""


class _ModelIdleError(TimeoutError):
    """Streaming model emitted no meaningful content within model_idle_seconds."""


def _usage(raw, provider_id, model):
    if not isinstance(raw, dict):
        raise _InvalidFrameError("usage must be an object")
    details = raw.get("prompt_tokens_details") or {}
    completion = raw.get("completion_tokens_details") or {}
    if not isinstance(details, dict) or not isinstance(completion, dict):
        raise _InvalidFrameError("usage details must be objects")
    return {
        "provider_id": provider_id,
        "model": model,
        "input_tokens": _token_count(raw.get("prompt_tokens")),
        "cached_input_tokens": _token_count(raw.get("prompt_cache_hit_tokens", details.get("cached_tokens"))),
        "uncached_input_tokens": _token_count(raw.get("prompt_cache_miss_tokens")),
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
        self._db_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="deepseek-db")
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="deepseek-io", daemon=True)
        self.thread.start()
        self.slots = asyncio.Semaphore(slots)
        self.client = openai.AsyncOpenAI(
            base_url=base_url, api_key=key or "unused-no-credentials", max_retries=0,
            timeout=httpx.Timeout(limits["read_idle_seconds"], connect=limits["connect_seconds"], pool=limits["pool_seconds"]),
        )

    def close(self):
        async def _shutdown():
            current = asyncio.current_task()
            pending = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            await self.client.close()

        try:
            asyncio.run_coroutine_threadsafe(_shutdown(), self.loop).result(timeout=5)
        except Exception:
            pass
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)
        if not self.loop.is_closed():
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
        while True:
            try:
                return future.result(timeout=0.05)
            except concurrent.futures.TimeoutError:
                if cancel():
                    future.cancel()
                    if not closed.wait(2):
                        raise RuntimeError("DeepSeek stream did not close after cancellation")
                    raise CancelledCall("cancelled")

    async def _attempt(self, request, begin, receive, finish):
        loop = asyncio.get_running_loop()
        start = time.monotonic()
        role = request["role"]
        attempt_box: list[str] = []
        reason = None
        phase = "pool_timeout"
        status = "transport_error"
        timing: dict[str, float] = {}
        cap = self.limits[
            "plan_review_attempt_seconds" if role in ("reviewer", "planner.structured", "planner.free")
            else "code_attempt_seconds" if "coder" in role or role.startswith("draft_")
            else "short_attempt_seconds"
        ]

        async def _begin_locked():
            aid = await loop.run_in_executor(self._db_executor, begin)
            attempt_box.append(aid)
            return aid

        async def _receive(aid, usage_payload):
            await loop.run_in_executor(self._db_executor, receive, aid, usage_payload)

        try:
            async with asyncio.timeout(cap):
                await asyncio.wait_for(self.slots.acquire(), timeout=self.limits["pool_seconds"])
                try:
                    attempt = await asyncio.shield(_begin_locked())
                    phase = "absolute_attempt_timeout"
                    timing["pool_seconds"] = time.monotonic() - start
                    reply = await self._chat(
                        request,
                        lambda usage: loop.run_in_executor(self._db_executor, receive, attempt, usage),
                        timing,
                        cap,
                    )
                    reason = reply.finish_reason
                    status = ("received" if reply.transport_complete else
                              "model_idle_timeout" if reason == "model_idle_timeout" else "no_transport_terminator")
                    return reply
                finally:
                    self.slots.release()
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        except TimeoutError as exc:
            status = phase
            raise TransientAPIError(phase) from exc
        except openai.APIStatusError as exc:
            if exc.status_code in NON_RETRYABLE_STATUS:
                status = "http_rejected"
                error = FatalAPIError
            elif exc.status_code == 429:
                status = "rate_limited"
                error = TransientAPIError
            elif exc.status_code == 503:
                status = "overloaded"
                error = TransientAPIError
            else:
                status = "http_error"
                error = TransientAPIError
            failure = error(f"HTTP {exc.status_code}")
            failure.status = exc.status_code
            hint = exc.response.headers.get("retry-after")
            if not hint:
                try:
                    body = exc.response.json()
                    if isinstance(body, dict):
                        hint = body.get("retry_after")
                except Exception:
                    pass
            failure.retry_after = retry_after(hint)
            raise failure from exc
        except (openai.APITimeoutError, openai.APIConnectionError, httpx.TransportError) as exc:
            category = exc.__cause__ if isinstance(exc, openai.APIConnectionError) else exc
            reason = {
                httpx.ReadTimeout: "read_idle_timeout", httpx.ConnectTimeout: "connect_timeout",
                httpx.PoolTimeout: "pool_timeout",
            }.get(type(category), "network_error")
            is_unsent = isinstance(category, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)) or (
                isinstance(exc, openai.APIConnectionError) and category is None
            )
            status = "not_sent" if is_unsent else reason
            failure = TransientAPIError(reason)
            failure.not_sent = is_unsent
            raise failure from exc
        except openai.APIError as exc:
            status = "api_error"
            raise TransientAPIError("api_error") from exc
        except _InvalidFrameError as exc:
            status = "invalid_transport_frame"
            raise TransientAPIError(status) from exc
        finally:
            if attempt_box:
                aid = attempt_box[0]
                fut = loop.run_in_executor(
                    self._db_executor,
                    lambda: finish(aid, status, reason, time.monotonic() - start, **timing),
                )
                try:
                    await asyncio.shield(fut)
                except asyncio.CancelledError:
                    pass

    async def _chat(self, request, receive, timing, cap):
        start = time.monotonic()
        options = {
            "model": request["model"], "messages": request["messages"], "max_tokens": request["max_tokens"],
            "extra_body": {"thinking": {"type": "enabled" if request["thinking"] else "disabled"}},
        }
        if request["thinking"]:
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
            non_stream_timeout = httpx.Timeout(
                cap, connect=self.limits["connect_seconds"], pool=self.limits["pool_seconds"],
            )
            reply = await self.client.chat.completions.create(**options, timeout=non_stream_timeout)
            provider_id, model = reply.id, reply.model
            if reply.usage is not None:
                usage = _usage(reply.usage.model_dump(), provider_id, model)
                await receive(usage)
            if reply.choices:
                reason = reply.choices[0].finish_reason
                text.append(reply.choices[0].message.content or "")
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
                            except json.JSONDecodeError as exc:
                                raise _InvalidFrameError("invalid JSON SSE frame") from exc
                            if not isinstance(chunk, dict) or chunk.get("error"):
                                raise _InvalidFrameError("invalid SSE frame")
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
                                if content is not None and not isinstance(content, str):
                                    raise _InvalidFrameError("invalid content delta")
                                if reasoning:
                                    timing.setdefault("first_reasoning_seconds", now)
                                if content:
                                    timing.setdefault("first_content_seconds", now)
                                    text.append(content)
                                if reasoning or (content and content.strip()) or choice.get("finish_reason"):
                                    timing["last_model_seconds"] = now
                                    model_timer.reschedule(asyncio.get_running_loop().time() + idle)
                                reason = choice.get("finish_reason") or reason
            except TimeoutError:
                return TransportResponse(
                    "".join(text), "model_idle_timeout",
                    usage.get("input_tokens"), usage.get("output_tokens"),
                    usage.get("reasoning_tokens"), usage.get("cached_input_tokens"),
                    provider_id, model,
                    timing.get("first_data_seconds"), timing.get("first_content_seconds"),
                    timing.get("first_reasoning_seconds"),
                    transport_complete=False,
                    uncached_tokens=usage.get("uncached_input_tokens"),
                )
        return TransportResponse(
            "".join(text), reason, usage.get("input_tokens"), usage.get("output_tokens"),
            usage.get("reasoning_tokens"), usage.get("cached_input_tokens"), provider_id, model,
            timing.get("first_data_seconds"), timing.get("first_content_seconds"), timing.get("first_reasoning_seconds"),
            transport_complete=bool(done and reason),
            uncached_tokens=usage.get("uncached_input_tokens"),
        )
