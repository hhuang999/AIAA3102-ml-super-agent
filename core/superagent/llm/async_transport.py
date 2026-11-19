"""Cancellable DeepSeek chat transport for synchronous candidate workers."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import threading
import time
from email.utils import parsedate_to_datetime

import httpx
import openai

from .gateway import NON_RETRYABLE_STATUS, FatalAPIError, TransientAPIError, TransportResponse, _token_count
from .ledger import CancelledCall


def _usage(raw, provider_id, model):
    details = raw.get("prompt_tokens_details") or {}
    completion = raw.get("completion_tokens_details") or {}
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
        return max(0.0, float(value))
    except (ValueError, TypeError):
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
        except (ValueError, TypeError, OverflowError):
            return 0.0


class AsyncDeepSeekTransport:
    def __init__(self, base_url, key, limits, slots):
        self.key, self.limits = key, limits
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="deepseek-io", daemon=True)
        self.thread.start()
        self.slots = asyncio.Semaphore(slots)
        self.client = openai.AsyncOpenAI(
            base_url=base_url, api_key=key or "unused-no-credentials", max_retries=0,
            timeout=httpx.Timeout(limits["read_idle_seconds"], connect=limits["connect_seconds"], pool=limits["pool_seconds"]),
        )

    def close(self):
        asyncio.run_coroutine_threadsafe(self.client.close(), self.loop).result(timeout=5)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)
        self.loop.close()

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
                return future.result(timeout=.05)
            except concurrent.futures.TimeoutError:
                if cancel():
                    future.cancel()
                    if not closed.wait(2):
                        raise RuntimeError("DeepSeek stream did not close after cancellation")
                    raise CancelledCall("cancelled")

    async def _attempt(self, request, begin, receive, finish):
        start = time.monotonic()
        role = request["role"]
        attempt = None
        reason = None
        phase = "pool_timeout"
        status = "transport_error"
        timing = {}
        cap = self.limits[
            "plan_review_attempt_seconds" if role in ("reviewer", "planner.structured", "planner.free")
            else "code_attempt_seconds" if "coder" in role or role.startswith("draft_")
            else "short_attempt_seconds"
        ]
        try:
            async with asyncio.timeout(cap):
                await asyncio.wait_for(self.slots.acquire(), timeout=self.limits["pool_seconds"])
                try:
                    # No await between registration and retaining its attempt ID.
                    attempt = begin()
                    phase = "absolute_attempt_timeout"
                    timing["pool_seconds"] = time.monotonic() - start
                    reply = await self._chat(request, lambda usage: receive(attempt, usage), timing)
                    reason = reply.finish_reason
                    status = "received" if reply.transport_complete else "no_transport_terminator"
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
            status = "http_error"
            error = FatalAPIError if exc.status_code in NON_RETRYABLE_STATUS else TransientAPIError
            failure = error(f"HTTP {exc.status_code}")
            failure.status = exc.status_code
            failure.retry_after = retry_after(exc.response.headers.get("retry-after"))
            raise failure from exc
        except (openai.APITimeoutError, openai.APIConnectionError, httpx.TransportError) as exc:
            category = exc.__cause__ if isinstance(exc, openai.APIConnectionError) else exc
            status = {
                httpx.ReadTimeout: "read_idle_timeout", httpx.ConnectTimeout: "connect_timeout",
                httpx.PoolTimeout: "pool_timeout",
            }.get(type(category), "network_error")
            raise TransientAPIError(status) from exc
        except (ValueError, TypeError, KeyError) as exc:
            status = "invalid_transport_frame"
            raise TransientAPIError(status) from exc
        finally:
            if attempt:
                finish(attempt, status, reason, time.monotonic() - start, **timing)

    async def _chat(self, request, receive, timing):
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
            reply = await self.client.chat.completions.create(**options)
            provider_id, model = reply.id, reply.model
            if reply.usage is not None:
                usage = _usage(reply.usage.model_dump(), provider_id, model)
                receive(usage)
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
                            chunk = json.loads(value)
                            if not isinstance(chunk, dict) or chunk.get("error"):
                                raise ValueError("invalid SSE frame")
                            provider_id = chunk.get("id") or provider_id
                            model = chunk.get("model") or model
                            if chunk.get("usage") is not None:
                                usage = _usage(chunk["usage"], provider_id, model)
                                receive(usage)
                            for choice in chunk.get("choices", []):
                                delta = choice.get("delta", {})
                                reasoning = delta.get("reasoning_content")
                                content = delta.get("content")
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
                return TransportResponse("".join(text), "model_idle_timeout", transport_complete=False)
        return TransportResponse(
            "".join(text), reason, usage.get("input_tokens"), usage.get("output_tokens"),
            usage.get("reasoning_tokens"), usage.get("cached_input_tokens"), provider_id, model,
            timing.get("first_data_seconds"), timing.get("first_content_seconds"), timing.get("first_reasoning_seconds"),
            transport_complete=bool(done and reason),
            uncached_tokens=usage.get("uncached_input_tokens"),
        )
