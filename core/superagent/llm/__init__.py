"""LLM access: gateway, transports, lenient JSON parsing and schemas."""

from .gateway import (
    ROLES,
    FatalAPIError,
    LLMGateway,
    OpenAITransport,
    abort_decision,
    call_cost,
    TextResult,
    TransientAPIError,
    TransportResponse,
    make_gateway,
)
from .schema import Field

__all__ = [
    "ROLES",
    "FatalAPIError",
    "Field",
    "LLMGateway",
    "OpenAITransport",
    "abort_decision",
    "call_cost",
    "TextResult",
    "TransientAPIError",
    "TransportResponse",
    "make_gateway",
]
