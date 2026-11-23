"""Stable import path for the handoff record. Implementation lives in ``course.back``."""

from .back.outcome import blocked_outcome, write_outcome

__all__ = ["blocked_outcome", "write_outcome"]
