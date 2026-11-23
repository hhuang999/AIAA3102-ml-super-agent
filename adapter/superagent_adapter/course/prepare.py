"""Stable import path for the input side. Implementation lives in ``course.front``."""

from .front.prepare import prepare_task, verify_inputs

__all__ = ["prepare_task", "verify_inputs"]
