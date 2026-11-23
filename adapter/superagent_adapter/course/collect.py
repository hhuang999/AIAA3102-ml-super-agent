"""Stable import path for result collection. Implementation lives in ``course.back``."""

from .back.collect import collect_result, evaluate_final
from .common import validate_predictions

__all__ = ["collect_result", "evaluate_final", "validate_predictions"]
