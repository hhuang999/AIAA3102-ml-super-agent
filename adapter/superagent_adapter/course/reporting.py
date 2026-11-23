"""Stable import path for the optional report draft. Implementation lives in ``course.back``."""

from .back.reporting import generate_report

__all__ = ["generate_report"]
