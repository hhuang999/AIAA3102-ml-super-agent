"""Stable import path for summary checks. Implementation lives in ``course.back``."""

from .back.consistency import check_summary

__all__ = ["check_summary"]
