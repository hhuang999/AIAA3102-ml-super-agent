"""Stable import path for the optional report writer. Implementation lives in ``course.back``."""

from .back.report_llm import ReportWriterError, make_chat_writer

__all__ = ["ReportWriterError", "make_chat_writer"]
