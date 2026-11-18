"""Exception types shared across the package."""

from __future__ import annotations


class InvalidInputError(Exception):
    """The task directory, configuration or environment is not usable (exit code 3).

    ``write_status`` is False when nothing may be written to the output directory
    (for example because it already holds the results of an earlier run).
    """

    def __init__(self, message: str, write_status: bool = True) -> None:
        super().__init__(message)
        self.write_status = write_status


class ConfigError(InvalidInputError):
    """A configuration key is unknown or has a value of the wrong type."""


class LLMUnavailableError(Exception):
    """The LLM endpoint kept failing after all retries (or the deadline passed)."""


class StructuredOutputError(Exception):
    """A structured LLM call did not return a valid JSON object within the attempt limit."""


class DeadlineReached(Exception):
    """Raised inside long operations when the hard deadline has been reached."""
