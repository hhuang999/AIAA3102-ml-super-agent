"""SuperAgent: an autonomous agent that builds classification models for a task directory."""

from __future__ import annotations

from typing import Any

__all__ = ["run_task"]
__version__ = "0.7.1"


def __getattr__(name: str) -> Any:
    # Imported lazily so that ``python -m superagent.run`` does not import the module twice.
    if name == "run_task":
        from .run import run_task

        return run_task
    raise AttributeError(name)
