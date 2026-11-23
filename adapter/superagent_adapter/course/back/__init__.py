"""Result side. Add new checks, scores, and handoff records here."""

from .collect import collect_result, evaluate_final
from .outcome import blocked_outcome, write_outcome

__all__ = ["blocked_outcome", "collect_result", "evaluate_final", "write_outcome"]
