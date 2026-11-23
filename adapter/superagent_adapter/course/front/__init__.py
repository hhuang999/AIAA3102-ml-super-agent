"""Input side. Add new course-format readers and task-directory rules here."""

from .prepare import prepare_task, verify_inputs

__all__ = ["prepare_task", "verify_inputs"]
