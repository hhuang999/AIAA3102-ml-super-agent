"""Course delivery split into an input front and a result back.

``front`` turns a course folder into a Core task directory.
``back`` checks the Core output, scores held-out labels, and records where the files are.
Neither side calls the model. ``course run`` is the only command that starts the Core.
"""

from superagent_adapter import __version__

__all__ = ["__version__"]
