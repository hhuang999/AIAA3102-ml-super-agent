"""Resource control: time allocation, cost cap and hardware hints, behind one controller interface."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from ..errors import ConfigError
from .base import MIN_SCRIPT_SECONDS, Controller, ControlLog, Decision, StepLimits, required_start_seconds
from .default import RuleController
from .hardware import available_cpus, detect_gpus, parse_nvidia_smi
from .rules import MODES, CostCapRule, ResourceRule, Rule, TimeAllocationRule
from .state import GpuInfo, RunState, StepTiming, build_state

if TYPE_CHECKING:  # pragma: no cover
    from ..context import RunContext

from .adaptive import AdaptiveController

BUILTIN = {"static": Controller, "rules": RuleController, "adaptive": AdaptiveController}


def load_controller_class(spec: str) -> type[Controller]:
    """``static``, ``rules`` or ``package.module:ClassName`` (a subclass of :class:`Controller`)."""
    if spec in BUILTIN:
        return BUILTIN[spec]
    module_name, sep, cls_name = spec.partition(":")
    if not sep or not module_name or not cls_name:
        raise ConfigError(f"control.controller must be 'static', 'rules' or 'module:Class', got {spec!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ConfigError(f"control.controller: cannot import {module_name!r}: {exc}") from exc
    cls = getattr(module, cls_name, None)
    if not isinstance(cls, type) or not issubclass(cls, Controller):
        raise ConfigError(f"control.controller: {spec!r} is not a subclass of superagent.control.Controller")
    return cls


def make_controller(ctx: "RunContext") -> Controller:
    return load_controller_class(str(ctx.cfg["control"]["controller"]))(ctx)


__all__ = [
    "BUILTIN", "MIN_SCRIPT_SECONDS", "MODES", "ControlLog", "Controller", "CostCapRule", "Decision", "GpuInfo", "ResourceRule",
    "Rule", "RuleController", "RunState", "StepLimits", "StepTiming", "TimeAllocationRule", "available_cpus", "build_state",
    "detect_gpus", "load_controller_class", "make_controller", "parse_nvidia_smi", "required_start_seconds",
]
