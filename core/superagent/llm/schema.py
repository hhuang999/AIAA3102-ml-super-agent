"""A tiny schema format for structured LLM answers, with coercing validation.

A schema is a mapping ``field -> Field``. Supported types: ``string``, ``number``,
``boolean``, ``enum``, ``string_list``, ``object``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class Field:
    type: str
    description: str
    required: bool = True
    nullable: bool = False
    values: tuple[str, ...] = field(default_factory=tuple)


Schema = Mapping[str, Field]

_TRUE = {"true", "yes", "1"}
_FALSE = {"false", "no", "0"}


def _coerce(name: str, spec: Field, value: Any) -> tuple[Any, str | None]:
    if value is None:
        if spec.nullable:
            return None, None
        return None, f"field '{name}' must not be null"
    t = spec.type
    if t == "string":
        if isinstance(value, str):
            return value, None
        if isinstance(value, (int, float, bool)):
            return str(value), None
        return None, f"field '{name}' must be a string"
    if t == "number":
        if isinstance(value, bool):
            return None, f"field '{name}' must be a number"
        if isinstance(value, (int, float)):
            return float(value), None
        if isinstance(value, str):
            s = value.strip()
            if s.lower() in {"null", "none", ""} and spec.nullable:
                return None, None
            try:
                return float(s), None
            except ValueError:
                return None, f"field '{name}' must be a number, got {value!r}"
        return None, f"field '{name}' must be a number"
    if t == "boolean":
        if isinstance(value, bool):
            return value, None
        if isinstance(value, str) and value.strip().lower() in _TRUE | _FALSE:
            return value.strip().lower() in _TRUE, None
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value), None
        return None, f"field '{name}' must be a boolean (true or false)"
    if t == "enum":
        if isinstance(value, str):
            v = value.strip()
            for allowed in spec.values:
                if v.lower() == allowed.lower():
                    return allowed, None
        return None, f"field '{name}' must be one of {list(spec.values)}, got {value!r}"
    if t == "string_list":
        if isinstance(value, list) and all(isinstance(x, str) for x in value):
            return value, None
        if isinstance(value, list):
            return [str(x) for x in value], None
        return None, f"field '{name}' must be an array of strings"
    if t == "object":
        if isinstance(value, dict):
            return value, None
        return None, f"field '{name}' must be a JSON object"
    if t == "any":
        return value, None
    raise ValueError(f"unknown schema type {t}")  # pragma: no cover


def validate(obj: Mapping[str, Any], schema: Schema) -> tuple[dict[str, Any], list[str]]:
    """Return the coerced object and a list of error messages (empty when valid)."""
    out: dict[str, Any] = dict(obj)
    errors: list[str] = []
    for name, spec in schema.items():
        if name not in obj:
            if spec.required:
                errors.append(f"missing required field '{name}'")
            else:
                out[name] = None
            continue
        value, err = _coerce(name, spec, obj[name])
        if err:
            errors.append(err)
        else:
            if isinstance(value, float) and not math.isfinite(value):
                if spec.nullable:
                    value = None
                else:
                    errors.append(f"field '{name}' must be a finite number")
                    continue
            out[name] = value
    return out, errors


def describe(schema: Schema) -> str:
    """Plain-text description of the fields, for the prompt."""
    lines = ["Return a single JSON object with these fields:"]
    for name, spec in schema.items():
        t = spec.type
        if t == "enum":
            t = "one of " + " | ".join(f'"{v}"' for v in spec.values)
        elif t == "string_list":
            t = "array of strings"
        if spec.nullable:
            t += " or null"
        lines.append(f'- "{name}" ({t}): {spec.description}')
    lines.append("Output only the JSON object, with no other text.")
    return "\n".join(lines)
