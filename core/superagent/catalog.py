"""Method catalog and pretrained-model policy."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .config import RESOURCE_DIR
from .prompts import policy

CATALOG_PATH = RESOURCE_DIR / "method_catalog.yaml"
MANIFEST_NAME = "manifest.json"

log = logging.getLogger("superagent")


@dataclass
class CatalogState:
    entries: list[dict[str, Any]]   # entries shown to the model
    text: str                        # rendered policy paragraph + entries (catalog_text)
    missing: list[str]               # pretrained entries not available locally (catalog mode)


def load_catalog(path: str | Path = CATALOG_PATH) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or []
    if not isinstance(data, list):
        raise ValueError("method catalog must be a list of entries")
    return data


def load_manifest(local_dir: str | Path) -> dict[str, dict[str, Any]] | None:
    p = Path(local_dir) / MANIFEST_NAME
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    items = data.get("entries", data) if isinstance(data, dict) else data
    return {str(e.get("id")): e for e in items if isinstance(e, dict)}


def select_entries(
    entries: Sequence[Mapping[str, Any]],
    mode: str,
    manifest: Mapping[str, Mapping[str, Any]] | None,
    modalities: Sequence[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Return (entries to show, pretrained ids missing locally)."""
    shown: list[dict[str, Any]] = []
    missing: list[str] = []
    for e in entries:
        if modalities and e.get("modality") not in modalities:
            continue
        if e.get("pretrained"):
            if mode == "off":
                continue
            if mode == "catalog":
                rec = (manifest or {}).get(str(e["id"]), {})
                ok = rec.get("state") == "forward_ok" and rec.get("runner") == sys.executable and rec.get("ok")
                if ok and e.get("fetch") != "warmup":
                    from .training import digest

                    snapshot = rec.get("snapshot")
                    weights = rec.get("weights") or {}
                    ok = bool(snapshot and weights)
                    if ok:
                        folder = Path(snapshot).resolve()
                        ok = all((folder / name).is_file() and digest(folder / name) == sha
                                 for name, sha in weights.items())
                    if ok:
                        example = str(e["example"])
                        if e.get("library") == "timm":
                            weight = str(folder / next(iter(rec["weights"])))
                            example = example.replace("pretrained=True", f"pretrained=True, pretrained_cfg_overlay={{'file':{weight!r}}}")
                        else:
                            example = example.replace(str(e["hub_id"]),folder.as_posix())
                        e = {**e,"example":example,"weight_hashes":weights,"revision":rec.get("revision")}
                if not ok:
                    missing.append(str(e["id"]))
                    continue
        shown.append(dict(e))
    return shown, missing


def render_entry(e: Mapping[str, Any]) -> str:
    lines = [f"### {e.get('title', e.get('id'))}"]
    lines.append(f"- Usage: {e.get('usage')}" + (" (pretrained weights)" if e.get("pretrained") else ""))
    if e.get("hub_id"):
        lines.append(f"- Model id: {e['hub_id']}")
    lines.append(f"- When to use: {e.get('when_to_use', '').strip()}")
    if e.get("weight_hashes"):
        lines.append("- Verified weight SHA256: " + json.dumps(e["weight_hashes"],sort_keys=True))
    if e.get("input_notes"):
        lines.append(f"- Input requirements: {str(e['input_notes']).strip()}")
    if e.get("pitfalls"):
        lines.append(f"- Known pitfalls: {str(e['pitfalls']).strip()}")
    if e.get("library"):
        lines.append(f"- Library: {e['library']}")
    if e.get("example"):
        lines.append("- Example:")
        lines.append("```python\n" + str(e["example"]).rstrip() + "\n```")
    return "\n".join(lines)


def render(entries: Sequence[Mapping[str, Any]], mode: str) -> str:
    parts = [policy.policy_paragraph(mode), "", "## Method catalog"]
    for e in entries:
        parts.append(render_entry(e))
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def build_catalog(mode: str, local_dir: str | Path, modalities: Sequence[str], path: str | Path = CATALOG_PATH) -> CatalogState:
    entries = load_catalog(path)
    manifest = load_manifest(local_dir) if mode == "catalog" else None
    if mode == "catalog" and manifest is None:
        log.warning("no %s in %s: only non-pretrained catalog entries are shown (run python -m superagent.fetch_models)",
                    MANIFEST_NAME, local_dir)
    shown, missing = select_entries(entries, mode, manifest, modalities)
    if mode == "catalog" and manifest is not None and missing:
        log.warning("pretrained catalog entries not available locally: %s", ", ".join(missing))
    return CatalogState(entries=shown, text=render(shown, mode), missing=missing)


def offline_env(mode: str, local_dir: str | Path, hub_endpoint: str = "") -> dict[str, str]:
    """Environment variables for solution scripts under the given policy."""
    local = str(local_dir)
    env = {
        "HF_HOME": local,
        "HF_HUB_CACHE": str(Path(local) / "hub"),
        "TORCH_HOME": str(Path(local) / "torch"),
        "SENTENCE_TRANSFORMERS_HOME": str(Path(local) / "hub"),
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    if mode in ("catalog", "off"):
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"
    elif hub_endpoint:
        env["HF_ENDPOINT"] = hub_endpoint
    return env
