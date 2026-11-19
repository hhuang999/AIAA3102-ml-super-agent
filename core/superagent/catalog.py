"""Method catalog and pretrained-model policy."""

from __future__ import annotations

import json
import logging
import shutil
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
                        e = {**e,"example":example,"weight_hashes":weights,"revision":rec.get("revision"),
                             "snapshot": folder.as_posix()}
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


@dataclass(frozen=True)
class MethodMatch:
    method_id: str
    version: str
    modality: str
    hypothesis: str
    source: str
    eligible: bool
    reason: str
    expected_seconds: float
    recipe_id: str | None
    entry: dict

    @property
    def key(self):
        return f"{self.method_id}@{self.version}"


BASELINE_ORDER = (
    "tab_linear", "text_tfidf_linear", "img_classic", "tab_forest", "img_scratch_cnn",
    "tab_gbdt", "text_scratch_nn",
)


def match_methods(facts, catalog, task_cards=(), *, local_dir=None):
    """Match choices against the actual reader, installed libraries and verified assets."""
    capabilities = facts["capabilities"]
    dependencies = capabilities["dependencies"]
    splits = facts["input_manifest"]["splits"]
    modalities = capabilities["modalities"]
    matches = []
    available = {entry["id"]: entry for entry in catalog.entries}
    for entry in load_catalog():
        ident = entry["id"]
        effective = dict(available.get(ident, entry))
        library = effective.get("library", "").split(" / ")
        requirements = set(effective.get("requires", ()))
        reason = "eligible"
        snapshot = effective.get("snapshot")
        if entry["modality"] not in modalities:
            reason = "incompatible_modality"
        elif facts["metric"] not in effective.get("metrics", ["accuracy", "log_loss"]):
            reason = "incompatible_metric"
        elif not any(name in dependencies for name in library) or not requirements <= dependencies.keys():
            reason = "missing_dependency"
        elif not set(effective.get("required_fields", ())) <= set(splits["train"]["features"]):
            reason = "missing_bound_features"
        elif effective["pretrained"] and not (snapshot and local_dir and
                 Path(snapshot).resolve().is_relative_to(Path(local_dir).resolve()) and effective.get("weight_hashes")):
            reason = "missing_verified_local_weights"
        elif effective.get("recipe_id") and not (
            all(b["format"] == "csv" and not b["file_features"] for b in splits.values())
            and all(set(b["features"]) == set(splits["train"]["features"]) for b in splits.values())
            and modalities == ["tabular"]
        ):
            reason = "recipe_requires_consistent_tabular_csv_fields"
        fallback = effective.get("alternative")
        effective.update(version=str(effective.get("version", 1)), source=effective.get("source", "SuperAgent bundled method catalog"),
                         required_fields=effective.get("required_fields", []), alternatives=[fallback] if fallback else [],
                         dependencies=sorted(requirements | set(library)))
        hypothesis = effective["title"] + ". " + effective["when_to_use"]
        matches.append(MethodMatch(ident, effective["version"], effective["modality"], hypothesis,
            effective["source"], reason == "eligible", reason, float(effective.get("estimated_seconds", 600)),
            effective.get("recipe_id"), effective))
    for card in task_cards:
        effective = {**card, "title": card["id"], "when_to_use": card["method"], "pitfalls": card["counterexample"],
                     "usage": "task method", "pretrained": False}
        requirements = card.get("requires", [])
        if not isinstance(requirements, list) or any(not isinstance(name, str) for name in requirements):
            raise ValueError(f"{card['id']}: requires must list installed packages")
        required_fields = card.get("required_fields", [])
        if not isinstance(required_fields, list) or any(not isinstance(name, str) for name in required_fields):
            raise ValueError(f"{card['id']}: required_fields must list bound fields")
        if not set(requirements) <= dependencies.keys():
            reason = "missing_dependency"
        elif not set(required_fields) <= set(splits["train"]["features"]):
            reason = "missing_bound_features"
        else:
            reason = "eligible"
        matches.append(MethodMatch(card["id"], card["version"], card["modality"], card["method"], card["source"],
                                   reason == "eligible", reason, 600, None, effective))
    priority = {name: index for index, name in enumerate(BASELINE_ORDER)}
    return sorted(matches, key=lambda m: (priority.get(m.method_id, len(priority)), m.expected_seconds, m.key))


def stage_method_assets(matches, root, local_dir):
    """Copy verified snapshots with their configuration into the training allowlist."""
    mounts = {}
    for match in matches:
        item = match.entry
        if not match.eligible or not item["pretrained"]:
            continue
        snapshot = Path(item["snapshot"]).resolve()
        if any(not path.resolve().is_relative_to(Path(local_dir).resolve()) for path in snapshot.rglob("*")):
            raise ValueError(f"asset path outside verified cache: {match.key}")
        destination = root / match.method_id
        shutil.copytree(snapshot, destination)
        from .training import digest

        if any(digest(destination / path) != sha for path, sha in item["weight_hashes"].items()):
            raise ValueError(f"asset contents changed while staging {match.key}")
        sandbox_path = f"/method_assets/{match.method_id}"
        item["example"] = item["example"].replace(snapshot.as_posix(), sandbox_path)
        mounts[sandbox_path] = destination
    return mounts
