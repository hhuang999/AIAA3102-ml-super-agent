"""Method catalog and pretrained-model policy."""

from __future__ import annotations

import json
import logging
import os
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


def assign_draft_hypotheses(
    entries: Sequence[Mapping[str, Any]],
    modalities: Sequence[str],
    n: int,
) -> list[str]:
    """Build ``n`` structurally orthogonal initial draft hypotheses from the active method catalog."""
    if n <= 0:
        return []
    mods = list(modalities or [])
    if len(mods) >= 2:
        fusion_archetypes = [
            f"Multimodal late-fusion baseline across {mods}: extract strong modality-specific representations for each input modality and train a regularized gradient-boosted or linear meta-classifier with inner K-fold cross-validation.",
            f"End-to-end joint neural multimodal architecture across {mods}: encode each modality into a shared embedding space with modality-specific backbones/projections, cross-modal interaction/attention, dropout, and cosine LR schedule.",
            f"Multi-view hybrid ensemble across {mods}: combine engineered statistical/lexical/visual summary features for tree boosters with neural modality embeddings via out-of-fold stacking.",
            f"Orthogonal backbone multimodal pipeline across {mods}: use distinct secondary catalog encoders per modality with calibrated probability blending.",
        ]
        return [fusion_archetypes[i % len(fusion_archetypes)] for i in range(n)]

    if entries:
        by_id = {str(e.get("id")): e for e in entries}
        candidates: list[str] = []
        # If tabular umbrella entries are active, expand into genuinely distinct competitive paradigms first
        if mods == ["tabular"] and ("tab_gbdt" in by_id or "tab_mlp" in by_id):
            if "tab_tabicl_incontext" in by_id:
                e = by_id["tab_tabicl_incontext"]
                candidates.append(f"{e['title']}: {e['when_to_use']}")
            if "tab_gbdt" in by_id:
                candidates.append(
                    "Gradient-boosted decision trees (LightGBM + XGBoost) with rich domain feature engineering "
                    "(interactions, aggregations, ratios, frequency/ordinal encodings) and K-fold fold-averaged predictions."
                )
            if "tab_mlp" in by_id:
                candidates.append(
                    "Tabular deep neural architecture (learned categorical embeddings + standardized numeric projections "
                    "+ residual MLP or Feature-Tokenizer Transformer blocks with AdamW and cosine schedule) using K-fold fold-averaging."
                )
            if "tab_gbdt" in by_id:
                candidates.append(
                    "CatBoost native categorical boosting combined with ExtraTrees / Random Forest on engineered features, "
                    "using inner out-of-fold probability blending."
                )
            for eid, e in by_id.items():
                if eid not in ("tab_tabicl_incontext", "tab_gbdt", "tab_mlp"):
                    candidates.append(f"{e['title']}: {e['when_to_use']}")
        else:
            # General catalog bucketing for image, text, and any future catalog modality:
            # interleave fine-tuned pretrained backbones, pretrained feature-extractors, and from-scratch architectures
            finetune = [e for e in entries if e.get("pretrained") and "finetune" in str(e.get("id", ""))]
            features = [e for e in entries if e.get("pretrained") and "finetune" not in str(e.get("id", ""))]
            scratch = [e for e in entries if not e.get("pretrained")]
            # Interleave distinct pretrained backbones first, then feature-based, then scratch
            ordered: list[Mapping[str, Any]] = []
            if finetune:
                ordered.append(finetune[0])
            if features:
                ordered.append(features[0])
            for e in finetune[1:]:
                ordered.append(e)
            for e in features[1:]:
                ordered.append(e)
            for e in scratch:
                ordered.append(e)
            candidates = [
                f"{e['title']} ({e['id']}): {e.get('when_to_use', '')}".strip()
                for e in ordered
            ]
        if candidates:
            return [candidates[i % len(candidates)] for i in range(n)]

    # Modality-agnostic fallback (audio, time-series, scientific arrays, or pretrained mode == off)
    generic = [
        "Domain-specific statistical/signal feature engineering paired with gradient-boosted decision trees and K-fold fold-averaging.",
        "End-to-end deep neural network tailored to the raw input structure (convolutional, recurrent, or self-attention blocks with normalization, dropout, and cosine LR).",
        "Hybrid multi-family architecture combining engineered structural features with a distinct representation-learning model via inner out-of-fold ensembling.",
    ]
    return [generic[i % len(generic)] for i in range(n)]


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
GPU_BASELINE_ORDER = (
    "tab_gbdt", "img_dinov2_base_features", "img_dinov2_small_features",
    "text_qwen3_embedding_features", "img_convnext_tiny_finetune", "img_efficientnetv2_s_finetune",
    "text_deberta_v3_base_finetune", "text_modernbert_base_finetune", "text_xlm_roberta_base_finetune",
    "img_resnet50_finetune", "tab_mlp", "tab_linear", "text_tfidf_linear", "img_classic",
    "tab_forest", "img_scratch_cnn", "text_scratch_nn",
)
DEFAULT_CATALOG_METRICS = (
    "accuracy", "log_loss", "roc_auc", "f1_macro", "f1_weighted", "balanced_accuracy",
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
        elif facts["metric"] not in effective.get("metrics", DEFAULT_CATALOG_METRICS):
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
    order = GPU_BASELINE_ORDER if int(capabilities.get("num_gpus", 0) or 0) > 0 else BASELINE_ORDER
    priority = {name: index for index, name in enumerate(order)}
    return sorted(matches, key=lambda m: (priority.get(m.method_id, len(priority)), m.expected_seconds, m.key))


def stage_method_assets(matches, root, local_dir):
    """Hard-link (or copy) dereferenced verified snapshots into the training allowlist."""
    mounts = {}
    local_root = Path(local_dir).resolve()
    for match in matches:
        item = match.entry
        if not match.eligible or not item["pretrained"]:
            continue
        snapshot = Path(item["snapshot"]).resolve()
        if any(not path.resolve().is_relative_to(local_root) for path in snapshot.rglob("*")):
            raise ValueError(f"asset path outside verified cache: {match.key}")
        destination = root / match.method_id
        destination.mkdir(parents=True, exist_ok=True)
        for src_entry in snapshot.rglob("*"):
            rel = src_entry.relative_to(snapshot)
            dst_entry = destination / rel
            if src_entry.is_dir():
                dst_entry.mkdir(parents=True, exist_ok=True)
                continue
            real_src = src_entry.resolve()
            dst_entry.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(real_src, dst_entry)
            except OSError:
                shutil.copy2(real_src, dst_entry)
        from .training import digest

        if any(digest(destination / path) != sha for path, sha in item["weight_hashes"].items()):
            raise ValueError(f"asset contents changed while staging {match.key}")
        sandbox_path = f"/method_assets/{match.method_id}"
        item["example"] = item["example"].replace(snapshot.as_posix(), sandbox_path)
        raw_snap = str(item.get("snapshot", ""))
        if raw_snap and raw_snap != snapshot.as_posix():
            item["example"] = item["example"].replace(raw_snap, sandbox_path)
        mounts[sandbox_path] = destination
    return mounts
