"""Experience store: records of successful nodes, with keyword + vector retrieval."""

from __future__ import annotations

from collections import OrderedDict
import json
import queue
import re
import statistics
import time
import logging
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .utils import append_jsonl, atomic_write_json, now_iso

log = logging.getLogger("superagent")

MAX_QUERY_CHARS = 8000
METHOD_SUMMARY_FALLBACK_CHARS = 500


@dataclass
class Record:
    record_id: str
    kind: str
    plan: str
    method_summary: str
    effect: int
    score: float | None
    parent_score: float | None
    exec_seconds: float | None
    error_text: str | None
    created_at: str

    def index_text(self) -> str:
        if self.kind in ("repair", "diagnostic"):
            return self.error_text or ""
        return f"{self.plan}\n{self.method_summary}"


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")
_CAMEL_SPLIT_1 = re.compile(r"([a-z0-9])([A-Z])")
_CAMEL_SPLIT_2 = re.compile(r"([A-Z]{2,})([A-Z][a-z])")


def effect_label(
    kind: str,
    score: float | None,
    parent_score: float | None,
    better: Callable[[float, float], bool],
    reference_scores: Sequence[float] = (),
    *,
    ok: bool = True,
) -> int:
    if not ok or kind == "diagnostic":
        return -1
    if kind == "repair":
        return 1 if score is not None else -1
    if kind in ("draft", "synthesis"):
        if score is None:
            return -1
        refs = [float(s) for s in reference_scores if s is not None]
        if len(refs) >= 2:
            med = statistics.median(refs)
            return -1 if better(med, score) else 1
        return 1
    if score is None:
        return -1
    if parent_score is None:
        return 0
    if better(score, parent_score):
        return 1
    if better(parent_score, score):
        return -1
    return 0


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for tok in _TOKEN_RE.findall(text or ""):
        low = tok.lower()
        tokens.append(low)
        if tok.isascii():
            split = _CAMEL_SPLIT_2.sub(r"\1 \2", _CAMEL_SPLIT_1.sub(r"\1 \2", tok))
            parts = [p.lower() for p in split.split() if p]
            if len(parts) > 1:
                tokens.extend(p for p in parts if p != low)
    return tokens


class BM25:
    """Okapi BM25 with the non-negative idf ln(1 + (N - n + 0.5) / (n + 0.5))."""

    K1 = 1.5
    B = 0.75

    def __init__(self, corpus: list[list[str]]) -> None:
        import math
        from collections import Counter

        self.n_docs = len(corpus)
        self.doc_len = [len(d) for d in corpus]
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        self.tf = [Counter(d) for d in corpus]
        df: Counter[str] = Counter()
        for d in corpus:
            df.update(set(d))
        self.idf = {
            t: math.log(1.0 + (self.n_docs - n + 0.5) / (n + 0.5))
            for t, n in df.items()
        }

    def get_scores(self, query: list[str]) -> list[float]:
        scores = []
        for tf, dl in zip(self.tf, self.doc_len):
            s = 0.0
            norm = (
                self.K1 * (1 - self.B + self.B * dl / self.avgdl)
                if self.avgdl
                else self.K1
            )
            for t in query:
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.K1 + 1) / (f + norm)
            scores.append(s)
        return scores


def rrf_fuse(
    kw_ranks: Mapping[int, int], vec_ranks: Mapping[int, int], a: float, k: int
) -> list[tuple[int, float]]:
    """Reciprocal-rank fusion. Ranks start at 1; a missing rank contributes 0."""
    ids = set(kw_ranks) | set(vec_ranks)
    scored = []
    for i in ids:
        s = 0.0
        if i in kw_ranks:
            s += a / (k + kw_ranks[i])
        if i in vec_ranks:
            s += (1 - a) / (k + vec_ranks[i])
        scored.append((i, s))
    scored.sort(key=lambda t: (-t[1], t[0]))
    return scored


class LocalEmbedder:
    """Encode experience records in process and log local compute and token usage."""

    def __init__(
        self, model_name: str, snapshot: str, device: str, usage_log: Path | None
    ) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.usage_log = usage_log
        self.totals = {"seconds": 0.0, "calls": 0, "tokens": 0, "cost": 0.0}
        self._lock = threading.Lock()
        if device == "cuda" and not torch.cuda.is_available():
            log.warning(
                "experience.vector.device is cuda but no GPU is available; using cpu"
            )
            device = "cpu"
        try:
            self.model = SentenceTransformer(snapshot, device=device, local_files_only=True)
        except TypeError:
            if not Path(snapshot).is_dir():
                raise
            self.model = SentenceTransformer(snapshot, device=device)
        if device == "cpu":
            self.model.float()
        self.model.max_seq_length = 1024
        log.info(
            "local embedding: %s, device=%s dtype=%s",
            model_name,
            device,
            next(self.model.parameters()).dtype,
        )

    def encode(self, texts: Sequence[str], purpose: str) -> np.ndarray:
        start = time.monotonic()
        vecs, n_tokens = self._embed(list(texts))
        seconds = time.monotonic() - start
        with self._lock:
            self.totals["seconds"] += seconds
            self.totals["calls"] += 1
            self.totals["tokens"] += n_tokens
        if self.usage_log is not None:
            append_jsonl(
                self.usage_log,
                {
                    "ts": now_iso(),
                    "model": self.model_name,
                    "backend": "local",
                    "purpose": purpose,
                    "seconds": seconds,
                    "num_texts": len(texts),
                    "tokens": n_tokens,
                    "cost": 0.0,
                },
            )
        return vecs

    def _embed(self, texts: list[str]) -> tuple[np.ndarray, int]:
        tok = self.model.tokenizer(
            texts, truncation=True, max_length=self.model.max_seq_length
        )
        n_tokens = sum(len(ids) for ids in tok["input_ids"])
        vecs = self.model.encode(
            texts, batch_size=8, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32), n_tokens


MAX_EMBED_FAILURES = 5


class ExperienceStore:
    def __init__(
        self,
        cfg: Mapping[str, Any],
        path: Path,
        embedder: LocalEmbedder | None,
        *,
        background=False,
    ) -> None:
        self.cfg = cfg
        self.path = Path(path)
        self.embedder = embedder
        self.records: list[Record] = []
        self._ids: set[str] = set()
        self._vectors: dict[
            str, np.ndarray
        ] = {}  # record id -> vector (records without one are keyword-only)
        self._embed_failures = 0
        self._bm25: Any = None
        self._lock = threading.Lock()
        self._encode_lock = threading.Lock()
        self._closed = threading.Event()
        self._queries = OrderedDict()
        self._jobs = queue.SimpleQueue()
        self._thread = None
        if background and embedder is not None:
            self._thread = threading.Thread(
                target=self._index_loop, name="experience-index", daemon=True
            )
            self._thread.start()

    @property
    def vector_enabled(self) -> bool:
        return self.embedder is not None

    def __len__(self) -> int:
        with self._lock:
            return len(self.records)

    def _rebuild_bm25(self) -> None:
        self._bm25 = (
            BM25([tokenize(r.index_text()) for r in self.records])
            if self.records
            else None
        )

    def _try_encode(self, text: str, purpose: str) -> np.ndarray | None:
        """One embedding, or None on failure. Vector search is switched off after repeated failures."""
        if self.embedder is None:
            return None
        try:
            vec = self.embedder.encode([text[:MAX_QUERY_CHARS]], purpose)[0]
            self._embed_failures = 0
            return vec
        except Exception as exc:  # noqa: BLE001
            self._embed_failures += 1
            if self._embed_failures >= MAX_EMBED_FAILURES:
                log.warning(
                    "embedding failed %d times in a row (%s); vector search is disabled for the rest of the run",
                    self._embed_failures,
                    exc,
                )
                self.embedder = None
            else:
                log.warning(
                    "embedding failed (%s); this lookup uses keywords only", exc
                )
            return None

    def _index(self, rec):
        with self._encode_lock:
            if self._closed.is_set():
                return
            vec = self._try_encode(rec.index_text(), "index")
        if vec is not None:
            with self._lock:
                if rec.record_id in self._ids:
                    self._vectors[rec.record_id] = vec

    def _index_loop(self):
        while not self._closed.is_set():
            rec = self._jobs.get()
            if rec is None:
                return
            self._index(rec)

    def close(self):
        self._closed.set()
        if self._thread is not None:
            self._jobs.put(None)

    def rebuild(self):
        """Rebuild derived keyword/vector indexes from persisted records."""
        for row in json.loads(self.path.read_text(encoding="utf-8")):
            self.add(Record(**row), persist=False)

    def add(self, rec: Record, *, persist=True) -> bool:
        with self._lock:
            if rec.record_id in self._ids:
                return False
            self.records.append(rec)
            self._ids.add(rec.record_id)
            self._rebuild_bm25()
            if persist:
                atomic_write_json(self.path, [asdict(r) for r in self.records])
        if self.embedder is not None:
            if self._thread is not None:
                self._jobs.put(rec)
            else:
                self._index(rec)
        return True

    def remove(self, record_id: str) -> None:
        with self._lock:
            if record_id not in self._ids:
                return
            self.records = [r for r in self.records if r.record_id != record_id]
            self._ids.discard(record_id)
            self._vectors.pop(record_id, None)
            self._rebuild_bm25()
            atomic_write_json(self.path, [asdict(r) for r in self.records])

    def _keyword_ranks(self, query, n, eligible=None):
        if not self._bm25:
            return {}
        scores = self._bm25.get_scores(tokenize(query))
        ids = range(len(scores)) if eligible is None else eligible
        order = sorted(
            (i for i in ids if scores[i] > 0), key=lambda i: (-scores[i], i)
        )[:n]
        return {idx: r + 1 for r, idx in enumerate(order)}

    def _query(self, query):
        # When the live indexer is busy, use keywords instead of delaying a worker.
        if not self._encode_lock.acquire(blocking=self._thread is None):
            return None
        try:
            if query in self._queries:
                self._queries.move_to_end(query)
                return self._queries[query]
            vec = self._try_encode(query, "query")
            if vec is not None:
                self._queries[query] = vec
                if len(self._queries) > 128:
                    self._queries.popitem(last=False)
            return vec
        finally:
            self._encode_lock.release()

    def search(
        self, query: str, top_k: int, effect: int | None = None, kind: str | None = None,
        *, record_ids: set[str] | None = None,
    ) -> list[Record]:
        query = (query or "")[:MAX_QUERY_CHARS].strip()
        if not query or top_k <= 0:
            return []
        n = int(self.cfg.get("candidates_per_list", 50))
        a = float(self.cfg.get("keyword_weight", 0.5))
        with self._lock:
            records = list(self.records)
            eligible = [
                i
                for i, r in enumerate(records)
                if (effect is None or r.effect == effect)
                and (kind is None or r.kind == kind)
                and (record_ids is None or r.record_id in record_ids)
            ]
            kw = self._keyword_ranks(query, n, eligible)
            vecs = [
                (i, self._vectors[records[i].record_id])
                for i in eligible
                if a < 1 and records[i].record_id in self._vectors
            ]
        q = self._query(query) if vecs else None
        vec = {}
        if q is not None:
            sims = np.stack([v for _, v in vecs]) @ q
            candidates = [
                (float(s), i)
                for (i, _), s in zip(vecs, sims)
                if s >= self.cfg.get("min_similarity", 0.35)
            ]
            vec = {
                i: r + 1
                for r, (_, i) in enumerate(
                    sorted(candidates, key=lambda p: (-p[0], p[1]))[:n]
                )
            }
        weight = 1 if not vec else 0 if not kw else a
        fused = rrf_fuse(
            kw if weight else {},
            vec if weight < 1 else {},
            weight,
            int(self.cfg.get("rrf_k", 60)),
        )
        return [records[i] for i, _ in fused[:top_k]]


def decision_context(facts, proposal, store=None):
    """Retrieve a parent and comparable support and counterexamples across rungs and parent siblings."""
    visible = {x["operation_id"]: x for x in facts if x["seq"] <= proposal.as_of_seq}
    chosen = [key for key in proposal.evidence_ids if key in visible]
    parent = next((x for x in visible.values() if x["node_id"] == proposal.parent_id), None)
    if not parent:
        return []
    if parent["operation_id"] not in chosen:
        chosen.insert(0, parent["operation_id"])
    parent_trial = parent.get("trial_id")
    matching = [
        row for row in visible.values()
        if row["operation_id"] not in chosen
        and (row["split_id"], row["metric"]) == (parent["split_id"], parent["metric"])
        and (
            row.get("rung") == parent.get("rung")
            or (parent_trial and row.get("trial_id") == parent_trial)
            or row.get("parent_id") == parent.get("node_id")
        )
        and (
            row["action"] == proposal.action
            or (proposal.target and row.get("target") == proposal.target)
            or (parent.get("failed_stage") and row.get("failed_stage") == parent["failed_stage"])
            or (parent_trial and row.get("trial_id") == parent_trial)
            or row.get("parent_id") == parent.get("node_id")
        )
    ]
    query_parts = [
        parent.get("hypothesis") or "",
        parent.get("method_summary") or "",
        proposal.hypothesis or "",
        parent.get("reason") or "",
    ]
    rich_query = "\n".join(str(p) for p in query_parts if p).strip() or proposal.hypothesis
    pos_pool = sorted(
        [x for x in matching if x.get("deliverable") and (x.get("gain") or 0) > 0],
        key=lambda x: (x.get("gain") or 0.0, x["seq"]),
    )
    neg_pool = sorted(
        [
            x for x in matching
            if (x.get("gain") is not None and x["gain"] <= 0)
            or x.get("status") == "failed"
            or (parent.get("failed_stage") and x.get("failed_stage") == parent["failed_stage"])
        ],
        key=lambda x: (x.get("parent_id") == parent.get("node_id"), x["seq"]),
    )

    def pick(pool, limit, *, prioritize_direct=False):
        picked = []
        if not pool:
            return picked
        if prioritize_direct:
            for x in reversed(pool):
                if x.get("parent_id") == parent.get("node_id") and x["operation_id"] not in picked:
                    picked.append(x["operation_id"])
                if len(picked) >= limit:
                    return picked
        if store is not None and len(pool) > 1:
            matches = store.search(rich_query, limit, record_ids={x["operation_id"] for x in pool})
            for m in matches:
                if m.record_id not in picked:
                    picked.append(m.record_id)
                if len(picked) >= limit:
                    return picked
        for x in reversed(pool):
            if x["operation_id"] not in picked:
                picked.append(x["operation_id"])
            if len(picked) >= limit:
                break
        return picked

    chosen.extend(pick(pos_pool, 2, prioritize_direct=False))
    chosen.extend(pick(neg_pool, 2, prioritize_direct=True))
    fields = (
        "operation_id", "parent_id", "action", "method", "hypothesis", "method_summary",
        "changes_summary", "target", "seed", "rung", "score", "score_verified",
        "deliverable", "gain", "diagnostic", "reason", "failed_stage", "seconds",
    )
    selected = [{field: visible[key].get(field) for field in fields} for key in dict.fromkeys(chosen)]
    for row in selected:
        if row["failed_stage"] not in {"executed", "assessed", "verified"}:
            row["reason"] = None
    return selected


def make_embedder(
    exp_cfg: Mapping[str, Any],
    local_dir: str,
    usage_log: Path | None,
) -> LocalEmbedder | None:
    """Load local weights; an unavailable encoder leaves keyword retrieval active."""
    vec = exp_cfg.get("vector", {})
    if not exp_cfg.get("enabled", True) or not vec.get("enabled", True):
        return None
    model = str(vec["model"])
    try:
        snapshot = model
        if not Path(snapshot).is_dir():
            from .catalog import load_manifest

            rec = (load_manifest(local_dir) or {}).get("experience_embedding", {})
            if (
                rec.get("hub_id") == model
                and rec.get("state") == "forward_ok"
                and Path(rec["snapshot"]).is_dir()
            ):
                snapshot = rec["snapshot"]
            else:
                from huggingface_hub import snapshot_download

                snapshot = snapshot_download(
                    model, cache_dir=str(Path(local_dir) / "hub"), local_files_only=True
                )
        return LocalEmbedder(model, snapshot, str(vec.get("device", "cpu")), usage_log)
    except Exception as exc:  # noqa: BLE001 - keyword retrieval is available without local weights
        log.warning(
            "could not load local embedding model %s (%s); using keywords only. "
            "To prepare the model, run python -m superagent.fetch_models --only experience_embedding",
            model,
            exc,
        )
        return None
