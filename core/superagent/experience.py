"""Experience store: records of successful nodes, with keyword + vector retrieval."""

from __future__ import annotations

import logging
import os
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
        if self.kind == "repair":
            return self.error_text or ""
        return f"{self.plan}\n{self.method_summary}"


def effect_label(kind: str, score: float | None, parent_score: float | None, better: Callable[[float, float], bool]) -> int:
    if kind in ("draft", "synthesis", "repair"):
        return 1
    if score is None or parent_score is None:
        return 0
    if better(score, parent_score):
        return 1
    if better(parent_score, score):
        return -1
    return 0


def tokenize(text: str) -> list[str]:
    return text.lower().split()


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
        self.idf = {t: math.log(1.0 + (self.n_docs - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def get_scores(self, query: list[str]) -> list[float]:
        scores = []
        for tf, dl in zip(self.tf, self.doc_len):
            s = 0.0
            norm = self.K1 * (1 - self.B + self.B * dl / self.avgdl) if self.avgdl else self.K1
            for t in query:
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.K1 + 1) / (f + norm)
            scores.append(s)
        return scores


def rrf_fuse(kw_ranks: Mapping[int, int], vec_ranks: Mapping[int, int], a: float, k: int) -> list[tuple[int, float]]:
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


class Embedder:
    """Common part of the embedding backends: usage accounting and the usage log."""

    def __init__(self, model_name: str, backend: str, price_per_million: float, usage_log: Path | None) -> None:
        self.model_name = model_name
        self.backend = backend
        self.price = float(price_per_million)
        self.usage_log = usage_log
        self.totals = {"calls": 0, "tokens": 0, "cost": 0.0}
        self._lock = threading.Lock()

    def _embed(self, texts: list[str]) -> tuple[np.ndarray, int]:
        """Return (one unit-length vector per text, number of tokens used)."""
        raise NotImplementedError

    def encode(self, texts: Sequence[str], purpose: str) -> np.ndarray:
        vecs, n_tokens = self._embed(list(texts))
        cost = n_tokens * self.price / 1e6
        with self._lock:
            self.totals["calls"] += 1
            self.totals["tokens"] += n_tokens
            self.totals["cost"] += cost
        if self.usage_log is not None:
            append_jsonl(self.usage_log, {"ts": now_iso(), "model": self.model_name, "backend": self.backend, "purpose": purpose,
                                          "num_texts": len(texts), "tokens": n_tokens, "cost": cost})
        return vecs


def unit_rows(vecs: np.ndarray) -> np.ndarray:
    """Scale every row to length 1 (rows of zeros stay zero)."""
    vecs = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms > 0, norms, 1.0)


class LocalEmbedder(Embedder):
    """Runs a sentence-transformers model on this machine."""

    def __init__(self, model_name: str, device: str, cache_dir: str | None, price_per_million: float, usage_log: Path | None) -> None:
        super().__init__(model_name, "local", price_per_million, usage_log)
        import torch
        from sentence_transformers import SentenceTransformer

        if device == "cuda" and not torch.cuda.is_available():
            log.warning("experience.vector.device is cuda but no GPU is available; using cpu")
            device = "cpu"
        kwargs = {"model_kwargs": {"dtype": torch.float32}} if device == "cpu" else {}
        try:
            self.model = SentenceTransformer(model_name, device=device, cache_folder=cache_dir, **kwargs)
        except TypeError:
            self.model = SentenceTransformer(model_name, device=device, cache_folder=cache_dir)
        self.model.max_seq_length = 1024

    def _embed(self, texts: list[str]) -> tuple[np.ndarray, int]:
        try:
            tok = self.model.tokenizer(texts, truncation=True, max_length=self.model.max_seq_length)
            n_tokens = int(sum(len(x) for x in tok["input_ids"]))
        except Exception:  # noqa: BLE001
            n_tokens = int(sum(len(t.split()) for t in texts))
        vecs = self.model.encode(texts, batch_size=8, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32), n_tokens


class ApiEmbedder(Embedder):
    """Calls an OpenAI-compatible embeddings endpoint."""

    BATCH = 16

    def __init__(self, model_name: str, base_url: str, api_key: str, timeout: float, price_per_million: float, usage_log: Path | None) -> None:
        super().__init__(model_name, "api", price_per_million, usage_log)
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=1)

    def _embed(self, texts: list[str]) -> tuple[np.ndarray, int]:
        rows: list[list[float]] = []
        n_tokens = 0
        for start in range(0, len(texts), self.BATCH):
            batch = [t if t.strip() else " " for t in texts[start:start + self.BATCH]]
            resp = self._client.embeddings.create(model=self.model_name, input=batch)
            data = sorted(resp.data, key=lambda d: d.index)
            if len(data) != len(batch):
                raise RuntimeError(f"embeddings endpoint returned {len(data)} vectors for {len(batch)} texts")
            rows.extend(d.embedding for d in data)
            usage = getattr(resp, "usage", None)
            used = getattr(usage, "total_tokens", None) or getattr(usage, "prompt_tokens", None)
            n_tokens += int(used) if used else int(sum(len(t.split()) for t in batch))
        return unit_rows(np.asarray(rows, dtype=np.float32)), n_tokens


MAX_EMBED_FAILURES = 5


class ExperienceStore:
    def __init__(self, cfg: Mapping[str, Any], path: Path, embedder: Embedder | None) -> None:
        self.cfg = cfg
        self.path = Path(path)
        self.embedder = embedder
        self.records: list[Record] = []
        self._ids: set[str] = set()
        self._vectors: dict[str, np.ndarray] = {}   # record id -> vector (records without one are keyword-only)
        self._embed_failures = 0
        self._bm25: Any = None
        self._lock = threading.Lock()

    @property
    def vector_enabled(self) -> bool:
        return self.embedder is not None

    def __len__(self) -> int:
        with self._lock:
            return len(self.records)

    def _rebuild_bm25(self) -> None:
        self._bm25 = BM25([tokenize(r.index_text()) for r in self.records]) if self.records else None

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
                log.warning("embedding failed %d times in a row (%s); vector search is disabled for the rest of the run",
                            self._embed_failures, exc)
                self.embedder = None
            else:
                log.warning("embedding failed (%s); this lookup uses keywords only", exc)
            return None

    def add(self, rec: Record) -> bool:
        with self._lock:
            if rec.record_id in self._ids:
                return False
            vec = self._try_encode(rec.index_text(), "index")
            self.records.append(rec)
            self._ids.add(rec.record_id)
            if vec is not None:
                self._vectors[rec.record_id] = vec
            self._rebuild_bm25()
            atomic_write_json(self.path, [asdict(r) for r in self.records])
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

    def _keyword_ranks(self, query: str, n: int) -> dict[int, int]:
        if self._bm25 is None or not self.records:
            return {}
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:n]
        return {idx: r + 1 for r, idx in enumerate(order)}

    def _vector_ranks(self, query: str, n: int) -> dict[int, int]:
        idxs = [i for i, r in enumerate(self.records) if r.record_id in self._vectors]
        if not idxs:
            return {}
        q = self._try_encode(query, "query")
        if q is None:
            return {}
        mat = np.stack([self._vectors[self.records[i].record_id] for i in idxs])
        dist = np.linalg.norm(mat - q[None, :], axis=1)
        order = sorted(range(len(idxs)), key=lambda k: (float(dist[k]), idxs[k]))[:n]
        return {idxs[k]: r + 1 for r, k in enumerate(order)}

    def search(self, query: str, top_k: int, effect: int | None = None, kind: str | None = None) -> list[Record]:
        with self._lock:
            if not self.records:
                return []
            query = (query or "")[:MAX_QUERY_CHARS]
            n = int(self.cfg.get("candidates_per_list", 50))
            a = float(self.cfg.get("keyword_weight", 0.5))
            k = int(self.cfg.get("rrf_k", 60))
            kw = self._keyword_ranks(query, n) if a > 0 or self.embedder is None else {}
            vec = self._vector_ranks(query, n) if a < 1 else {}
            if not vec and not kw:
                kw = self._keyword_ranks(query, n)  # vectors unavailable: fall back to keywords
            if not vec:
                fused = sorted(kw.items(), key=lambda t: t[1])
                order = [i for i, _r in fused]
            elif not kw:
                order = [i for i, _r in sorted(vec.items(), key=lambda t: t[1])]
            else:
                order = [i for i, _s in rrf_fuse(kw, vec, a, k)]
            out: list[Record] = []
            for i in order:
                r = self.records[i]
                if effect is not None and r.effect != effect:
                    continue
                if kind is not None and r.kind != kind:
                    continue
                out.append(r)
                if len(out) >= top_k:
                    break
            return out


def make_embedder(
    exp_cfg: Mapping[str, Any], local_dir: str, usage_log: Path | None, environ: Mapping[str, str] | None = None
) -> Embedder | None:
    """Build the embedding backend chosen in the configuration. None means keyword search only."""
    vec = exp_cfg.get("vector", {})
    if not exp_cfg.get("enabled", True) or not vec.get("enabled", True):
        return None
    price = float(vec.get("price_per_million", 0.0))
    try:
        if str(vec.get("backend", "local")) == "api":
            env = os.environ if environ is None else environ
            key = env.get(str(vec.get("api_key_env", "")), "")
            if not key:
                raise RuntimeError(f"environment variable {vec.get('api_key_env')} is not set")
            return ApiEmbedder(str(vec["api_model"]), str(vec["api_base_url"]), key, float(vec.get("api_timeout_seconds", 60)),
                               price, usage_log)
        return LocalEmbedder(str(vec["model"]), str(vec.get("device", "cuda")), str(Path(local_dir) / "hub"), price, usage_log)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not set up the embedding model (%s); experience search uses keywords only", exc)
        return None


def vector_key_missing(exp_cfg: Mapping[str, Any], environ: Mapping[str, str]) -> str | None:
    """Name of the API key variable that is required but not set, or None."""
    vec = exp_cfg.get("vector", {})
    if exp_cfg.get("enabled", True) and vec.get("enabled", True) and str(vec.get("backend", "local")) == "api":
        name = str(vec.get("api_key_env", ""))
        if not environ.get(name):
            return name
    return None
