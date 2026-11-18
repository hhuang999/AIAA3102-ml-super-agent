import pytest

from superagent.experience import ExperienceStore, Record, effect_label, rrf_fuse


def better_hi(a, b):
    return a > b


def better_lo(a, b):
    return a < b


@pytest.mark.parametrize("kind,score,parent,better,label", [
    ("draft", 0.8, None, better_hi, 1),
    ("repair", 0.8, None, better_hi, 1),
    ("refine", 0.82, 0.80, better_hi, 1),
    ("refine", 0.78, 0.80, better_hi, -1),
    ("refine", 0.45, 0.50, better_lo, 1),
    ("borrow", 0.80, 0.80, better_hi, 0),
])
def test_labels(kind, score, parent, better, label):
    assert effect_label(kind, score, parent, better) == label


def test_fusion_numbers():
    fused = dict(rrf_fuse({0: 1, 1: 2, 2: 3}, {0: 3, 1: 1}, 0.5, 60))
    assert fused[0] == pytest.approx(0.016133, abs=1e-6)
    assert fused[1] == pytest.approx(0.016261, abs=1e-6)
    assert fused[2] == pytest.approx(0.007937, abs=1e-6)
    order = [i for i, _ in rrf_fuse({0: 1, 1: 2, 2: 3}, {0: 3, 1: 1}, 0.5, 60)]
    assert order == [1, 0, 2]


def rec(i, kind="refine", effect=1, plan="", err=None):
    return Record(i, kind, plan, "summary", effect, 0.5, None, 1.0, err, "now")


def store(tmp_path):
    return ExperienceStore({"candidates_per_list": 50, "keyword_weight": 0.5, "rrf_k": 60}, tmp_path / "r.json", None)


def test_filter_after_fusion(tmp_path):
    s = store(tmp_path)
    s.add(rec("B", effect=-1, plan="alpha beta gamma"))
    s.add(rec("A", effect=1, plan="alpha beta"))
    s.add(rec("C", effect=1, plan="unrelated words"))
    got = s.search("alpha beta gamma", 1, effect=1)
    assert [r.record_id for r in got] == ["A"]


def test_repair_indexed_by_error(tmp_path):
    s = store(tmp_path)
    s.add(rec("R", kind="repair", plan="add a cast to float", err="ValueError: could not convert string to float in column price"))
    s.add(rec("F", kind="refine", plan="add a cast to float"))
    s.add(rec("X", kind="refine", plan="tune trees"))
    got = s.search("ValueError: could not convert string to float", 1, kind="repair")
    assert [r.record_id for r in got] == ["R"]
    ranks = s._keyword_ranks("add a cast to float", 50)
    ids = [s.records[i].record_id for i, _ in sorted(ranks.items(), key=lambda t: t[1])]
    assert ids.index("F") < ids.index("R")


def test_keyword_only_when_vectors_off(tmp_path):
    s = store(tmp_path)
    s.add(rec("A", plan="gradient boosting trees"))
    s.add(rec("B", plan="convolutional network"))
    assert s.search("convolutional", 2)[0].record_id == "B"
    assert (tmp_path / "r.json").exists()


@pytest.mark.slow
def test_real_embedding_model(tmp_path):
    import os
    from pathlib import Path

    from superagent.experience import make_embedder

    local = Path(os.environ.get("SUPERAGENT_PRETRAIN_DIR", "pretrained_cache")).resolve()
    if not (local / "hub" / "models--Qwen--Qwen3-Embedding-0.6B").exists():
        pytest.skip("embedding model not downloaded")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        emb = make_embedder({"enabled": True, "vector": {"enabled": True, "model": "Qwen/Qwen3-Embedding-0.6B", "device": "cuda", "backend": "local",
                                                         "price_per_million": 0.5}}, str(local), tmp_path / "emb.jsonl")
    finally:
        os.environ.pop("HF_HUB_OFFLINE", None)
    assert emb is not None
    s = ExperienceStore({"candidates_per_list": 50, "keyword_weight": 0.0, "rrf_k": 60}, tmp_path / "r.json", emb)
    s.add(rec("A", plan="gradient boosted decision trees with early stopping"))
    s.add(rec("B", plan="convolutional neural network for images with augmentation"))
    got = s.search("image CNN with random flips", 1)
    assert got[0].record_id == "B"
    assert emb.totals["tokens"] > 0 and (tmp_path / "emb.jsonl").exists()


def test_unit_rows():
    import numpy as np

    from superagent.experience import unit_rows

    out = unit_rows(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert out[0].tolist() == pytest.approx([0.6, 0.8]) and out[1].tolist() == [0.0, 0.0]


def test_vector_key_missing():
    from superagent.experience import vector_key_missing

    api = {"enabled": True, "vector": {"enabled": True, "backend": "api", "api_key_env": "SOME_KEY"}}
    assert vector_key_missing(api, {}) == "SOME_KEY"
    assert vector_key_missing(api, {"SOME_KEY": "x"}) is None
    assert vector_key_missing({"enabled": True, "vector": {"enabled": True, "backend": "local"}}, {}) is None
    assert vector_key_missing({"enabled": False, "vector": {"enabled": True, "backend": "api", "api_key_env": "K"}}, {}) is None


def test_api_backend_without_key_falls_back_to_keywords(tmp_path):
    from superagent.experience import make_embedder

    cfg = {"enabled": True, "vector": {"enabled": True, "backend": "api", "api_model": "m", "api_base_url": "http://127.0.0.1:9/v1",
                                       "api_key_env": "SOME_KEY", "api_timeout_seconds": 2, "price_per_million": 0.5}}
    assert make_embedder(cfg, str(tmp_path), None, environ={}) is None


def test_unreachable_embedding_endpoint_degrades_to_keywords(tmp_path):
    """The endpoint address accepts no connections, so every embedding request really fails."""
    from superagent.experience import MAX_EMBED_FAILURES, ApiEmbedder

    emb = ApiEmbedder("m", "http://127.0.0.1:9/v1", "not-a-real-key", 2, 0.5, tmp_path / "emb.jsonl")
    s = ExperienceStore({"candidates_per_list": 50, "keyword_weight": 0.5, "rrf_k": 60}, tmp_path / "r.json", emb)
    assert s.add(rec("A", plan="gradient boosting trees"))
    assert s.add(rec("B", plan="convolutional network"))
    assert s.vector_enabled                       # two failures: still enabled
    assert s.search("convolutional", 1)[0].record_id == "B"   # this lookup used keywords only
    for i in range(MAX_EMBED_FAILURES):
        s.add(rec(f"X{i}", plan="another plan"))
    assert not s.vector_enabled                   # repeated failures switch vector search off
    assert s.search("convolutional", 1)[0].record_id == "B"
    assert not (tmp_path / "emb.jsonl").exists()  # nothing was embedded, nothing was charged


def test_vector_only_weight_still_answers_without_vectors(tmp_path):
    s = ExperienceStore({"candidates_per_list": 50, "keyword_weight": 0.0, "rrf_k": 60}, tmp_path / "r.json", None)
    s.add(rec("A", plan="gradient boosting trees"))
    s.add(rec("B", plan="convolutional network"))
    assert s.search("convolutional", 1)[0].record_id == "B"


@pytest.mark.embed_api
def test_real_embedding_endpoint(tmp_path):
    import os

    from superagent.config import load_defaults
    from superagent.experience import make_embedder

    exp = load_defaults()["experience"]
    exp["vector"]["backend"] = "api"
    emb = make_embedder(exp, str(tmp_path), tmp_path / "emb.jsonl", environ=os.environ)
    assert emb is not None
    s = ExperienceStore({"candidates_per_list": 50, "keyword_weight": 0.0, "rrf_k": 60}, tmp_path / "r.json", emb)
    s.add(rec("A", plan="gradient boosted decision trees with early stopping"))
    s.add(rec("B", plan="convolutional neural network for images with augmentation"))
    assert s.search("image CNN with random flips", 1)[0].record_id == "B"
    assert emb.totals["tokens"] > 0 and (tmp_path / "emb.jsonl").exists()
