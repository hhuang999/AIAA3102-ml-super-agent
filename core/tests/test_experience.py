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
def test_real_embedding_model(tmp_path, monkeypatch):
    from pathlib import Path

    from superagent.config import build_config
    from superagent.experience import make_embedder
    from superagent.utils import read_jsonl

    cfg = build_config(None)
    cache = Path(cfg["pretrained"]["local_dir"])
    if not (cache / "hub" / "models--Qwen--Qwen3-Embedding-0.6B").exists():
        pytest.skip("embedding model not downloaded")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    emb = make_embedder(cfg["experience"], str(cache), tmp_path / "emb.jsonl")
    assert emb is not None
    s = ExperienceStore({"candidates_per_list": 50, "keyword_weight": 0.0, "rrf_k": 60}, tmp_path / "r.json", emb)
    s.add(rec("A", plan="gradient boosted decision trees with early stopping"))
    s.add(rec("B", plan="convolutional neural network for images with augmentation"))
    got = s.search("image CNN with random flips", 1)
    assert got[0].record_id == "B"
    assert emb.totals["tokens"] > 0
    usage = read_jsonl(tmp_path / "emb.jsonl")
    assert {row["backend"] for row in usage} == {"local"}
    assert {row["model"] for row in usage} == {"Qwen/Qwen3-Embedding-0.6B"}
    assert {row["cost"] for row in usage} == {0.0}


def test_missing_local_snapshot_degrades_to_keywords(tmp_path):
    from superagent.config import load_defaults
    from superagent.experience import make_embedder

    assert make_embedder(load_defaults()["experience"], str(tmp_path), None) is None


def test_failing_local_encoder_degrades_to_keywords(tmp_path):
    from superagent.experience import MAX_EMBED_FAILURES

    class UnavailableEncoder:
        def encode(self, texts, purpose):
            raise RuntimeError("local inference failed")

    emb = UnavailableEncoder()
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
