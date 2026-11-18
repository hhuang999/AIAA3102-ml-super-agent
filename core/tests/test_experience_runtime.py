import threading
import numpy as np

from superagent.experience import ExperienceStore
from superagent.knowledge import read_cards
from test_experience import rec


def test_filter_before_truncation_and_no_unrelated(tmp_path):
    store = ExperienceStore({"candidates_per_list": 2}, tmp_path / "r.json", None)
    for i in range(55):
        store.add(rec(str(i), effect=-1, plan="decoder decoder jpeg"))
    store.add(rec("wanted", plan="decoder alternative"))
    assert [r.record_id for r in store.search("decoder", 1, effect=1)] == ["wanted"]
    assert not store.search("entirely-unseen", 2)
    rebuilt = ExperienceStore({}, store.path, None)
    rebuilt.rebuild()
    assert rebuilt.search("decoder", 1, effect=1)[0].record_id == "wanted"


def test_one_query_embedding_for_two_polarities(tmp_path):
    class Embed:
        def __init__(self):
            self.calls = []

        def encode(self, texts, purpose):
            self.calls.append(purpose)
            return np.array([[1.0, 0.0]])

    emb = Embed()
    store = ExperienceStore({}, tmp_path / "r.json", emb)
    store.add(rec("a", effect=1, plan="cnn"))
    store.add(rec("b", effect=-1, plan="cnn"))
    assert all(store.search("image", 1, effect=e) for e in (1, -1))
    assert emb.calls == ["index", "index", "query"]


def test_blocked_encoder_leaves_records_searchable(tmp_path):
    started, release = threading.Event(), threading.Event()

    class Embed:
        def encode(self, texts, purpose):
            started.set()
            assert release.wait(3)
            return np.ones((1, 1))

    store = ExperienceStore({}, tmp_path / "r.json", Embed(), background=True)
    try:
        store.add(rec("a", plan="cnn"))
        assert started.wait(1)
        assert store.search("cnn", 1)[0].record_id == "a"
        store.remove("a")
    finally:
        store.close()
        release.set()
        store._thread.join(2)
    assert not store._vectors


def test_versioned_cards_filtered_by_modality(tmp_path):
    import yaml
    import pytest
    from superagent.errors import InvalidInputError

    card = dict(
        id="freeze",
        version="1",
        source="experiment-42",
        method="cache frozen features",
        counterexample="do not cache across stochastic augmentations",
        modality="image",
    )
    (tmp_path / "knowledge.yaml").write_text(yaml.safe_dump([card]))
    assert "freeze@1" in read_cards(tmp_path, ["image"])
    assert not read_cards(tmp_path, ["tabular"])
    del card["source"]
    (tmp_path / "knowledge.yaml").write_text(yaml.safe_dump([card]))
    with pytest.raises(InvalidInputError):
        read_cards(tmp_path, ["image"])
