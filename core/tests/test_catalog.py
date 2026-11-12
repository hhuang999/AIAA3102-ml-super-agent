import json
from pathlib import Path

from superagent.catalog import build_catalog, load_catalog, offline_env


def test_catalog_entries_have_english_fields():
    for e in load_catalog():
        for f in ("title", "when_to_use", "input_notes"):
            assert all(ord(ch) < 128 for ch in str(e.get(f) or "")), (e["id"], f)
        assert e["example"]


def test_off_mode_has_no_pretrained(tmp_path):
    st = build_catalog("off", tmp_path, [])
    pretrained_titles = [e["title"] for e in load_catalog() if e["pretrained"]]
    assert all(t not in st.text for t in pretrained_titles)
    assert "FORBIDDEN" in st.text


def test_catalog_mode_uses_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"entries": [
        {"id": "img_resnet50_finetune", "ok": True}, {"id": "img_dinov2_small_features", "ok": False}]}))
    st = build_catalog("catalog", tmp_path, [])
    assert "ResNet-50" in st.text
    assert "DINOv2 small" not in st.text
    assert "DeBERTa" not in st.text


def test_catalog_mode_without_manifest(tmp_path):
    st = build_catalog("catalog", tmp_path, [])
    assert all(not e["pretrained"] for e in st.entries)


def test_open_mode_has_everything(tmp_path):
    st = build_catalog("open", tmp_path, [])
    assert len(st.entries) == len(load_catalog())


def test_modality_filter(tmp_path):
    st = build_catalog("open", tmp_path, ["tabular"])
    assert {e["modality"] for e in st.entries} == {"tabular"}


def test_offline_env():
    assert offline_env("catalog", "/x")["HF_HUB_OFFLINE"] == "1"
    assert offline_env("off", "/x")["HF_HUB_OFFLINE"] == "1"
    assert "HF_HUB_OFFLINE" not in offline_env("open", "/x", "https://mirror")
    assert offline_env("open", "/x", "https://mirror")["HF_ENDPOINT"] == "https://mirror"
