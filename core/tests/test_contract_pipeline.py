import json
import sys
from pathlib import Path

import pytest
from superagent.contract import load_spec, records
from superagent.profiler import profile
from superagent.config import load_defaults
from superagent.publication import publish, version

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from flower_regression import create


def test_flower_contract(tmp_path):
    task = tmp_path / "task"
    create(task)
    spec = load_spec(task, task / "data")
    data = profile(task / "data", load_defaults()["profiler"])
    assert "image" in data.modalities
    assert data.preview.count("image: bytes (JPEG)") == 3
    assert data.preview.count("class: int64") == 2
    assert "id: bytes" in data.preview
    out = tmp_path / "out"
    out.mkdir()
    (out / "val_predictions.csv").write_text(
        "id,label\n" + "".join(f"val/00.tfrec:{i},0\n" for i in reversed(range(6)))
    )
    pred = out / "predictions.csv"
    pred.write_text((task / "data/sample_submission.csv").read_text())
    assert spec.grade(out) == pytest.approx(1 / 3)
    pred.write_text(pred.read_text().replace("f1,0", "f1,104"))
    with pytest.raises(ValueError, match="domain"):
        spec.grade(out)
    pred.write_text(pred.read_text().replace("f1,104", "f0,0"))
    with pytest.raises(ValueError, match="IDs"):
        spec.grade(out)
    shard = task / "data/val/00.tfrec"
    shard.write_bytes(shard.read_bytes()[:-1])
    with pytest.raises(ValueError, match="CRC"):
        list(records(shard))


def test_partial_publish_preserves_previous(tmp_path):
    v = version(tmp_path)
    (v / "predictions.csv").write_text("id,label\nx,1")
    (v / "source.json").write_text(json.dumps({"available": True}))
    publish(tmp_path, v)
    with pytest.raises(ValueError, match="incomplete"):
        publish(tmp_path, version(tmp_path))
    assert (tmp_path / "final/predictions.csv").read_text().endswith("x,1")


def test_missing_incumbent_validation_does_not_reject_valid_child(tmp_path):
    from helpers import make_ctx, mk
    from superagent.assessment import assess
    from superagent.executor import ExecResult

    task = tmp_path / "task"
    create(task)
    ctx = make_ctx(tmp_path)
    ctx.task_spec = load_spec(task, task / "data")
    best = mk(ctx.tree, "draft", ctx.tree.root, score=1 / 3)
    ctx.tree.best = best
    child = mk(ctx.tree, "refine", best, failed=None)
    out = ctx.executor.run_dir(child.id) / "out"
    out.mkdir(parents=True)
    (out / "predictions.csv").write_text((task / "data/sample_submission.csv").read_text())
    (out / "val_predictions.csv").write_text(
        "id,label\n" + "".join(f"val/00.tfrec:{i},0\n" for i in range(6))
    )
    result = assess(ctx, child, ExecResult("trained", 1, None, None, False, 0, 10))
    assert child.ok and child.score == pytest.approx(1 / 3)
    assert "1 of 104 declared classes" in child.output
    assert "6/6 validation predictions" in child.output
    assert result["comparison_unavailable"]
    assert (out / "val_labels.csv").is_file()
