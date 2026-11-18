import hashlib
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import yaml
from superagent.catalog import select_entries
from superagent.contract import load_spec
from superagent.publication import incumbent, latest


def test_weight_changed_after_forward_not_offered(tmp_path):
    w = tmp_path / "checkpoint.safetensors"
    w.write_bytes(b"verified")
    entry = dict(
        id="backbone",
        pretrained=True,
        modality="image",
        library="timm",
        example="net(pretrained=True)",
    )
    manifest = {
        "backbone": dict(
            state="forward_ok",
            ok=True,
            runner=sys.executable,
            snapshot=str(tmp_path),
            weights={w.name: hashlib.sha256(w.read_bytes()).hexdigest()},
        )
    }
    assert select_entries([entry], "catalog", manifest, ["image"])[0]
    w.write_bytes(b"modified")
    assert select_entries([entry], "catalog", manifest, ["image"])[1] == ["backbone"]


def test_csv_probability_contract_uses_trusted_labels(tmp_path):
    (tmp_path / "sample.csv").write_text("id,no,yes\nt0,1,0\nt1,0,1\n")
    (tmp_path / "truth.csv").write_text("id,y\na,no\nb,yes\n")
    (tmp_path / "task.yaml").write_text(
        yaml.safe_dump(
            dict(
                version=1,
                metric="log_loss",
                classes=["no", "yes"],
                encoding="probability",
                sample="sample.csv",
                validation=dict(format="csv", path="truth.csv", id="id", label="y"),
            )
        )
    )
    spec = load_spec(tmp_path, tmp_path)
    (tmp_path / "predictions.csv").write_bytes((tmp_path / "sample.csv").read_bytes())
    (tmp_path / "val_predictions.csv").write_text("id,no,yes\nb,.25,.75\na,.75,.25\n")
    (tmp_path / "val_labels.csv").write_text("invented score 1")
    import math

    assert spec.grade(tmp_path) == pytest.approx(-math.log(0.75))
    (tmp_path / "predictions.csv").write_text("id,no,yes\nt1,1,0\nt0,0,1\n")
    with pytest.raises(ValueError, match="order"):
        spec.grade(tmp_path)


@pytest.mark.parametrize("fail", ["copyfile", "metadata", "replace"])
def test_interrupted_incumbent_keeps_last_submission(tmp_path, monkeypatch, fail):
    import superagent.publication as pub

    def fail_here(*a, **kw):
        raise OSError("simulated interruption")

    ctx = SimpleNamespace(
        paths=SimpleNamespace(root=tmp_path),
        executor=SimpleNamespace(run_dir=lambda _: tmp_path),
    )
    n = SimpleNamespace(id="n1", score=0.8, score_source="task_contract")
    (tmp_path / "out").mkdir()
    (tmp_path / "out/predictions.csv").write_text("id,label\nt,1")
    incumbent(ctx, n)
    (tmp_path / "out/predictions.csv").write_text("id,label\nt,0")
    n.id = "n2"
    if fail == "copyfile":
        monkeypatch.setattr(pub.shutil, "copyfile", fail_here)
    elif fail == "metadata":
        monkeypatch.setattr(pub, "atomic_write_json", fail_here)
    else:
        replace = pub.os.replace

        def stop_at_pointer(src, dest):
            if Path(dest).name == "final":
                fail_here()
            return replace(src, dest)

        monkeypatch.setattr(pub.os, "replace", stop_at_pointer)
    with pytest.raises(OSError):
        incumbent(ctx, n)
    assert latest(tmp_path)["node_id"] == "n1"
    assert (tmp_path / "final/predictions.csv").read_text().endswith("t,1")


def test_running_status_exposes_published_incumbent(tmp_path):
    from helpers import make_ctx, mk
    from superagent.bookkeeping import status_payload

    ctx = make_ctx(tmp_path)
    ctx.task_spec = SimpleNamespace(metric="accuracy")
    n = mk(ctx.tree, "draft", ctx.tree.root, score=.8)
    d = ctx.executor.run_dir(n.id) / "out"
    d.mkdir(parents=True)
    (d / "predictions.csv").write_text("id,label\nt,1")
    incumbent(ctx, n)
    status = status_payload(ctx, "running")
    assert status["final_available"] and status["final_kind"] == "single"
    assert status["final_metric"] == "accuracy" and status["final_validation_score"] == .8
