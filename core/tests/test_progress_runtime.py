import copy
import os

import numpy as np
import pytest

from superagent.training import state_fingerprint, verify_continuation


def records():
    before = {
        "epoch": 7,
        "sha256": "a" * 64,
        "progress": {
            "optimizer_steps": 100,
            "model_sha256": "b" * 64,
            "optimizer_sha256": "c" * 64,
        },
    }
    after = {
        "epoch": 9,
        "sha256": "d" * 64,
        "progress": {
            "optimizer_steps": 130,
            "model_sha256": "e" * 64,
            "optimizer_sha256": "f" * 64,
            "restored_from": "a" * 64,
        },
    }
    return before, after


def test_generic_progress_does_not_assume_fixed_epochs():
    a, b = records()
    assert verify_continuation(a, b)["to_steps"] == 130
    assert state_fingerprint([[], []]) != state_fingerprint([[[]]])
    assert state_fingerprint(np.array([1.0, 2.0])) != state_fingerprint(
        np.array([2.0, 1.0])
    )


@pytest.mark.parametrize(
    "fault",
    [
        "same_epoch",
        "same_steps",
        "same_model",
        "same_optimizer",
        "wrong_parent",
        "no_progress",
    ],
)
def test_metadata_bump_is_not_training(fault):
    a, b = records()
    if fault == "same_epoch":
        b["epoch"] = a["epoch"]
    elif fault == "same_steps":
        b["progress"]["optimizer_steps"] = 100
    elif fault == "same_model":
        b["progress"]["model_sha256"] = a["progress"]["model_sha256"]
    elif fault == "same_optimizer":
        b["progress"]["optimizer_sha256"] = a["progress"]["optimizer_sha256"]
    elif fault == "wrong_parent":
        b["progress"]["restored_from"] = "0" * 64
    else:
        b.pop("progress")
    with pytest.raises(ValueError):
        verify_continuation(a, b)


@pytest.mark.parametrize(
    "field,value",
    [
        ("epoch", True),
        ("epoch", 0),
        ("rung", -1),
        ("can_continue", "yes"),
        ("next_seconds", float("nan")),
        ("delivery_seconds", 0),
        ("sha256", None),
    ],
)
def test_checkpoint_rejects_invalid_metadata(tmp_path, monkeypatch, field, value):
    import json
    from superagent.training import save_checkpoint, checkpoint

    for key, val in {
        "CHECKPOINT_DIR": str(tmp_path / "state"),
        "TRIAL_SIGNATURE": "trial",
        "RUNG": "0",
    }.items():
        monkeypatch.setenv("SUPERAGENT_" + key, val)
    payload = tmp_path / "payload"
    payload.write_bytes(b"state")
    save_checkpoint(payload, epoch=1, next_seconds=0.1, delivery_seconds=0.2)
    path = tmp_path / "state/checkpoint.json"
    record = json.loads(path.read_text())
    record[field] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        checkpoint(path.parent, "trial")


@pytest.mark.parametrize("momentum", [0.0, 0.9])
def test_pytorch_actual_updates_include_sgd_without_builtin_step_counter(
    tmp_path, monkeypatch, momentum
):
    torch = pytest.importorskip("torch")
    from superagent.training import save_torch, restore_torch, checkpoint

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SUPERAGENT_TRIAL_SIGNATURE", "trial")
    monkeypatch.setenv("SUPERAGENT_CHECKPOINT_DIR", str(tmp_path / "first"))
    monkeypatch.setenv("SUPERAGENT_RUNG", "0")
    monkeypatch.delenv("SUPERAGENT_RESUME_DIR", raising=False)
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=momentum)
    restore_torch(model, opt)

    def update():
        opt.zero_grad()
        model(torch.ones(2, 2)).sum().backward()
        opt.step()

    for _ in range(3):
        update()
    save_torch(model, opt, epoch=2, next_seconds=0.1, delivery_seconds=0.2)
    first = checkpoint(tmp_path / "first", "trial")
    assert first["progress"]["optimizer_steps"] == 3
    monkeypatch.setenv("SUPERAGENT_RESUME_DIR", str(tmp_path / "first"))
    monkeypatch.setenv("SUPERAGENT_CHECKPOINT_DIR", str(tmp_path / "second"))
    monkeypatch.setenv("SUPERAGENT_RUNG", "1")
    assert restore_torch(model, opt)[0] == 2
    for _ in range(2):
        update()
    save_torch(model, opt, epoch=3, next_seconds=0.1, delivery_seconds=0.2)
    second = checkpoint(tmp_path / "second", "trial")
    assert verify_continuation(first, second)["to_steps"] == 5


@pytest.mark.skipif(
    os.name == "nt", reason="production executor uses POSIX symlinks/process groups"
)
def test_production_runner_rejects_prediction_only_promotion(tmp_path):
    import json
    from helpers import make_ctx, mk
    from superagent.search import SearchController
    from superagent.llm.ledger import fingerprint
    from superagent.training import digest

    ctx = make_ctx(tmp_path)
    code = """import json,os,shutil
from pathlib import Path
dest=Path(os.environ["SUPERAGENT_CHECKPOINT_DIR"]);dest.mkdir()
previous=Path(os.environ["SUPERAGENT_RESUME_DIR"])
meta=json.loads((previous/"checkpoint.json").read_text())
shutil.copyfile(previous/meta["sha256"],dest/meta["sha256"])
meta["epoch"]+=1
meta["rung"]=1
(dest/"checkpoint.json").write_text(json.dumps(meta))
"""
    parent = mk(ctx.tree, "draft", ctx.tree.root, score=0.5, code=code)
    folder = ctx.executor.run_dir(parent.id) / "checkpoints"
    folder.mkdir(parents=True)
    payload = folder / "payload"
    payload.write_bytes(b"same old model")
    sha = digest(payload)
    payload.rename(folder / sha)
    meta = {
        "signature": fingerprint([None, code]),
        "epoch": 3,
        "rung": 0,
        "sha256": sha,
        "can_continue": True,
        "next_seconds": 1,
        "delivery_seconds": 1,
        "progress": {
            "optimizer_steps": 5,
            "model_sha256": "b" * 64,
            "optimizer_sha256": "c" * 64,
        },
    }
    (folder / "checkpoint.json").write_text(json.dumps(meta))
    parent.checkpoint = copy.deepcopy(meta)
    child = mk(ctx.tree, "promote", parent, code=code, rung=1)
    result = SearchController(ctx)._run(child)
    assert result.error_type == "ContinuationError"
    assert not parent.checkpoint["can_continue"]
    assert not child.continuation_verification["passed"]
