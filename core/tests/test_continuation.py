import pytest
from superagent.executor import Executor
from superagent.training import checkpoint, verify_continuation
from test_contract_pipeline import create
from superagent.contract import load_spec


def test_two_real_subprocess_rungs_and_stale_checkpoint(tmp_path):
    task = tmp_path / "task"
    create(task)
    spec = load_spec(task, task / "data")
    ex = Executor(tmp_path / "runs", task / "data", 1, 1, 0, {})
    code = (task / "baseline.py").read_text()
    previous = None
    previous_record = None
    for i in range(2):
        directory = ex.run_dir(str(i)) / "checkpoints"
        env = dict(
            SUPERAGENT_TRIAL_SIGNATURE=spec.signature,
            SUPERAGENT_CHECKPOINT_DIR=str(directory),
            SUPERAGENT_RUNG=str(i),
            SUPERAGENT_DELIVERY_SECONDS="2",
        )
        if previous:
            env["SUPERAGENT_RESUME_DIR"] = str(previous)
        result = ex.run(str(i), code, 30, trial_env=env)
        assert result.exit_code == 0, result.output
        assert 0 <= spec.grade(ex.run_dir(str(i)) / "out") <= 1
        info = checkpoint(directory, spec.signature)
        assert info["epoch"] == i + 1 and info["rung"] == i
        if previous_record:
            assert verify_continuation(previous_record, info)["passed"]
        previous_record = info
        previous = directory
    with pytest.raises(ValueError, match="does not match"):
        checkpoint(previous, "changed")
    (previous / info["sha256"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="changed"):
        checkpoint(previous, spec.signature)
