import json

import pytest

from superagent.contract import load_spec
from superagent.executor import Executor
from test_contract_pipeline import create


def execute(tmp_path, change=lambda c: c):
    task = tmp_path / "task"
    create(task)
    spec = load_spec(task, task / "data")
    ex = Executor(tmp_path / "runs", task / "data", 1, 1, 0, {})
    code = change((task / "baseline.py").read_text())
    result = ex.run(
        "node",
        code,
        30,
        inference_inputs=spec.inference_inputs,
        trial_env={
            "SUPERAGENT_CHECKPOINT_DIR": str(ex.run_dir("node") / "checkpoints"),
            "SUPERAGENT_TRIAL_SIGNATURE": spec.signature,
            "SUPERAGENT_RUNG": "0",
        },
    )
    return ex, result, spec


def test_real_independent_reload_and_corrupt_weight_rejection(tmp_path):
    ex, result, spec = execute(tmp_path)
    assert result.error_type is None, result.output
    proof = result.artifact_verification
    assert proof["passed"] and proof["corrupt_weights_rejected"]
    assert not (ex.run_dir("node") / "work").exists()
    assert 0 <= spec.grade(ex.run_dir("node") / "out") <= 1
    stored = json.loads((ex.run_dir("node") / "artifact_verification.json").read_text())
    assert stored == proof


@pytest.mark.parametrize("kind", ["temporary_only", "wrong_reload", "ignores_weights"])
def test_invalid_package_never_passes_execution_gate(tmp_path, kind):
    def change(code):
        if kind == "temporary_only":
            return code.replace('Path("out/model")', 'Path("work/model")').replace(
                "model.mkdir(exist_ok=True)", "model.mkdir(exist_ok=True, parents=True)"
            )
        if kind == "wrong_reload":
            return code.replace(
                'predict(state["weights"], args.data_dir, args.output_dir)',
                'predict(state["weights"] * 0, args.data_dir, args.output_dir)',
            )
        return code.replace(
            'with np.load(args.model_dir / "weights.npz", allow_pickle=False) as state:',
            "if True:",
        ).replace(
            'predict(state["weights"], args.data_dir, args.output_dir)',
            "fake = np.zeros((4,104)); fake[-1,2] = 1; predict(fake,args.data_dir,args.output_dir)",
        )

    _, result, _ = execute(tmp_path, change)
    assert result.error_type == "ArtifactError"
    assert not result.artifact_verification["passed"]
    expected = {
        "temporary_only": "package is missing",
        "wrong_reload": "differ",
        "ignores_weights": "accepted corrupt",
    }
    assert expected[kind] in result.error_message
