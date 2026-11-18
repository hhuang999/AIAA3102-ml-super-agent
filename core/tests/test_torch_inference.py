"""Exercise the production inference wrapper with a real PyTorch payload."""

import os
import subprocess
import sys

import pytest

from superagent.artifacts import WRAPPER


@pytest.mark.parametrize("action", ["predict", "optimizer", "backward"])
def test_real_torch_reload_and_inference_only_guard(tmp_path, action):
    torch = pytest.importorskip("torch")
    torch.save(torch.nn.Linear(2, 1).state_dict(), tmp_path / "weights.pt")
    code = """import torch
model = torch.nn.Linear(2, 1)
model.load_state_dict(torch.load('weights.pt', weights_only=True))
model.eval()
output = model(torch.ones(1, 2))
ACTION
print('PREDICTED', output.detach().tolist())
"""
    operations = {
        "predict": "",
        "optimizer": "torch.optim.SGD(model.parameters(), lr=.1)",
        "backward": "output.sum().backward()",
    }
    (tmp_path / "solution.py").write_text(code.replace("ACTION", operations[action]))
    (tmp_path / "verify_inference.py").write_text(WRAPPER)

    def run():
        return subprocess.run(
            [sys.executable, "verify_inference.py"],
            cwd=tmp_path,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1"),
            capture_output=True,
            text=True,
            timeout=30,
        )

    result = run()
    if action == "predict":
        assert result.returncode == 0, result.stderr
        assert "PREDICTED" in result.stdout
        (tmp_path / "weights.pt").write_bytes(b"corrupt weights")
        broken = run()
        assert broken.returncode != 0 and "PREDICTED" not in broken.stdout
    else:
        assert result.returncode != 0
        assert "inference-only verification attempted training" in result.stderr
