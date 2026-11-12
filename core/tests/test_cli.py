import json
from pathlib import Path

import pytest

from superagent.review import policy_scan
from superagent.run import main
from superagent.utils import read_jsonl


def task(tmp_path: Path) -> Path:
    t = tmp_path / "task"
    (t / "data").mkdir(parents=True)
    (t / "description.md").write_text("Predict y. Metric: accuracy.")
    (t / "data" / "train.csv").write_text("x,y\n1,0\n2,1\n")
    return t


def test_nonempty_output_dir(tmp_path, capsys):
    out = tmp_path / "out"
    out.mkdir()
    (out / "old.txt").write_text("keep")
    assert main(["--task-dir", str(task(tmp_path)), "--output-dir", str(out)]) == 3
    assert sorted(p.name for p in out.iterdir()) == ["old.txt"]
    assert "not empty" in capsys.readouterr().err


def test_bad_config(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy")
    cfg = tmp_path / "c.yaml"
    cfg.write_text("search:\n  bogus: 1\n")
    out = tmp_path / "out"
    assert main(["--task-dir", str(task(tmp_path)), "--output-dir", str(out), "--config", str(cfg)]) == 3
    st = json.loads((out / "status.json").read_text())
    assert st["status"] == "invalid_input" and "search.bogus" in st["error"]


def test_missing_description(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy")
    t = task(tmp_path)
    (t / "description.md").unlink()
    out = tmp_path / "out"
    assert main(["--task-dir", str(t), "--output-dir", str(out)]) == 3
    assert json.loads((out / "status.json").read_text())["status"] == "invalid_input"


def test_missing_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # no .env file here
    out = tmp_path / "out"
    assert main(["--task-dir", str(task(tmp_path)), "--output-dir", str(out)]) == 3
    assert "DEEPSEEK_API_KEY" in json.loads((out / "status.json").read_text())["error"]


def test_cpus_below_parallel(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy")
    out = tmp_path / "out"
    assert main(["--task-dir", str(task(tmp_path)), "--output-dir", str(out), "--num-cpus", "1", "--parallel", "2"]) == 3


def test_api_key_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("DEEPSEEK_API_KEY=dummy\n")
    out = tmp_path / "out"
    # The key is found, so the run gets past the key check and stops at the next input check.
    assert main(["--task-dir", str(task(tmp_path)), "--output-dir", str(out), "--num-cpus", "1", "--parallel", "2"]) == 3
    assert "num_cpus" in json.loads((out / "status.json").read_text())["error"]


@pytest.mark.llm
def test_cli_direction_skips_judge(tmp_path):
    """Real endpoint: the time limit is too short to start any script, so only the description is cleaned."""
    cfg = tmp_path / "c.yaml"
    cfg.write_text("run:\n  finalize_reserve_seconds: 100\nexperience:\n  enabled: false\n")
    out = tmp_path / "out"
    code = main(["--task-dir", str(task(tmp_path)), "--output-dir", str(out), "--config", str(cfg), "--time-limit-secs", "120",
                 "--higher-is-better", "false", "--num-gpus", "0", "--num-cpus", "2"])
    assert code == 2
    st = json.loads((out / "status.json").read_text())
    assert st["direction_source"] == "cli" and st["higher_is_better"] is False and st["status"] == "no_valid_solution"
    assert all(r["agent"] != "direction_judge" for r in read_jsonl(out / "logs" / "llm_calls.jsonl"))
    assert "sk-" not in (out / "logs" / "config_used.yaml").read_text()


def test_policy_scan():
    code = "import timm\nm = timm.create_model('resnet18', pretrained=True)\n# m = AutoModel.from_pretrained('x')\n"
    hits = policy_scan(code, ["pretrained=True", "from_pretrained("])
    assert [h[0] for h in hits] == [2]


def test_api_embedding_backend_needs_its_key(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy")
    monkeypatch.delenv("AIHUBMIX_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "c.yaml"
    cfg.write_text("experience:\n  vector:\n    backend: api\n")
    out = tmp_path / "out"
    assert main(["--task-dir", str(task(tmp_path)), "--output-dir", str(out), "--config", str(cfg)]) == 3
    assert "AIHUBMIX_API_KEY" in json.loads((out / "status.json").read_text())["error"]
