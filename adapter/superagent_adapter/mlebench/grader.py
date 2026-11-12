"""Grade Core outputs with the official MLE-bench grader (``mlebench grade-sample``)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ..common.submission import candidate_files, make_submission, primary_name

# The grader logs "Competition report:" and then the JSON, possibly on a new line with a log prefix
# such as "[2026-11-12 04:21:12,571] [cli.py:203] {".
REPORT_RE = re.compile(r"Competition report:.*?(\{\s*\n.*\})", re.DOTALL)


def parse_report(text: str) -> dict[str, Any] | None:
    m = REPORT_RE.search(text)
    if not m:
        return None
    body = m.group(1)
    return json.loads(body[: body.rfind("}") + 1])


def grade(mlebench_bin: str, submission: Path, competition: str, data_dir: Path, raw_log: Path) -> dict[str, Any]:
    """Run the official grader; keep its complete output in ``raw_log``; return the parsed report."""
    proc = subprocess.run([mlebench_bin, "grade-sample", str(submission), competition, "--data-dir", str(data_dir)],
                          capture_output=True, text=True)
    text = proc.stdout + proc.stderr
    raw_log.parent.mkdir(parents=True, exist_ok=True)
    raw_log.write_text(text, encoding="utf-8")
    report = parse_report(text)
    if report is None:
        raise RuntimeError(f"grader output has no report (exit {proc.returncode}); see {raw_log}")
    return report


def grade_all(core_out: Path, sample_csv: Path, mlebench_bin: str, competition: str, data_dir: Path,
              results_dir: Path, include_top: bool = False) -> dict[str, Any]:
    """Grade the primary result and every comparison file; write ``grades.json`` and ``submission.csv``."""
    primary = primary_name(core_out)
    results: dict[str, Any] = {"primary": primary, "competition": competition, "grades": {}}
    for name, pred in candidate_files(core_out, include_top).items():
        sub = make_submission(pred, sample_csv, results_dir / "submissions" / f"{name}.csv")
        results["grades"][name] = grade(mlebench_bin, sub, competition, data_dir, results_dir / "grader_raw" / f"{name}.log")
    (results_dir / "grades.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    shutil.copyfile(results_dir / "submissions" / f"{primary}.csv", results_dir / "submission.csv")
    return results
