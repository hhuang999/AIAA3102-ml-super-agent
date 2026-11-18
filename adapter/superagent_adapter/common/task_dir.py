"""Build a Core task directory from the public part of a split (description, sample submission, data files),
and check that nothing from the private part (the answers) reached it."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

SAMPLE_NAME = "sample_predictions.csv"
DESCRIPTION_NAME = "description.md"
SAMPLE_SUBMISSION_NAME = "sample_submission.csv"


@dataclass
class IsolationReport:
    ok: bool
    problems: list[str] = field(default_factory=list)
    files_checked: int = 0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_task_dir(public_dir: Path, task_dir: Path, *, contract: Path | None = None) -> Path:
    """Copy the public competition files into ``task_dir`` in the layout the Core expects.

    - ``description.md`` comes from the public MLE-bench description, unchanged.
    - ``data/`` holds every other public file.
    - ``data/sample_predictions.csv`` is the public sample submission, renamed.
    """
    public_dir = Path(public_dir)
    task_dir = Path(task_dir)
    if task_dir.exists() and any(task_dir.iterdir()):
        raise FileExistsError(f"task dir is not empty: {task_dir}")
    desc = public_dir / DESCRIPTION_NAME
    sample = public_dir / SAMPLE_SUBMISSION_NAME
    if not desc.is_file() or not sample.is_file():
        raise FileNotFoundError(f"{public_dir} must contain {DESCRIPTION_NAME} and {SAMPLE_SUBMISSION_NAME}")
    data = task_dir / "data"
    data.mkdir(parents=True)
    shutil.copyfile(desc, task_dir / DESCRIPTION_NAME)
    shutil.copyfile(sample, data / SAMPLE_NAME)
    for entry in sorted(public_dir.iterdir()):
        if entry.name in (DESCRIPTION_NAME, SAMPLE_SUBMISSION_NAME):
            continue
        if entry.is_dir():
            shutil.copytree(entry, data / entry.name)
        else:
            shutil.copyfile(entry, data / entry.name)
    if contract is not None:
        import yaml
        spec=yaml.safe_load(Path(contract).read_text(encoding="utf-8"))
        if spec.get("sample")==SAMPLE_SUBMISSION_NAME:
            spec["sample"]=SAMPLE_NAME
        if isinstance(spec.get("inference_inputs"), list):
            spec["inference_inputs"] = [
                SAMPLE_NAME if path == SAMPLE_SUBMISSION_NAME else path
                for path in spec["inference_inputs"]
            ]
        (task_dir/"task.yaml").write_text(yaml.safe_dump(spec,sort_keys=False),encoding="utf-8")
    return task_dir


def check_isolation(task_dir: Path, private_dir: Path, test_file: str = "test.csv") -> IsolationReport:
    """Verify that nothing from the private (answer) part reached the task directory.

    Checks:
    1. No file in the task dir has the same content as any private file.
    2. No file in the task dir has the same relative name as a private file, unless it is the
       public test file and its content differs (the public test file shares the name ``test.csv``).
    3. The public test file has none of the private answer columns other than the id column.
    4. No CSV in the task dir contains the private answers table's columns AND its ids together.
    """
    task_dir = Path(task_dir)
    private_dir = Path(private_dir)
    problems: list[str] = []
    private_files = [p for p in private_dir.rglob("*") if p.is_file()]
    if not private_files:
        return IsolationReport(False, [f"no private files to check in {private_dir}"])
    private_hashes = {_sha256(p): p for p in private_files}
    task_files = [p for p in task_dir.rglob("*") if p.is_file()]
    for f in task_files:
        h = _sha256(f)
        if h in private_hashes:
            problems.append(f"{f} has the same content as private file {private_hashes[h]}")
    private_names = {p.relative_to(private_dir).as_posix() for p in private_files}
    for f in task_files:
        rel = f.relative_to(task_dir / "data").as_posix() if (task_dir / "data") in f.parents else f.name
        if rel in private_names and rel != test_file:
            problems.append(f"{f} has the same name as a private file")
    answer_cols: list[str] = []
    answer_ids: set[str] = set()
    for p in private_files:
        if p.suffix.lower() == ".csv":
            df = pd.read_csv(p, dtype=str)
            answer_cols += [c for c in df.columns[1:]]
            answer_ids |= set(df.iloc[:, 0].astype(str))
    test_path = task_dir / "data" / test_file
    if test_path.is_file():
        tcols = list(pd.read_csv(test_path, nrows=5).columns)
        leaked = [c for c in tcols[1:] if c in answer_cols]
        if leaked:
            problems.append(f"public test file has answer columns {leaked}")
    for f in task_files:
        if f.suffix.lower() != ".csv" or f.name == SAMPLE_NAME:
            continue
        df = pd.read_csv(f, dtype=str)
        cols = set(df.columns)
        if answer_cols and set(answer_cols) <= cols and answer_ids and answer_ids <= set(df.iloc[:, 0].astype(str)):
            problems.append(f"{f} contains the answer columns for the test ids")
    return IsolationReport(ok=not problems, problems=problems, files_checked=len(task_files))
