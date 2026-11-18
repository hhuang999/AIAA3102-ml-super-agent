import os
import zipfile
from pathlib import Path

import pytest

from superagent.errors import InvalidInputError
from superagent.task_io import check_output_dir, check_task_dir, prepare_data


def make_task(tmp_path: Path) -> Path:
    t = tmp_path / "task"
    (t / "data").mkdir(parents=True)
    (t / "description.md").write_text("Classify things.")
    (t / "data" / "train.csv").write_text("a,b\n1,2\n")
    return t


def test_output_dir_nonempty(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "x").write_text("old")
    with pytest.raises(InvalidInputError) as ei:
        check_output_dir(out)
    assert ei.value.write_status is False


def test_task_dir_checks(tmp_path):
    t = make_task(tmp_path)
    check_task_dir(t)
    (t / "description.md").write_text("  \n")
    with pytest.raises(InvalidInputError, match="empty"):
        check_task_dir(t)
    with pytest.raises(InvalidInputError):
        check_task_dir(tmp_path / "missing")


def test_zip_single_same_name_dir_is_lifted(tmp_path):
    t = make_task(tmp_path)
    zp = t / "data" / "images.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr("images/a.png", b"x")
        zf.writestr("images/sub/b.png", b"y")
        zf.writestr("__MACOSX/images/._a.png", b"junk")
        zf.writestr("images/.DS_Store", b"junk")
    before = zp.read_bytes()
    prep = prepare_data(t / "data", tmp_path / "prepared")
    assert (prep / "images" / "a.png").is_file()
    assert (prep / "images" / "sub" / "b.png").is_file()
    assert not (prep / "images" / "images").exists()
    assert not (prep / "images" / "__MACOSX").exists()
    assert not (prep / "images" / ".DS_Store").exists()
    assert (prep / "train.csv").is_symlink()
    assert zp.read_bytes() == before
    assert not os.access(prep / "images" / "a.png", os.W_OK) or os.geteuid() == 0


def test_zip_multiple_top_entries_kept(tmp_path):
    t = make_task(tmp_path)
    with zipfile.ZipFile(t / "data" / "pack.zip", "w") as zf:
        zf.writestr("x.txt", "1")
        zf.writestr("y/z.txt", "2")
    prep = prepare_data(t / "data", tmp_path / "prepared")
    assert (prep / "pack" / "x.txt").is_file()
    assert (prep / "pack" / "y" / "z.txt").is_file()
