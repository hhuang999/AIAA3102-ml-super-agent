"""Download the pretrained models of the method catalog (and the embedding model) for offline use.

    python -m superagent.fetch_models [--config CONFIG] [--only ID ...]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .catalog import MANIFEST_NAME, load_catalog
from .config import build_config
from .utils import atomic_write_json, load_env_file, now_iso

EMBEDDING_ENTRY_ID = "experience_embedding"
IGNORE_PATTERNS = ["*.h5", "*.msgpack", "*.ot", "*.onnx", "flax_model*", "tf_model*", "rust_model*", "*.tflite",
                   "coreml/*", "onnx/*", "openvino/*", "*.gguf"]
WARMUP_TIMEOUT_SECONDS = 1800


def _snapshot(hub_id: str, local_dir: Path, endpoint: str) -> Path:
    from huggingface_hub import list_repo_files, snapshot_download

    ignore = list(IGNORE_PATTERNS)
    try:
        files = list_repo_files(hub_id, endpoint=endpoint or None)
        if any(f.endswith(".safetensors") for f in files):
            ignore.append("*.bin")  # the safetensors copy of the weights is enough
    except Exception:  # noqa: BLE001 - listing is only an optimisation
        pass
    return Path(snapshot_download(repo_id=hub_id, cache_dir=str(local_dir / "hub"), endpoint=endpoint or None, ignore_patterns=ignore))


def _warmup(example: str, local_dir: Path, endpoint: str, offline: bool = False) -> None:
    env = dict(os.environ)
    env.update({"HF_HOME": str(local_dir), "HF_HUB_CACHE": str(local_dir / "hub"), "TORCH_HOME": str(local_dir / "torch")})
    env["HF_HUB_OFFLINE"] = "1" if offline else "0"
    if endpoint:
        env["HF_ENDPOINT"] = endpoint
    with tempfile.TemporaryDirectory() as d:
        script = Path(d) / "warmup.py"
        script.write_text(example, encoding="utf-8")
        r = subprocess.run([sys.executable, str(script)], cwd=d, env=env, capture_output=True, text=True,
                           timeout=WARMUP_TIMEOUT_SECONDS)
        if r.returncode != 0:
            raise RuntimeError((r.stderr or r.stdout)[-1500:])


def fetch(cfg: dict[str, Any], only: list[str] | None = None) -> list[dict[str, Any]]:
    local_dir = Path(cfg["pretrained"]["local_dir"])
    local_dir.mkdir(parents=True, exist_ok=True)
    endpoint = str(cfg["pretrained"].get("hub_endpoint") or "")
    manifest_path = local_dir / MANIFEST_NAME
    previous: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():

        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            previous = {e["id"]: e for e in data.get("entries", [])}
        except (ValueError, KeyError):
            previous = {}
    jobs: list[tuple[str, str | None, str, str | None]] = []
    for e in load_catalog():
        if not e.get("pretrained"):
            continue
        jobs.append((str(e["id"]), e.get("hub_id"), str(e.get("fetch") or "snapshot"), e.get("example")))
    vec = cfg["experience"]["vector"]
    if cfg["experience"]["enabled"] and vec["enabled"]:
        jobs.append((EMBEDDING_ENTRY_ID, str(vec["model"]), "snapshot", None))
    results: list[dict[str, Any]] = []
    for entry_id, hub_id, how, example in jobs:
        if only and entry_id not in only:
            continue
        print(f"[fetch] {entry_id} ({how} {hub_id or ''}) ...", flush=True)
        rec: dict[str, Any] = {"id": entry_id, "hub_id": hub_id, "fetched_at": now_iso(), "ok": False, "error": None}
        try:
            if how == "warmup":
                if not example:
                    raise RuntimeError("warmup entry without example code")
                _warmup(str(example), local_dir, endpoint)
                _warmup(str(example), local_dir, endpoint, offline=True)
                rec.update(state="forward_ok",runner=sys.executable)
            else:
                if not hub_id:
                    raise RuntimeError("snapshot entry without hub_id")
                snapshot = _snapshot(str(hub_id), local_dir, endpoint)
                rec.update(state="downloaded",snapshot=str(snapshot))
                library = "timm" if hub_id.startswith("timm/") else "embedding" if entry_id == EMBEDDING_ENTRY_ID or "SentenceTransformer" in str(example) else "transformers"
                proc = subprocess.run([sys.executable,str(Path(__file__).with_name("capability.py")),hub_id,str(snapshot),library],
                                      text=True,capture_output=True,timeout=WARMUP_TIMEOUT_SECONDS,
                                      env={**os.environ,"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"})
                if proc.returncode:
                    raise RuntimeError(proc.stderr[-500:])
                rec.update(json.loads(proc.stdout.strip().splitlines()[-1]))
            rec["ok"] = True
            print(f"[fetch] {entry_id}: ok", flush=True)
        except Exception as exc:  # noqa: BLE001 - keep going with the other entries
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
            print(f"[fetch] {entry_id}: FAILED {rec['error']}", flush=True)
        previous[entry_id] = rec
        results.append(rec)
        atomic_write_json(manifest_path, {"updated_at": now_iso(), "entries": list(previous.values())})
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m superagent.fetch_models", description=__doc__)
    ap.add_argument("--config")
    ap.add_argument("--only", nargs="+", metavar="ID")
    args = ap.parse_args(argv)
    load_env_file(Path.cwd() / ".env")
    cfg = build_config(args.config)
    results = fetch(cfg, args.only)
    ok = [r["id"] for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    print(f"\nDownloaded {len(ok)} of {len(results)} entries into {cfg['pretrained']['local_dir']}")
    for r in bad:
        print(f"  failed: {r['id']}: {r['error']}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
