"""Linux execution boundary for task scripts and standalone model reload."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .errors import InvalidInputError

_NVIDIA_CTRL_DEVICES = (
    "/dev/nvidiactl",
    "/dev/nvidia-uvm",
    "/dev/nvidia-uvm-tools",
    "/dev/nvidia-modeset",
    "/dev/nvidia-caps",
)


class Bubblewrap:
    def __init__(self, training: Path, inference: Path, forbidden: Path, *, assets=None, num_gpus: int = 0):
        self.training = training
        self.inference = inference
        self.assets = assets or {}
        self.num_gpus = int(num_gpus or 0)
        self.binary = shutil.which("bwrap")
        if sys.platform != "linux" or not self.binary:
            raise InvalidInputError("verification_environment_unavailable: install Linux bubblewrap")
        self.runtime = [
            self.binary, "--unshare-all", "--die-with-parent", "--cap-drop", "ALL",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--dir", "/home/agent",
        ]
        roots = [Path(p) for p in ("/usr", "/bin", "/lib", "/lib64", "/sbin") if Path(p).exists()]
        prefixes = sorted({Path(sys.prefix), Path(sys.base_prefix)}, key=lambda p: len(p.parts))
        for prefix in prefixes:
            if prefix == Path("/") or prefix == Path.home():
                raise InvalidInputError("verification_environment_unavailable: put Python in a dedicated environment")
            if not any(prefix.is_relative_to(root) for root in roots):
                roots.append(prefix)
        for root in roots:
            if root.is_symlink():
                self.runtime.extend(("--symlink", os.readlink(root), str(root)))
            else:
                self.runtime.extend(("--ro-bind", str(root), str(root)))
        for path in ("/etc/ld.so.cache", "/etc/localtime"):
            if Path(path).is_file():
                self.runtime.extend(("--ro-bind", path, path))
        if self.num_gpus > 0:
            for base in ("/usr/local/nvidia", "/usr/local/cuda"):
                p = Path(base)
                if not p.exists():
                    continue
                resolved = p.resolve()
                if p.is_symlink():
                    if not any(resolved.is_relative_to(r) for r in roots):
                        self.runtime.extend(("--ro-bind", str(resolved), str(resolved)))
                    self.runtime.extend(("--symlink", os.readlink(p), str(p)))
                elif not any(p.is_relative_to(r) for r in roots):
                    self.runtime.extend(("--ro-bind", str(p), str(p)))
            for dev in _NVIDIA_CTRL_DEVICES:
                if Path(dev).exists():
                    self.runtime.extend(("--dev-bind", dev, dev))
            cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
            gpu_nodes: list[Path] = []
            if cuda_vis and all(tok.strip().isdigit() for tok in cuda_vis.split(",") if tok.strip()):
                for tok in cuda_vis.split(","):
                    t = tok.strip()
                    if t:
                        cand = Path(f"/dev/nvidia{t}")
                        if cand.exists():
                            gpu_nodes.append(cand)
            else:
                gpu_nodes = sorted(Path("/dev").glob("nvidia[0-9]*"))
            for node in dict.fromkeys(gpu_nodes):
                if node.exists():
                    self.runtime.extend(("--dev-bind", str(node), str(node)))
        package = Path(__file__).resolve().parent
        self.runtime.extend(("--ro-bind", str(package), "/runtime/superagent", "--chdir", "/job"))
        self._preflight(forbidden)

    def _preflight(self, forbidden):
        import tempfile

        with tempfile.TemporaryDirectory(prefix="superagent-sandbox-") as temp:
            job = Path(temp)
            (job / "out").mkdir()
            (job / "work").mkdir()
            (job / "check.py").write_text(
                "import os, pathlib, sys\n"
                "assert pathlib.Path('data/input_manifest.json').is_file()\n"
                "assert not pathlib.Path(sys.argv[1]).exists()\n"
                "assert not any('DEEPSEEK' in k.upper() for k in os.environ)\n"
                "pathlib.Path('out/ready').touch()\n",
                encoding="utf-8",
            )
            try:
                command, env = self.command(job, ["check.py", str(forbidden)], {}, inference=True)
                result = subprocess.run(command, env=env, capture_output=True, timeout=10, text=True)
                if result.returncode or not (job / "out/ready").exists():
                    raise ValueError(result.stderr.strip() or "sandbox input/output preflight failed")
            except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                raise InvalidInputError(f"verification_environment_unavailable: {exc}") from exc

    def command(self, job, argv, source_env, *, inference):
        command = self.runtime.copy()
        command.extend(("--ro-bind", str(self.inference if inference else self.training), "/job/data"))
        if not inference:
            for target, source in self.assets.items():
                command.extend(("--ro-bind", str(source), target))
        for name in ("solution.py", "verify_inference.py", "check.py", "model"):
            source = job / name
            if source.exists():
                command.extend(("--ro-bind", str(source), f"/job/{name}"))
        for name in ("out", "work"):
            if (job / name).is_dir():
                command.extend(("--bind", str(job / name), f"/job/{name}"))
        env = {
            "PATH": "/usr/bin:/bin", "HOME": "/home/agent", "LANG": "C.UTF-8",
            "PYTHONPATH": "/runtime", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        }
        pass_vars = [
            "SUPERAGENT_SMOKE", "SUPERAGENT_TRAIN_UNTIL", "SUPERAGENT_TRIAL_SIGNATURE",
            "SUPERAGENT_RUNG", "SUPERAGENT_DELIVERY_SECONDS", "SUPERAGENT_FEATURE_NAMESPACE",
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
        ]
        if self.num_gpus > 0:
            pass_vars.extend(("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "VECLIB_MAXIMUM_THREADS", "TORCH_NUM_THREADS"))
        for name in pass_vars:
            if name in source_env:
                env[name] = source_env[name]
        for name, mount, writable in (
            ("SUPERAGENT_CHECKPOINT_DIR", "/job/checkpoints", True),
            ("SUPERAGENT_RESUME_DIR", "/job/resume", False),
            ("SUPERAGENT_FEATURE_DIR", "/job/features", True),
        ):
            if name in source_env and not inference:
                path = Path(source_env[name])
                if writable:
                    path.mkdir(parents=True, exist_ok=True)
                command.extend(("--bind" if writable else "--ro-bind", str(path), mount))
                env[name] = mount
        command.extend(("--", sys.executable, *argv))
        return command, env
