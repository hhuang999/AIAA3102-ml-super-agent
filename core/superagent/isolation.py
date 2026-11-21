"""Linux execution boundary for task scripts and standalone model reload."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .errors import InvalidInputError


_NVIDIA_CONTROL_DEVICES = (
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
        self.num_gpus = max(0, int(num_gpus))
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
        if self.num_gpus > 0:
            for extra in ("/usr/local/nvidia", "/usr/local/cuda"):
                ep = Path(extra)
                if not ep.exists():
                    continue
                if ep.is_symlink():
                    real = ep.resolve().as_posix()
                    self.runtime.extend(("--ro-bind", real, real))
                else:
                    self.runtime.extend(("--ro-bind", ep.as_posix(), extra))
        for path in ("/etc/ld.so.cache", "/etc/localtime"):
            if Path(path).is_file():
                self.runtime.extend(("--ro-bind", path, path))
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
        for name in (
            "SUPERAGENT_SMOKE", "SUPERAGENT_TRAIN_UNTIL", "SUPERAGENT_TRIAL_SIGNATURE",
            "SUPERAGENT_RUNG", "SUPERAGENT_DELIVERY_SECONDS", "SUPERAGENT_FEATURE_NAMESPACE",
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "TORCH_NUM_THREADS",
        ):
            if name in source_env:
                env[name] = source_env[name]
        if self.num_gpus > 0:
            cuda_devs = source_env.get("CUDA_VISIBLE_DEVICES", "")
            if cuda_devs:
                env["CUDA_VISIBLE_DEVICES"] = cuda_devs
                if "NVIDIA_VISIBLE_DEVICES" in source_env:
                    env["NVIDIA_VISIBLE_DEVICES"] = source_env["NVIDIA_VISIBLE_DEVICES"]
                for ctrl in _NVIDIA_CONTROL_DEVICES:
                    if Path(ctrl).exists():
                        command.extend(("--dev-bind", ctrl, ctrl))
                tokens = [t.strip() for t in cuda_devs.split(",") if t.strip()]
                if tokens and all(t.isdecimal() for t in tokens):
                    for t in tokens:
                        dev = Path(f"/dev/nvidia{t}")
                        if dev.exists():
                            command.extend(("--dev-bind", dev.as_posix(), dev.as_posix()))
                elif tokens:
                    for dev in sorted(Path("/dev").glob("nvidia[0-9]*")):
                        command.extend(("--dev-bind", dev.as_posix(), dev.as_posix()))
            elif "CUDA_VISIBLE_DEVICES" in source_env:
                env["CUDA_VISIBLE_DEVICES"] = source_env["CUDA_VISIBLE_DEVICES"]
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
