"""Detection of the CPUs and GPUs available to this process."""

from __future__ import annotations

import os
import shutil
import subprocess

from .state import GpuInfo

NVIDIA_SMI_TIMEOUT_SECONDS = 15


def available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux
        return os.cpu_count() or 1


def parse_nvidia_smi(text: str) -> GpuInfo:
    """Parse ``nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits`` output."""
    names: list[str] = []
    mems: list[float] = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            mem = float(parts[-1])
        except ValueError:
            continue
        names.append(",".join(parts[:-1]))
        mems.append(mem)
    return GpuInfo(count=len(mems), memory_mb=tuple(mems), names=tuple(names))


def detect_gpus() -> GpuInfo:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return GpuInfo(count=0)
    try:
        res = subprocess.run([exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=NVIDIA_SMI_TIMEOUT_SECONDS, check=False)
    except (OSError, subprocess.SubprocessError):
        return GpuInfo(count=0)
    if res.returncode != 0:
        return GpuInfo(count=0)
    return parse_nvidia_smi(res.stdout)
