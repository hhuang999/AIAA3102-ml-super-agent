"""Detection of the CPUs and GPUs available to this process."""

from __future__ import annotations

import os
import shutil
import subprocess

from .state import GpuInfo

NVIDIA_SMI_TIMEOUT_SECONDS = 15


def cuda_mask() -> tuple[str, ...] | None:
    """The scheduler's ordered CUDA allow-list, or None when unset."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    devices = []
    for token in raw.split(","):
        item = token.strip()
        if not (item.isdecimal() or item.startswith(("GPU-", "MIG-"))):
            break
        devices.append(item)
    return tuple(devices)


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
    mask = cuda_mask()
    if mask == ():
        return GpuInfo(count=0)
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return GpuInfo(count=0)
    fields = "index,uuid,name,memory.total" if mask else "name,memory.total"
    try:
        res = subprocess.run([exe, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=NVIDIA_SMI_TIMEOUT_SECONDS, check=False)
    except (OSError, subprocess.SubprocessError):
        return GpuInfo(count=0)
    if res.returncode != 0:
        return GpuInfo(count=0)
    if mask is None:
        return parse_nvidia_smi(res.stdout)
    rows = [row.split(",", 2) for row in res.stdout.splitlines() if row.count(",") >= 2]
    matched = []
    for device in mask:
        candidates = [record for idx, uuid, record in rows
                      if idx.strip() == device or (device.startswith("GPU-") and uuid.strip().startswith(device))]
        if len(candidates) != 1:
            # MIG or an unrecognized allocation: avoid memory hints from a different GPU.
            return GpuInfo(count=len(mask))
        matched.append(candidates[0])
    return parse_nvidia_smi("\n".join(matched))
