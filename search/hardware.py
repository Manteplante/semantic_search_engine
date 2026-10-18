"""What this machine can actually run: the GPU if there is one, the CPU otherwise.

`make hardware` prints this.
"""

from __future__ import annotations

import os
import platform
import shutil
from typing import Any

from .config import env, pick_device

GIGABYTE = 1024**3


def has_nvidia_driver() -> bool:
    """A GPU the machine has, whether or not the installed torch can use it."""
    return shutil.which("nvidia-smi") is not None


def advice(driver: bool, torch_cuda: str | None, usable: bool) -> str:
    """The one line worth acting on, given what is installed and what is here."""
    if usable:
        return "using the GPU - nothing to do"
    if driver and not torch_cuda:
        return "an NVIDIA driver is here but torch is a CPU build: reinstall torch with CUDA"
    if driver and torch_cuda:
        return "torch has CUDA but cannot reach the GPU - check the driver, then `make hardware`"
    return "no GPU on this machine: models run on the CPU, so expect `make index` to be slow"


def cpu_name() -> str:
    """The processor, as the machine describes itself."""
    model = platform.processor() or platform.machine()
    try:  # Linux keeps the readable name here; nothing else does
        with open("/proc/cpuinfo") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return model


def memory_gb() -> float | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / GIGABYTE
    except (ValueError, OSError, AttributeError):
        return None


def gpus() -> list[dict[str, Any]]:
    import torch

    if not torch.cuda.is_available():
        return []
    found = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        free, total = torch.cuda.mem_get_info(index)
        found.append(
            {
                "index": index,
                "name": properties.name,
                "capability": f"sm_{properties.major}{properties.minor}",
                "memory_gb": total / GIGABYTE,
                "free_gb": free / GIGABYTE,
                "multiprocessors": properties.multi_processor_count,
            }
        )
    return found


def report() -> list[str]:
    """One printable block: the GPU, the CPU, and what this project will use."""
    import torch

    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    cards = gpus()
    lines = [
        f"torch         {torch.__version__}",
        f"cuda build    {torch.version.cuda or 'none (cpu wheels)'}",
        f"nvidia-smi    {'found' if has_nvidia_driver() else 'not found'}",
    ]
    if cards:
        for card in cards:
            lines.append(
                f"gpu {card['index']}         {card['name']} ({card['capability']}, "
                f"{card['memory_gb']:.1f} GB, {card['free_gb']:.1f} GB free, "
                f"{card['multiprocessors']} SMs)"
            )
    else:
        lines.append("gpu           none usable by torch")
    if mps:
        lines.append("mps           available (Apple silicon)")
    memory = memory_gb()
    lines += [
        f"cpu           {cpu_name()}",
        f"cores         {os.cpu_count()} logical, torch uses {torch.get_num_threads()} threads",
        f"memory        {f'{memory:.1f} GB' if memory else 'unknown'}",
        f"device        {pick_device()}" + ("  (DEVICE is set)" if env("DEVICE") else ""),
        f"advice        {advice(has_nvidia_driver(), torch.version.cuda, bool(cards))}",
    ]
    return lines


def as_advice(error: Exception) -> str | None:
    """Turn the two GPU failures people actually hit into a fix they can run."""
    text = str(error).lower()
    if "c compiler" in text or "triton" in text:
        return (
            "this GPU kernel is compiled on the fly and needs a C compiler: install one "
            "(on Debian/Ubuntu, `apt install build-essential`) or re-run with DEVICE=cpu"
        )
    if "out of memory" in text:
        return "the GPU ran out of memory: lower `search.batch_size`, or re-run with DEVICE=cpu"
    return None
