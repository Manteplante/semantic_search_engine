"""Paths, device selection, and the offline switch.

Importing this module puts the process offline: `HF_HUB_OFFLINE=1` is set
before anything imports transformers. Exactly one command - `make get` - calls
`allow_network()` first. Everything else can only read weights already on disk,
so a missing model is an error, never a silent download.
"""

from __future__ import annotations

import functools
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def go_offline() -> None:
    """Forbid every network call the Hugging Face stack might make."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def without_compiler_skip_torch_jit() -> None:
    """Torch compiles some GPU kernels (e.g. Qwen3's rotary embedding) with Triton, which
    needs a C compiler. Without one, fall back to the plain kernels instead of crashing."""
    if not any(shutil.which(name) for name in ("cc", "gcc", "clang")):
        os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")


def allow_network() -> None:
    """Called by `make get` only, and only before transformers is imported."""
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"


def clean(value: Any) -> Any:
    """Strip the trailing comment some editors keep when exporting a .env value.

    `python-dotenv` removes `KEY=value  # note`; a shell that exports the file
    into a subprocess does not, and the comment ends up inside your path.
    """
    if isinstance(value, str):
        value = value.split(" #", 1)[0].strip()
    return value


def load_env(path: Path | None = None) -> dict[str, str]:
    """Read `.env` into the environment. The shell wins over the file.

    Hand-rolled rather than a python-dotenv dependency: fifteen lines, and
    `clean()` already handles the trailing-comment trap that bites either way.
    Secrets live here - `.env` is gitignored, and nothing prints a value back.
    """
    path = Path(path or ROOT / ".env")
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    for line in path.read_text().splitlines():
        name, separator, value = line.strip().removeprefix("export ").partition("=")
        name = name.strip()
        if not separator or not name or name.startswith("#"):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]  # quoted: taken verbatim, so a `#` can be in a token
        else:
            value = str(clean(value))
        if name in os.environ:  # an exported variable beats the file
            continue
        os.environ[name] = value
        loaded[name] = value
    return loaded


# .env first, then the offline switch - so no .env line can turn the switch off.
load_env()
go_offline()
without_compiler_skip_torch_jit()


def env(name: str, default: str = "") -> str:
    return str(clean(os.environ.get(name, default)))


@dataclass(frozen=True)
class Settings:
    root: Path
    models_dir: Path
    input_dir: Path  # parsed markdown, written by `make parse`
    data_dir: Path  # source PDFs, the SQLite database and exports
    output_dir: Path  # corpus and search index

    def local_path(self, model_id: str) -> Path:
        """Where the weights for a Hub id live on this machine."""
        return self.models_dir / model_id.replace("/", "__")


@functools.lru_cache(maxsize=1)
def settings() -> Settings:
    """Cached, so clear it at the start of every CLI invocation.

    An in-process caller can otherwise warm the cache before the environment is
    set, and every later command silently uses the wrong paths.
    """
    root = Path(env("PROJECT_ROOT") or ROOT)
    models = Path(env("MODELS_DIR") or root / "models_local")
    return Settings(
        root=root,
        models_dir=models if models.is_absolute() else root / models,
        input_dir=root / "parsing" / "output",
        data_dir=root / "data",
        output_dir=root / "search" / "output",
    )


def reset() -> None:
    """Drop cached settings. The CLI calls this before doing anything."""
    settings.cache_clear()
    pick_device.cache_clear()


@functools.lru_cache(maxsize=1)
def pick_device() -> str:
    """DEVICE from the environment, or the best thing torch can see."""
    chosen = env("DEVICE")
    if chosen:
        return chosen
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ConfigError(Exception):
    """A config file is wrong, and the message says which key and what to do."""


def read_yaml(path: Path, allowed: set[str], what: str) -> dict[str, Any]:
    """Read a config file, rejecting keys that are not settings - a typo is an error."""
    if not path.exists():
        raise ConfigError(f"No {path.name} in {path.parent}. Restore it from git.")
    loaded = yaml.safe_load(path.read_text()) or {}
    if not isinstance(loaded, dict):
        kind = type(loaded).__name__
        raise ConfigError(f"{path.name} should be a mapping of settings, got {kind}")
    data = {key: clean(value) for key, value in loaded.items()}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(
            f"{path.name}: unknown {what} key {unknown[0]!r}. "
            f"Valid keys: {', '.join(sorted(allowed))}"
        )
    return data
