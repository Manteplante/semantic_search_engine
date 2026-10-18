"""The model is data, not code: read model.yaml, download it, load it.

model.yaml names one model per role: the `embedder` turns text into vectors for
semantic search, the `reranker` re-scores the embedder's best hits for relevance.
Changing one is an edit to model.yaml plus `make get`. Nothing in here, and
nothing that calls it, names a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import config
from .config import ConfigError, settings

ROLES = ("embedder", "reranker")
ENGINES = ("embed", "rerank")
# Each role is driven by different code, so its engine is fixed.
ROLE_ENGINE = {"embedder": "embed", "reranker": "rerank"}
# How token vectors become one text vector: the average (e5, BERT-style models) or the
# last real token's vector (decoder-based embedders such as Qwen3-Embedding).
POOLINGS = ("mean", "last")
FIELDS = {"id", "engine", "about", "max_length", "query_prefix", "passage_prefix", "pooling"}


@dataclass(frozen=True)
class ModelSpec:
    role: str
    id: str
    engine: str
    about: str = ""
    max_length: int = 512
    query_prefix: str = ""
    passage_prefix: str = ""
    pooling: str = "mean"

    @property
    def path(self) -> Path:
        return settings().local_path(self.id)

    @property
    def downloaded(self) -> bool:
        return self.path.is_dir() and any(self.path.iterdir())

    def require(self) -> Path:
        if not self.downloaded:
            raise ConfigError(f"Model {self.id!r} ({self.role}) is not downloaded. Run: make get")
        return self.path


def parse_block(role: str, block: Any, name: str = "model.yaml") -> ModelSpec:
    """One role's block, validated. <ROLE>_ID / _ENGINE in the environment win."""
    if not isinstance(block, dict):
        raise ConfigError(f"{name} needs a `{role}:` block with at least an `id:`")
    unknown = sorted(set(block) - FIELDS)
    if unknown:
        raise ConfigError(
            f"{name}: unknown {role} key {unknown[0]!r}. Valid keys: {', '.join(sorted(FIELDS))}"
        )
    # Prefixes are significant whitespace ("query: "), so they skip clean().
    block = {
        key: value if key.endswith("_prefix") else config.clean(value)
        for key, value in block.items()
        if value is not None
    }
    # EMBEDDER_ID etc. let CI point the whole loop at a tiny model without
    # editing the file people actually read.
    prefix = role.upper()
    block["id"] = config.env(f"{prefix}_ID") or block.get("id", "")
    block["engine"] = config.env(f"{prefix}_ENGINE") or block.get("engine", "")
    if not block["id"]:
        raise ConfigError(f"{name}: `{role}.id` is missing - it is the Hugging Face id")
    if block["engine"] not in ENGINES:
        raise ConfigError(
            f"{name}: {role} engine {block['engine']!r} is not one of {', '.join(ENGINES)}"
        )
    if block["engine"] != ROLE_ENGINE[role]:
        raise ConfigError(f"{name}: the {role} needs `engine: {ROLE_ENGINE[role]}`")
    if block.get("pooling", "mean") not in POOLINGS:
        raise ConfigError(
            f"{name}: {role} pooling {block['pooling']!r} is not one of {', '.join(POOLINGS)}"
        )
    return ModelSpec(role=role, **block)


def specs(path: Path | None = None) -> dict[str, ModelSpec]:
    """Every role in model.yaml, keyed by role."""
    path = path or settings().root / "model.yaml"
    data = config.read_yaml(path, set(ROLES), "role")
    missing = [role for role in ROLES if role not in data]
    if missing:
        raise ConfigError(f"{path.name}: missing the `{missing[0]}:` block")
    return {role: parse_block(role, data[role], path.name) for role in ROLES}


def spec(role: str, path: Path | None = None) -> ModelSpec:
    return specs(path)[role]


# What the Hub says when the token is missing, expired, or has not accepted a licence.
REFUSED = ("gated", "401", "403", "authenticat", "authoriz", "restricted")


def access_advice(model_id: str, error: Exception) -> str | None:
    """Turn "the Hub said no" into the thing to actually do about it."""
    if not any(word in str(error).lower() for word in REFUSED):
        return None
    return (
        f"the Hub refused access to {model_id!r}. If it is gated, accept its licence on "
        f"https://huggingface.co/{model_id} , then put a read token in .env as "
        "HF_TOKEN=... (see .env.example). If it is private, check the token owns it."
    )


def download(found: ModelSpec) -> Path:
    """The only function that may touch the network.

    A token is read from the environment by huggingface_hub itself - `.env` is
    loaded before this runs, so HF_TOKEN there is enough for gated and private
    models. It is never passed around, printed, or written to disk by this repo.
    """
    config.allow_network()
    from huggingface_hub import snapshot_download

    target = found.path
    junk = ["*.h5", "*.msgpack", "*.onnx*", "*.ot", "*.tflite", "onnx/*", "openvino/*"]
    try:
        snapshot_download(found.id, local_dir=target, ignore_patterns=[*junk, "*.bin"])
        if not any(target.glob("*.safetensors")):  # older repos ship .bin only
            snapshot_download(found.id, local_dir=target, ignore_patterns=junk)
    except Exception as problem:
        advice = access_advice(found.id, problem)
        if advice is None:
            raise
        raise ConfigError(advice) from problem
    finally:
        config.go_offline()  # even when the download failed: offline is the default
    return target


def load_tokenizer(found: ModelSpec) -> Any:
    """The tokenizer alone: enough to count tokens, without loading any weights."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(found.require()))


def load(found: ModelSpec) -> tuple[Any, Any]:
    """A (tokenizer, model) pair; `search.embed` / `search.rerank` run it."""
    tokenizer = load_tokenizer(found)
    path = str(found.require())
    from transformers import AutoModel, AutoModelForSequenceClassification

    kind = AutoModelForSequenceClassification if found.engine == "rerank" else AutoModel
    model = kind.from_pretrained(path, dtype="auto").to(config.pick_device()).eval()
    return tokenizer, model
