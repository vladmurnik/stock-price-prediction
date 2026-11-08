"""Shared Hugging Face / offline plumbing for the sentiment backends.

Two things every backend needs and none of them should reimplement:

* :func:`configure_hf_cache` — on Windows the hub tries to create symlinks, which fails for
  non-admins (``WinError 1314``).  We ask the hub to copy instead of symlink.
* :func:`resolve_model_version` — a *stable* cache key.  T4/T5 require ``model_version`` to be a
  deterministic function of what the backend actually does, so a question edit or a revision bump
  invalidates cached shards.  We combine the HF repo revision (if easily obtainable) with a hash of
  the backend configuration.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

__all__ = ["configure_hf_cache", "resolve_model_version", "stable_hash"]


def configure_hf_cache() -> None:
    """Disable hub symlinks (Windows non-admin) unless the operator says otherwise."""
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    # Keep transformers' progress bars out of the log files; we emit our own progress.
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


def stable_hash(payload: Any) -> str:
    """sha1 of a canonical JSON dump (sorted keys) — stable across processes."""
    blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def resolve_model_version(
    repo_id: str, config_payload: dict[str, Any], repo_type: str = "model"
) -> str:
    """Build a ``model_version`` string: ``<repo>@<rev>+cfg<sha1[:12]>``.

    The revision lookup is best-effort: if the hub is unreachable we fall back to ``rev=unknown``
    but the configuration hash still guarantees that a semantic change invalidates the cache.
    """
    configure_hf_cache()
    revision = "unknown"
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        info = api.dataset_info(repo_id) if repo_type == "dataset" else api.model_info(repo_id)
        revision = info.sha or "unknown"
    except Exception:  # noqa: BLE001 - offline is a legitimate state; hash still separates configs
        pass
    cfg_hash = stable_hash(config_payload)[:12]
    return f"{repo_id}@{revision}+cfg{cfg_hash}"
