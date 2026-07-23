"""MLX-local Gemma 4 assistant config loading without Transformers hooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

ASSISTANT_ARCHITECTURE = "Gemma4AssistantForCausalLM"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read Gemma 4 assistant config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Gemma 4 assistant config {path} must be a JSON object")
    return value


def load_assistant_config_dict(
    path_or_repo: str,
    *,
    revision: Optional[str] = None,
    configuration_file: Optional[str] = None,
) -> dict[str, Any]:
    """Load assistant metadata only; never execute model code."""

    if not path_or_repo.strip():
        raise ValueError("Gemma 4 assistant path must not be empty")
    config_name = configuration_file or "config.json"
    local = Path(path_or_repo).expanduser()
    if local.is_file():
        return _read_json(local)
    if local.is_dir():
        return _read_json(local / config_name)

    try:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(
            repo_id=path_or_repo,
            filename=config_name,
            revision=revision,
        )
    except Exception as exc:
        raise ValueError(
            f"Cannot resolve assistant config {path_or_repo!r} at {revision!r}: {exc}"
        ) from exc
    return _read_json(Path(config_path))


def is_gemma4_assistant_config(config: dict[str, Any]) -> bool:
    return config.get("model_type") == "gemma4_assistant" and (
        config.get("architectures") == [ASSISTANT_ARCHITECTURE]
    )
