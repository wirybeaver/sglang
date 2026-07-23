"""MLX-local Gemma 4 assistant config loading without Transformers hooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.server_args import ServerArgs

ASSISTANT_ARCHITECTURE = "Gemma4AssistantForCausalLM"
ASSISTANT_ARCHITECTURES = frozenset(
    {ASSISTANT_ARCHITECTURE, "Gemma4UnifiedAssistantForCausalLM"}
)
MLX_GEMMA4_MTP_MAX_CONTEXT = 2048
MLX_GEMMA4_MTP_VERIFY_WIDTH = 2
MLX_GEMMA4_MTP_TARGET_ARCHITECTURE = "Gemma4ForConditionalGeneration"


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


def is_gemma4_assistant_family(config: dict[str, Any]) -> bool:
    """Recognize Gemma 4 assistants before strict MVP validation."""

    architectures = config.get("architectures")
    return config.get("model_type") == "gemma4_assistant" or any(
        architecture in ASSISTANT_ARCHITECTURES
        for architecture in (architectures if isinstance(architectures, list) else ())
    )


def _attr(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _validate_target_config(server_args: ServerArgs) -> None:
    model_config = server_args.get_model_config()
    config = model_config.hf_config
    text = _attr(config, "text_config")
    if (
        _attr(config, "model_type") != "gemma4"
        or _attr(config, "architectures", []) != [MLX_GEMMA4_MTP_TARGET_ARCHITECTURE]
        or text is None
    ):
        raise ValueError(
            "MLX Frozen-KV MTP currently supports only the Gemma 4 E2B text "
            f"target ({MLX_GEMMA4_MTP_TARGET_ARCHITECTURE})."
        )
    if (
        _attr(text, "model_type") != "gemma4_text"
        or _attr(text, "hidden_size") != 1536
        or _attr(text, "num_hidden_layers") != 35
        or _attr(text, "vocab_size") != 262144
    ):
        raise ValueError(
            "MLX Frozen-KV MTP requires the Gemma 4 E2B target shape "
            "(hidden_size=1536, layers=35, vocab_size=262144)."
        )
    if bool(_attr(model_config, "is_multimodal", False)):
        raise ValueError("MLX Frozen-KV MTP supports text-only execution.")


def _validate_assistant_mvp_config(config: dict[str, Any]) -> None:
    if config.get("model_type") != "gemma4_assistant" or config.get(
        "architectures"
    ) != [ASSISTANT_ARCHITECTURE]:
        raise ValueError(
            "MLX Frozen-KV MTP requires a canonical gemma4_assistant checkpoint."
        )
    if any(
        config.get(key) not in (None, {}, [], "") for key in ("auto_map", "model_file")
    ):
        raise ValueError("MLX Frozen-KV MTP does not load assistant custom code.")
    text = config.get("text_config")
    if (
        not isinstance(text, dict)
        or config.get("backbone_hidden_size") != 1536
        or text.get("model_type") != "gemma4_text"
        or text.get("hidden_size") != 256
        or text.get("num_hidden_layers") != 4
        or text.get("vocab_size") != 262144
    ):
        raise ValueError(
            "MLX Frozen-KV MTP requires the matching four-layer E2B assistant."
        )
    if list(text.get("layer_types") or []) != [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]:
        raise ValueError(
            "The E2B assistant must end with three sliding and one full layer."
        )
    if (
        not bool(config.get("use_ordered_embeddings"))
        or config.get("num_centroids") != 2048
        or config.get("centroid_intermediate_top_k") != 32
    ):
        raise ValueError(
            "The E2B assistant requires ordered embeddings with 2,048 centroids "
            "and intermediate top-k 32."
        )


def validate_mlx_frozen_kv_mtp_args(server_args: ServerArgs) -> None:
    """Validate the exact Level-1 server shape before assistant weights load."""

    if not getattr(server_args, "speculative_draft_model_path", None):
        raise ValueError("MLX Frozen-KV MTP requires --speculative-draft-model-path.")
    if int(getattr(server_args, "speculative_eagle_topk", -1)) != 1:
        raise ValueError("MLX Frozen-KV MTP requires --speculative-eagle-topk 1.")
    if int(getattr(server_args, "speculative_num_steps", -1)) != 1:
        raise ValueError("MLX Frozen-KV MTP requires --speculative-num-steps 1.")
    if (
        int(getattr(server_args, "speculative_num_draft_tokens", -1))
        != MLX_GEMMA4_MTP_VERIFY_WIDTH
    ):
        raise ValueError("MLX Frozen-KV MTP requires --speculative-num-draft-tokens 2.")
    if bool(getattr(server_args, "speculative_use_rejection_sampling", False)):
        raise ValueError("MLX Frozen-KV MTP does not support rejection sampling.")
    if not bool(getattr(server_args, "disable_overlap_schedule", False)):
        raise ValueError("MLX Frozen-KV MTP requires synchronous scheduling.")
    if not bool(getattr(server_args, "disable_radix_cache", False)):
        raise NotImplementedError("MLX Frozen-KV MTP requires --disable-radix-cache.")
    if int(getattr(server_args, "chunked_prefill_size", 0)) != -1:
        raise NotImplementedError(
            "MLX Frozen-KV MTP requires --chunked-prefill-size -1."
        )
    if bool(getattr(server_args, "enable_mixed_chunk", False)):
        raise NotImplementedError(
            "MLX Frozen-KV MTP does not support mixed chunked prefill."
        )
    if int(getattr(server_args, "max_running_requests", 0)) != 1:
        raise NotImplementedError(
            "MLX Frozen-KV MTP requires --max-running-requests 1."
        )

    model_config = server_args.get_model_config()
    context_length = getattr(server_args, "context_length", None)
    context_length = (
        model_config.context_len if context_length is None else context_length
    )
    if int(context_length) > MLX_GEMMA4_MTP_MAX_CONTEXT:
        raise NotImplementedError(
            "MLX Frozen-KV MTP supports context length no greater than 2,048."
        )
    max_total_tokens = getattr(server_args, "max_total_tokens", None)
    if max_total_tokens is not None and int(max_total_tokens) < int(context_length):
        raise NotImplementedError(
            "MLX Frozen-KV MTP requires --max-total-tokens no smaller than "
            "--context-length."
        )

    for field, flag in (
        ("tp_size", "tp-size"),
        ("dp_size", "dp-size"),
        ("pp_size", "pp-size"),
        ("nnodes", "nnodes"),
    ):
        if int(getattr(server_args, field, 1)) != 1:
            raise NotImplementedError(f"MLX Frozen-KV MTP requires --{flag} 1.")
    if bool(getattr(server_args, "enable_dp_attention", False)):
        raise NotImplementedError("MLX Frozen-KV MTP does not support DP attention.")
    disaggregation = str(getattr(server_args, "disaggregation_mode", "null")).lower()
    if disaggregation not in ("null", "none") and not disaggregation.endswith(".null"):
        raise NotImplementedError("MLX Frozen-KV MTP does not support disaggregation.")
    if bool(getattr(server_args, "language_only", False)) or bool(
        getattr(server_args, "encoder_only", False)
    ):
        raise NotImplementedError(
            "MLX Frozen-KV MTP does not support encoder disaggregation modes."
        )

    _validate_target_config(server_args)
    assistant_config = load_assistant_config_dict(
        server_args.speculative_draft_model_path,
        revision=getattr(server_args, "speculative_draft_model_revision", None),
        configuration_file=(
            getattr(server_args, "decrypted_draft_config_file", None) or None
        ),
    )
    _validate_assistant_mvp_config(assistant_config)
    server_args._mlx_gemma4_mtp_assistant_config = assistant_config


def validate_mlx_frozen_kv_mtp_request(
    req: Req, *, has_multimodal: bool = False
) -> Optional[str]:
    """Return an admission error for request features outside the MVP."""

    params = req.sampling_params
    if int(params.top_k) != 1:
        return (
            "MLX Frozen-KV MTP requires greedy requests with temperature=0 "
            "(normalized top_k must equal 1)."
        )
    if (
        float(params.frequency_penalty) != 0.0
        or float(params.presence_penalty) != 0.0
        or float(params.repetition_penalty) != 1.0
        or int(params.min_new_tokens) != 0
    ):
        return "MLX Frozen-KV MTP does not support sampling penalties."
    if params.logit_bias is not None:
        return "MLX Frozen-KV MTP does not support logit bias."
    if any(
        bool(value)
        for value in (
            params.json_schema,
            params.regex,
            params.ebnf,
            params.structural_tag,
            params.stop_regex_strs,
        )
    ):
        return "MLX Frozen-KV MTP does not support constrained decoding."
    if int(params.n) != 1:
        return "MLX Frozen-KV MTP supports one completion per request."
    if params.custom_params:
        return "MLX Frozen-KV MTP does not support custom sampling parameters."
    if bool(getattr(req, "return_logprob", False)):
        return "MLX Frozen-KV MTP does not support logprobs."
    if bool(getattr(req, "return_hidden_states", False)):
        return "MLX Frozen-KV MTP does not return target hidden states."
    if bool(getattr(req, "return_sampling_mask", False)):
        return "MLX Frozen-KV MTP does not return sampling masks."
    if getattr(req, "custom_logit_processor", None) is not None:
        return "MLX Frozen-KV MTP does not support custom logits processors."
    if (
        getattr(req, "session", None) is not None
        or getattr(req, "session_id", None) is not None
    ):
        return "MLX Frozen-KV MTP does not support sessions."
    if getattr(req, "lora_id", None) is not None:
        return "MLX Frozen-KV MTP does not support LoRA requests."
    if getattr(req, "input_embeds", None) is not None:
        return "MLX Frozen-KV MTP requires token-ID inputs."
    if has_multimodal or getattr(req, "multimodal_inputs", None) is not None:
        return "MLX Frozen-KV MTP is text-only."
    return None
