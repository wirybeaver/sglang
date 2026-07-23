"""Gemma 4 assistant loading and read-only target-KV proposal on MLX."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, Sequence

import mlx.core as mx

from sglang.srt.hardware_backend.mlx.model_adapter import MlxTargetSeed
from sglang.srt.hardware_backend.mlx.spec_config import (
    ASSISTANT_ARCHITECTURE,
    load_assistant_config_dict,
)

_LAYER_TYPES = frozenset({"sliding_attention", "full_attention"})


def _text_target(model: Any) -> tuple[Any, Any, Any]:
    causal = getattr(model, "language_model", model)
    backbone = getattr(causal, "model", None)
    args = getattr(causal, "args", None) or getattr(backbone, "config", None)
    if backbone is None or args is None:
        raise TypeError("expected an mlx-lm Gemma 4 text target")
    model_type = getattr(causal, "model_type", None) or getattr(
        args, "model_type", None
    )
    if model_type not in {"gemma4", "gemma4_text"}:
        raise ValueError(f"unsupported target model_type {model_type!r}")
    return causal, backbone, args


def _reject_remote_hooks(config: Mapping[str, Any], prefix: str = "config") -> None:
    for key, value in config.items():
        path = f"{prefix}.{key}"
        if key in {"auto_map", "model_file"} and value not in (None, {}, [], ""):
            raise ValueError(f"assistant custom-code hook {path!r} is not allowed")
        if isinstance(value, Mapping):
            _reject_remote_hooks(value, path)


@dataclass(frozen=True)
class Gemma4MTPAssistantMetadata:
    backbone_hidden_size: int
    layer_types: tuple[str, ...]
    sliding_head_dim: int
    full_head_dim: int
    final_logit_softcapping: float | None

    def head_dim_for(self, layer_type: str) -> int:
        return (
            self.full_head_dim
            if layer_type == "full_attention"
            else self.sliding_head_dim
        )


def validate_gemma4_assistant_config(
    config: Mapping[str, Any], target_model: Any
) -> Gemma4MTPAssistantMetadata:
    """Validate the assistant shape against a loaded Gemma 4 target."""

    _reject_remote_hooks(config)
    if config.get("model_type") != "gemma4_assistant" or config.get(
        "architectures"
    ) != [ASSISTANT_ARCHITECTURE]:
        raise ValueError(
            "assistant must use model_type='gemma4_assistant' and architecture "
            f"{ASSISTANT_ARCHITECTURE!r}"
        )
    text = config.get("text_config")
    if not isinstance(text, Mapping) or text.get("model_type") != "gemma4_text":
        raise ValueError("assistant requires a gemma4_text text_config")

    causal, _backbone, target = _text_target(target_model)
    layer_types = tuple(text.get("layer_types") or ())
    num_layers = int(text.get("num_hidden_layers", -1))
    target_layers = tuple(getattr(target, "layer_types", ()) or ())
    if (
        num_layers <= 0
        or len(layer_types) != num_layers
        or any(layer_type not in _LAYER_TYPES for layer_type in layer_types)
        or tuple(target_layers[-num_layers:]) != layer_types
    ):
        raise ValueError("assistant layer types must match the target's final tail")

    vocab_size = int(text.get("vocab_size", -1))
    if vocab_size <= 0 or vocab_size != int(getattr(target, "vocab_size", -1)):
        raise ValueError("assistant and target vocabulary sizes must match")
    backbone_hidden = int(config.get("backbone_hidden_size", -1))
    if backbone_hidden != int(getattr(target, "hidden_size", -1)):
        raise ValueError("assistant backbone width must match the target")
    if int(text.get("hidden_size", -1)) <= 0:
        raise ValueError("assistant hidden_size must be positive")
    if int(text.get("num_kv_shared_layers", 0)) != num_layers:
        raise ValueError("every assistant layer must read shared target KV")

    tied = bool(config.get("tie_word_embeddings", False))
    if tied != bool(text.get("tie_word_embeddings", tied)) or tied != bool(
        getattr(causal, "tie_word_embeddings", True)
    ):
        raise ValueError("target and assistant tied-embedding settings must match")

    sliding_dim = int(text.get("head_dim", -1))
    full_dim = int(text.get("global_head_dim", sliding_dim))
    if sliding_dim != int(getattr(target, "head_dim", -1)) or full_dim != int(
        getattr(target, "global_head_dim", sliding_dim)
    ):
        raise ValueError("assistant and target KV head dimensions must match")
    if int(text.get("sliding_window", -1)) != int(
        getattr(target, "sliding_window", -1)
    ):
        raise ValueError("assistant and target sliding windows must match")

    if bool(config.get("use_ordered_embeddings", False)):
        centroids = int(config.get("num_centroids", 0))
        topk = int(config.get("centroid_intermediate_top_k", 0))
        if centroids <= 0 or vocab_size % centroids or not 0 < topk <= centroids:
            raise ValueError("assistant ordered-embedding metadata is invalid")

    softcap = text.get("final_logit_softcapping")
    if softcap is not None and float(softcap) <= 0:
        raise ValueError("assistant final_logit_softcapping must be positive")
    return Gemma4MTPAssistantMetadata(
        backbone_hidden_size=backbone_hidden,
        layer_types=layer_types,
        sliding_head_dim=sliding_dim,
        full_head_dim=full_dim,
        final_logit_softcapping=None if softcap is None else float(softcap),
    )


@dataclass(frozen=True)
class Gemma4MTPKVSharingPlan:
    """Map each assistant attention type to a compact target cache slot."""

    cache_index_by_type: tuple[tuple[str, int], ...]
    expected_cache_entries: int

    @classmethod
    def from_target(
        cls, target_model: Any, metadata: Gemma4MTPAssistantMetadata
    ) -> Gemma4MTPKVSharingPlan:
        _causal, backbone, target = _text_target(target_model)
        layers = tuple(backbone.layers)
        target_types = tuple(getattr(target, "layer_types", ()) or ())
        previous = tuple(getattr(backbone, "previous_kvs", ()))
        if len(layers) != len(target_types) or len(previous) != len(layers):
            raise ValueError("target cache ownership metadata is incomplete")

        compact_owners = tuple(
            index
            for index, layer in enumerate(layers)
            if bool(getattr(layer.self_attn, "has_kv", True))
        )
        owner_to_compact = {owner: index for index, owner in enumerate(compact_owners)}
        mapping = []
        for layer_type in dict.fromkeys(metadata.layer_types):
            logical = max(
                index for index, value in enumerate(target_types) if value == layer_type
            )
            owner = int(previous[logical])
            if (
                owner < 0
                or owner >= len(layers)
                or int(previous[owner]) != owner
                or target_types[owner] != layer_type
                or owner not in owner_to_compact
            ):
                raise ValueError("target YOCO owner metadata is inconsistent")
            mapping.append((layer_type, owner_to_compact[owner]))
        return cls(tuple(mapping), len(compact_owners))


class Gemma4MTPKVView:
    """Short-lived, generation-checked read-only view of target KV."""

    def __init__(
        self,
        runtime: Gemma4MTPAssistantRuntime,
        request_id: str,
        binding: int,
        cache: Sequence[Any],
    ) -> None:
        plan = runtime.sharing_plan
        if len(cache) != plan.expected_cache_entries:
            raise ValueError("target cache cardinality does not match the sharing plan")
        self._runtime = runtime
        self._runtime_generation = runtime.generation
        self._request_id = request_id
        self._binding = binding
        self._cache = tuple(cache)

    @property
    def request_id(self) -> str:
        return self._request_id

    def _validate(self) -> None:
        self._runtime._validate_generation(self._runtime_generation)
        if self._runtime._bindings.get(self._request_id) != self._binding:
            raise RuntimeError(f"stale target-KV view for {self._request_id!r}")

    @property
    def position(self) -> int:
        self._validate()
        offsets = {int(entry.offset) for entry in self._cache}
        if len(offsets) != 1:
            raise RuntimeError("target cache offsets disagree")
        return offsets.pop()

    @staticmethod
    def _logical_kv(entry: Any) -> tuple[mx.array, mx.array]:
        keys = getattr(entry, "keys", None)
        values = getattr(entry, "values", None)
        if keys is None or values is None:
            raise RuntimeError("assistant cannot read an empty target cache")
        offset = int(entry.offset)
        if not hasattr(entry, "_temporal_order"):
            return keys[..., :offset, :], values[..., :offset, :]

        keys = entry._temporal_order(keys)
        values = entry._temporal_order(values)
        valid = min(offset, int(entry.max_size))
        keys = keys[..., -valid:, :] if valid else keys[..., :0, :]
        values = values[..., -valid:, :] if valid else values[..., :0, :]
        padding = offset - valid
        if padding:
            k_pad = mx.zeros((*keys.shape[:-2], padding, keys.shape[-1]), keys.dtype)
            v_pad = mx.zeros(
                (*values.shape[:-2], padding, values.shape[-1]), values.dtype
            )
            keys = mx.concatenate((k_pad, keys), axis=-2)
            values = mx.concatenate((v_pad, values), axis=-2)
        return keys, values

    def shared_kv_states(self) -> dict[str, tuple[mx.array, mx.array]]:
        self._validate()
        shared = {}
        for layer_type, cache_index in self._runtime.sharing_plan.cache_index_by_type:
            keys, values = self._logical_kv(self._cache[cache_index])
            expected = self._runtime.metadata.head_dim_for(layer_type)
            if keys.shape[-1] != expected or values.shape[-1] != expected:
                raise ValueError(f"{layer_type} target KV width does not match")
            shared[layer_type] = keys, values
        return shared


class Gemma4MTPAssistantRuntime:
    """Load-once, generation-checked one-token assistant runtime."""

    def __init__(
        self,
        *,
        owner: Gemma4MTPAssistantLoader,
        generation: int,
        model: Any,
        metadata: Gemma4MTPAssistantMetadata,
        sharing_plan: Gemma4MTPKVSharingPlan,
    ) -> None:
        self._owner = owner
        self.generation = generation
        self._model = model
        self.metadata = metadata
        self.sharing_plan = sharing_plan
        self._active = True
        self._next_binding = 0
        self._bindings: dict[str, int] = {}

    def _validate_generation(self, generation: int) -> None:
        if (
            not self._active
            or generation != self.generation
            or self._owner.runtime is not self
        ):
            raise RuntimeError("stale Gemma 4 assistant runtime")

    def bind_request(self, request_id: str, cache: Sequence[Any]) -> Gemma4MTPKVView:
        self._validate_generation(self.generation)
        binding = self._next_binding + 1
        view = Gemma4MTPKVView(self, request_id, binding, cache)
        self._next_binding = binding
        self._bindings[request_id] = binding
        return view

    def release_request(self, request_id: str) -> None:
        self._bindings.pop(request_id, None)

    def clear_request_bindings(self) -> None:
        self._bindings.clear()

    @property
    def request_binding_count(self) -> int:
        return len(self._bindings)

    def invalidate(self) -> None:
        self.clear_request_bindings()
        self._active = False

    def propose_one(self, seed: MlxTargetSeed, view: Gemma4MTPKVView) -> int:
        self._validate_generation(self.generation)
        if view._runtime is not self:
            raise ValueError("target-KV view belongs to another assistant runtime")
        view._validate()
        expected = (1, 1, self.metadata.backbone_hidden_size)
        if (
            tuple(seed.hidden_state.shape) != expected
            or tuple(seed.token_embedding.shape) != expected
        ):
            raise ValueError(f"assistant seed tensors must both have shape {expected}")

        inputs = mx.concatenate((seed.token_embedding, seed.hidden_state), axis=-1)
        positions = mx.array([[view.position]], dtype=mx.int32)
        _projected, logits = self._model(inputs, view.shared_kv_states(), positions)
        if self.metadata.final_logit_softcapping is not None:
            cap = self.metadata.final_logit_softcapping
            logits = mx.tanh(logits / cap) * cap
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        return int(token.item())


class Gemma4MTPAssistantLoader:
    """Strict, pinned assistant loader; live replacement is unsupported."""

    def __init__(self, target_model: Any):
        _text_target(target_model)
        self.target_model = target_model
        self.generation = 0
        self.runtime: Gemma4MTPAssistantRuntime | None = None

    def load(
        self,
        path_or_repo: str,
        *,
        revision: str | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> Gemma4MTPAssistantRuntime:
        if self.runtime is not None:
            raise RuntimeError("live assistant replacement is not supported")
        try:
            provider_version = version("mlx-vlm")
        except PackageNotFoundError:
            provider_version = None
        if provider_version != "0.5.0":
            raise RuntimeError("Gemma 4 MTP requires mlx-vlm==0.5.0")

        config = config or load_assistant_config_dict(path_or_repo, revision=revision)
        metadata = validate_gemma4_assistant_config(config, self.target_model)
        sharing_plan = Gemma4MTPKVSharingPlan.from_target(self.target_model, metadata)

        from mlx_vlm.utils import get_model_path, load_model

        local = Path(path_or_repo).expanduser()
        provider_path = str(local) if local.exists() else path_or_repo
        checkpoint = get_model_path(provider_path, revision=revision)
        if checkpoint.is_file():
            checkpoint = checkpoint.parent
        model = load_model(checkpoint)
        model.bind(self.target_model)

        self.generation += 1
        self.runtime = Gemma4MTPAssistantRuntime(
            owner=self,
            generation=self.generation,
            model=model,
            metadata=metadata,
            sharing_plan=sharing_plan,
        )
        return self.runtime

    def unload(self) -> None:
        if self.runtime is not None:
            self.runtime.invalidate()
        self.runtime = None
        self.generation += 1
