"""Isolated verification transactions for mlx-lm native caches."""

from __future__ import annotations

import copy
from typing import Any, Callable, Iterable, Sequence

import mlx.core as mx


def _clone_value(value: Any) -> Any:
    if isinstance(value, mx.array):
        return mx.array(value)
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    return copy.deepcopy(value)


def clone_native_cache_entry(entry: Any) -> Any:
    """Clone one cache object, including provider-specific ring metadata."""

    clone = type(entry).__new__(type(entry))
    if hasattr(entry, "__dict__"):
        clone.__dict__.update(_clone_value(entry.__dict__))
        return clone

    for cls in type(entry).__mro__:
        declared = getattr(cls, "__slots__", ())
        if isinstance(declared, str):
            declared = (declared,)
        for name in declared:
            if name not in {"__dict__", "__weakref__"} and hasattr(entry, name):
                setattr(clone, name, _clone_value(getattr(entry, name)))
    return clone


def _iter_arrays(value: Any) -> Iterable[mx.array]:
    if isinstance(value, mx.array):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_arrays(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_arrays(item)
    else:
        state = getattr(value, "state", None)
        if state is not None:
            yield from _iter_arrays(state)


def clone_native_cache(cache: Sequence[Any]) -> list[Any]:
    """Clone all arrays, offsets, valid lengths, and ring metadata."""

    cloned = [clone_native_cache_entry(entry) for entry in cache]
    arrays = list(_iter_arrays(cloned))
    if arrays:
        mx.eval(*arrays)
    return cloned


ReplayForward = Callable[[list[Any], tuple[int, ...]], Any]


class MlxNativeCacheTransaction:
    """Verify on a clone and atomically install only the accepted replay."""

    def __init__(
        self,
        cache: list[Any],
        query_token_ids: Sequence[int],
        replay_forward: ReplayForward,
    ) -> None:
        queries = tuple(int(token) for token in query_token_ids)
        if not queries or any(token < 0 for token in queries):
            raise ValueError("a cache transaction requires non-negative queries")
        self._cache = cache
        self._queries = queries
        self._replay_forward = replay_forward
        self._active = False
        self._finished = False
        self._speculative_cache: list[Any] | None = None
        self._candidate_cache: list[Any] | None = None

    @property
    def active(self) -> bool:
        return self._active

    def begin(self) -> list[Any]:
        if self._active or self._finished:
            raise RuntimeError("native-cache transactions are single-use")
        self._speculative_cache = clone_native_cache(self._cache)
        self._active = True
        return self._speculative_cache

    @property
    def candidate_cache(self) -> list[Any]:
        if not self._active or self._candidate_cache is None:
            raise RuntimeError("the transaction has no prepared commit candidate")
        return self._candidate_cache

    def prepare(self, count: int) -> Any:
        if not self._active:
            raise RuntimeError("only an active native-cache transaction can prepare")
        if self._candidate_cache is not None:
            raise RuntimeError("the transaction already has a commit candidate")
        if count < 1 or count > len(self._queries):
            raise ValueError(f"commit count must be in [1, {len(self._queries)}]")

        # Replay into another clone. If forward/materialization fails, the live
        # cache was never touched and needs no rollback mutation.
        committed_cache = clone_native_cache(self._cache)
        try:
            result = self._replay_forward(committed_cache, self._queries[:count])
            arrays = list(_iter_arrays(result)) + list(_iter_arrays(committed_cache))
            if arrays:
                mx.eval(*arrays)
        except BaseException:
            self._speculative_cache = None
            self._active = False
            self._finished = True
            raise
        self._candidate_cache = committed_cache
        return result

    def commit(self) -> None:
        if not self._active or self._candidate_cache is None:
            raise RuntimeError("prepare() must succeed before commit()")
        self._cache[:] = self._candidate_cache
        self._speculative_cache = None
        self._candidate_cache = None
        self._active = False
        self._finished = True

    def abort(self) -> None:
        if not self._active:
            raise RuntimeError("only an active native-cache transaction can abort")
        self._speculative_cache = None
        self._candidate_cache = None
        self._active = False
        self._finished = True
