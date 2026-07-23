from __future__ import annotations

import importlib.util
import unittest

from sglang.test.ci.ci_register import register_mlx_ci

register_mlx_ci(est_time=2, suite="stage-a-unit-test-mlx")

_HAS_MLX = (
    importlib.util.find_spec("mlx") is not None
    and importlib.util.find_spec("mlx_lm.models.gemma4_text") is not None
)

if _HAS_MLX:
    import mlx.core as mx
    from registered.unit.hardware_backend.mlx.utils import (
        assert_native_cache_equal,
        tiny_gemma4,
    )

    from sglang.srt.hardware_backend.mlx.kv_cache.native_transaction import (
        MlxNativeCacheTransaction,
        clone_native_cache,
    )
    from sglang.srt.hardware_backend.mlx.spec_decode import verify_one_draft


@unittest.skipUnless(_HAS_MLX, "requires MLX and mlx-lm Gemma 4")
class TestGemma4NativeCacheTransaction(unittest.TestCase):
    def setUp(self):
        self.model = tiny_gemma4()

    def _sequential_forward(self, cache, tokens):
        outputs = [
            self.model(mx.array([[token]], dtype=mx.int32), cache=cache)
            for token in tokens
        ]
        return mx.concatenate(outputs, axis=1)

    def _prefill(self, prompt):
        cache = self.model.make_cache()
        logits = self.model(mx.array([prompt], dtype=mx.int32), cache=cache)
        root = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(root, *[item for entry in cache for item in entry.state])
        return cache, int(root.item())

    def _exercise(self, prompt, *, accept):
        cache, root = self._prefill(prompt)
        probe = clone_native_cache(cache)
        draft_logits = self._sequential_forward(probe, (root,))
        draft = int(mx.argmax(draft_logits[:, -1, :], axis=-1).item())
        if not accept:
            draft = (draft + 1) % self.model.args.vocab_size

        transaction = MlxNativeCacheTransaction(
            cache, (root, draft), self._sequential_forward
        )
        before = clone_native_cache(cache)
        speculative = transaction.begin()
        logits = self._sequential_forward(speculative, (root, draft))
        target_ids = tuple(
            int(token) for token in mx.argmax(logits, axis=-1).reshape(-1).tolist()
        )
        decision = verify_one_draft("request", draft, target_ids)
        transaction.prepare(decision.committed_query_count)
        assert_native_cache_equal(self, cache, before)
        transaction.commit()

        reference, _ = self._prefill(prompt)
        self._sequential_forward(
            reference, (root, draft)[: decision.committed_query_count]
        )
        mx.eval(*[item for entry in reference for item in entry.state])
        assert_native_cache_equal(self, cache, reference)

    def test_accept_reject_and_rotation_match_target_only(self):
        for prompt_len, accept in ((3, True), (8, False), (17, True)):
            with self.subTest(prompt_len=prompt_len, accept=accept):
                prompt = [1 + index % 29 for index in range(prompt_len)]
                self._exercise(prompt, accept=accept)

    def test_abort_and_replay_failure_leave_live_cache_unchanged(self):
        cache, root = self._prefill(list(range(1, 12)))
        before = clone_native_cache(cache)

        transaction = MlxNativeCacheTransaction(
            cache, (root,), self._sequential_forward
        )
        transaction.begin()
        transaction.abort()
        assert_native_cache_equal(self, cache, before)

        def failing_replay(target_cache, tokens):
            result = self._sequential_forward(target_cache, tokens)
            mx.eval(result, *[item for entry in target_cache for item in entry.state])
            raise RuntimeError("synthetic replay failure")

        transaction = MlxNativeCacheTransaction(cache, (root,), failing_replay)
        transaction.begin()
        with self.assertRaisesRegex(RuntimeError, "synthetic replay failure"):
            transaction.prepare(1)
        assert_native_cache_equal(self, cache, before)


if __name__ == "__main__":
    unittest.main()
