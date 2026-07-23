from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from sglang.test.ci.ci_register import register_mlx_ci

register_mlx_ci(est_time=1, suite="stage-a-unit-test-mlx")

_HAS_MLX = (
    importlib.util.find_spec("mlx") is not None
    and importlib.util.find_spec("mlx_lm.models.gemma4_text") is not None
)

if _HAS_MLX:
    import mlx.core as mx
    from registered.unit.hardware_backend.mlx.utils import (
        assert_native_cache_equal,
        build_runner,
        tiny_gemma4,
    )

    from sglang.srt.hardware_backend.mlx.model_adapter import (
        Gemma4TargetAdapter,
        MlxTargetForwardOutput,
    )
    from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner


@unittest.skipUnless(_HAS_MLX, "requires MLX and mlx-lm Gemma 4")
class TestGemma4TargetAdapter(unittest.TestCase):
    def test_wrapped_and_unwrapped_capture_preserve_exact_target_math(self):
        for wrapped in (False, True):
            with self.subTest(wrapped=wrapped):
                model = tiny_gemma4(wrapped=wrapped)
                adapter = Gemma4TargetAdapter(model)
                ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

                normal_cache = model.make_cache()
                normal = model(ids, cache=normal_cache)
                captured_cache = model.make_cache()
                captured = adapter.forward(
                    ids, cache=captured_cache, collect_hidden_states=True
                )
                mx.eval(normal, captured.logits, captured.hidden_states)

                np.testing.assert_array_equal(normal, captured.logits)
                np.testing.assert_array_equal(
                    mx.argmax(normal, axis=-1), mx.argmax(captured.logits, axis=-1)
                )
                self.assertEqual(captured.hidden_states.shape, (1, 4, 16))
                assert_native_cache_equal(self, captured_cache, normal_cache)

    def test_prefill_retains_only_the_scaled_final_seed_row(self):
        model = tiny_gemma4()
        runner = build_runner(model)
        token, output = runner.prefill_for_mtp(
            "request", [1, 2, 3], [1, 2, 3], [], [], 0
        )
        adapter = runner._get_target_adapter()
        seed = adapter.make_seed(output, hidden_row_index=0, emitted_token_id=token)
        expected_embedding = (
            model.model.embed_tokens(mx.array([[token]], dtype=mx.int32))
            * model.model.embed_scale
        )
        mx.eval(seed.hidden_state, seed.token_embedding, expected_embedding)

        self.assertEqual(output.hidden_states.shape, (1, 1, 16))
        self.assertEqual(seed.hidden_state.shape, (1, 1, 16))
        np.testing.assert_array_equal(seed.token_embedding, expected_embedding)

    def test_verify_and_replay_use_one_token_forward_shapes(self):
        calls = []

        class RecordingAdapter:
            def forward(self, input_ids, *, cache, collect_hidden_states):
                del cache
                calls.append((tuple(input_ids.shape), collect_hidden_states))
                width = input_ids.shape[1]
                return MlxTargetForwardOutput(
                    logits=mx.zeros((1, width, 7)),
                    hidden_states=(
                        mx.zeros((1, width, 5)) if collect_hidden_states else None
                    ),
                )

        runner = object.__new__(MlxModelRunner)
        runner._native_cache_fallback = True
        runner._target_adapter = RecordingAdapter()
        output = runner._forward_native_queries_sequential(
            [], (11, 12), collect_hidden_states=True
        )
        self.assertEqual(calls, [((1, 1), True), ((1, 1), True)])
        self.assertEqual(output.hidden_states.shape, (1, 2, 5))

        calls.clear()
        runner._forward_native_queries_sequential(
            [], (13, 14), collect_hidden_states=False
        )
        self.assertEqual(calls, [((1, 1), False), ((1, 1), False)])

    def test_target_loader_receives_pinned_revision(self):
        runner = MlxModelRunner.__new__(MlxModelRunner)
        runner.model_path = "target"
        runner.revision = "pinned-revision"
        runner.trust_remote_code = False
        runner._quantization = None
        fake_model = SimpleNamespace(parameters=lambda: ())
        with (
            mock.patch(
                "sglang.srt.hardware_backend.mlx.model_runner.mlx_lm_load",
                return_value=(fake_model, None, {}),
            ) as load,
            mock.patch("sglang.srt.hardware_backend.mlx.model_runner.mx.eval"),
        ):
            runner._load_model()
        self.assertEqual(load.call_args.kwargs["revision"], "pinned-revision")


if __name__ == "__main__":
    unittest.main()
