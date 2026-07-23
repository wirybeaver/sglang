from __future__ import annotations

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sglang.test.ci.ci_register import register_mlx_ci

register_mlx_ci(est_time=2, suite="stage-a-unit-test-mlx")

_HAS_PROVIDER = (
    importlib.util.find_spec("mlx") is not None
    and importlib.util.find_spec("mlx_vlm") is not None
    and importlib.util.find_spec("mlx_vlm.speculative.drafters.gemma4_assistant")
    is not None
)

if _HAS_PROVIDER:
    import mlx.core as mx
    from registered.unit.hardware_backend.mlx.utils import (
        assert_native_cache_equal,
        tiny_assistant_config,
        tiny_gemma4,
        write_tiny_assistant_checkpoint,
    )

    from sglang.srt.hardware_backend.mlx.gemma4_mtp import (
        Gemma4MTPAssistantLoader,
        validate_gemma4_assistant_config,
    )
    from sglang.srt.hardware_backend.mlx.kv_cache.native_transaction import (
        clone_native_cache,
    )
    from sglang.srt.hardware_backend.mlx.model_adapter import Gemma4TargetAdapter


@unittest.skipUnless(_HAS_PROVIDER, "requires mlx-vlm Gemma 4 assistant provider")
class TestGemma4MTPAssistant(unittest.TestCase):
    def test_config_strict_load_and_lifecycle(self):
        target = tiny_gemma4()
        metadata = validate_gemma4_assistant_config(
            tiny_assistant_config(ordered=True), target
        )
        self.assertEqual(metadata.layer_types, tuple(target.args.layer_types))
        self.assertEqual(metadata.sliding_head_dim, 4)
        self.assertEqual(metadata.full_head_dim, 8)

        mutations = {
            "custom-code": lambda value: value.update(auto_map={"AutoModel": "x.py"}),
            "vocabulary": lambda value: value["text_config"].update(vocab_size=31),
            "final tail": lambda value: value["text_config"]["layer_types"].reverse(),
            "ordered-embedding": lambda value: value.update(num_centroids=3),
        }
        for message, mutate in mutations.items():
            with self.subTest(message=message):
                config = copy.deepcopy(tiny_assistant_config(ordered=True))
                mutate(config)
                with self.assertRaisesRegex(ValueError, message):
                    validate_gemma4_assistant_config(config, target)

        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp)
            write_tiny_assistant_checkpoint(checkpoint)
            loader = Gemma4MTPAssistantLoader(target)
            with mock.patch(
                "sglang.srt.hardware_backend.mlx.gemma4_mtp.version",
                return_value="0.5.1",
            ), self.assertRaisesRegex(RuntimeError, "requires mlx-vlm==0.5.0"):
                loader.load(str(checkpoint))

            runtime = loader.load(str(checkpoint / "config.json"))
            self.assertIs(loader.runtime, runtime)
            cache = target.make_cache()
            target(mx.array([[1, 2, 3]], dtype=mx.int32), cache=cache)
            view = runtime.bind_request("request", cache)
            self.assertEqual(view.position, 3)
            self.assertEqual(runtime.request_binding_count, 1)
            with self.assertRaisesRegex(ValueError, "cardinality"):
                runtime.bind_request("request", cache[:-1])
            self.assertEqual(view.position, 3)
            self.assertEqual(runtime.request_binding_count, 1)
            runtime.release_request("request")
            with self.assertRaisesRegex(RuntimeError, "stale"):
                _ = view.position

            with self.assertRaisesRegex(RuntimeError, "replacement"):
                loader.load(str(checkpoint))

            stale_view = runtime.bind_request("request", cache)
            loader.unload()
            self.assertIsNone(loader.runtime)
            with self.assertRaisesRegex(RuntimeError, "stale"):
                runtime.bind_request("late", target.make_cache())
            with self.assertRaisesRegex(RuntimeError, "stale"):
                _ = stale_view.position

    def test_proposal_reads_rotating_yoco_cache_without_mutation(self):
        for ordered in (False, True):
            with self.subTest(ordered=ordered), tempfile.TemporaryDirectory() as temp:
                target = tiny_gemma4()
                adapter = Gemma4TargetAdapter(target)
                cache = target.make_cache()
                prompt = list(range(1, 18))
                output = adapter.forward(
                    mx.array([prompt], dtype=mx.int32),
                    cache=cache,
                    collect_hidden_states=True,
                )
                target_token = int(mx.argmax(output.logits[:, -1, :], axis=-1).item())
                seed = adapter.make_seed(
                    output,
                    hidden_row_index=len(prompt) - 1,
                    emitted_token_id=target_token,
                )
                reference_cache = clone_native_cache(cache)

                checkpoint = Path(temp)
                write_tiny_assistant_checkpoint(checkpoint, ordered=ordered)
                runtime = Gemma4MTPAssistantLoader(target).load(str(checkpoint))
                self.assertEqual(
                    runtime.sharing_plan.cache_index_by_type,
                    (("sliding_attention", 0), ("full_attention", 1)),
                )
                self.assertEqual(runtime.sharing_plan.expected_cache_entries, 2)

                view = runtime.bind_request("request", cache)
                self.assertEqual(view.position, len(prompt))
                shared = view.shared_kv_states()
                self.assertEqual(
                    shared["sliding_attention"][0].shape, (1, 1, len(prompt), 4)
                )
                self.assertEqual(
                    shared["full_attention"][0].shape, (1, 1, len(prompt), 8)
                )

                inputs = mx.concatenate(
                    (seed.token_embedding, seed.hidden_state), axis=-1
                )
                _projected, logits = runtime._model(
                    inputs,
                    shared,
                    mx.array([[len(prompt)]], dtype=mx.int32),
                )
                if runtime.metadata.final_logit_softcapping is not None:
                    cap = runtime.metadata.final_logit_softcapping
                    logits = mx.tanh(logits / cap) * cap
                expected = int(mx.argmax(logits[:, -1, :], axis=-1).item())
                self.assertEqual(runtime.propose_one(seed, view), expected)
                assert_native_cache_equal(self, cache, reference_cache)

                runtime.release_request("request")
                with self.assertRaisesRegex(RuntimeError, "stale"):
                    _ = view.position


if __name__ == "__main__":
    unittest.main()
