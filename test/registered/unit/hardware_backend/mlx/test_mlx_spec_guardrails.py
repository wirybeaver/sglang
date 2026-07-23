"""Focused guardrails for the Gemma 4 Frozen-KV MTP MLX prototype."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from sglang.srt.arg_groups.speculative_hook import (
    _handle_frozen_kv_mtp,
    _resolve_speculative_algorithm_alias,
)
from sglang.srt.hardware_backend.mlx.spec_config import (
    validate_mlx_frozen_kv_mtp_args,
    validate_mlx_frozen_kv_mtp_request,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mlx_ci(est_time=1, suite="stage-a-unit-test-mlx")

_HAS_MLX = importlib.util.find_spec("mlx") is not None


def _target_config():
    text = SimpleNamespace(
        model_type="gemma4_text",
        hidden_size=1536,
        num_hidden_layers=35,
        vocab_size=262144,
    )
    config = SimpleNamespace(
        model_type="gemma4",
        architectures=["Gemma4ForConditionalGeneration"],
        text_config=text,
    )
    return SimpleNamespace(hf_config=config, context_len=2048, is_multimodal=False)


def _assistant_config():
    return {
        "model_type": "gemma4_assistant",
        "architectures": ["Gemma4AssistantForCausalLM"],
        "backbone_hidden_size": 1536,
        "use_ordered_embeddings": True,
        "num_centroids": 2048,
        "centroid_intermediate_top_k": 32,
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 256,
            "num_hidden_layers": 4,
            "vocab_size": 262144,
            "layer_types": [
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
        },
    }


def _server_args(**overrides):
    values = {
        "speculative_algorithm": "FROZEN_KV_MTP",
        "speculative_draft_model_path": "assistant",
        "speculative_draft_model_revision": "assistant-revision",
        "speculative_eagle_topk": 1,
        "speculative_num_steps": 1,
        "speculative_num_draft_tokens": 2,
        "speculative_use_rejection_sampling": False,
        "max_running_requests": 1,
        "disable_overlap_schedule": True,
        "disable_radix_cache": True,
        "chunked_prefill_size": -1,
        "enable_mixed_chunk": False,
        "context_length": 2048,
        "max_total_tokens": 2048,
        "tp_size": 1,
        "dp_size": 1,
        "pp_size": 1,
        "nnodes": 1,
        "enable_dp_attention": False,
        "disaggregation_mode": "null",
        "language_only": False,
        "encoder_only": False,
        "decrypted_draft_config_file": None,
        "get_model_config": _target_config,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestMlxGemma4MTPGuardrails(unittest.TestCase):
    def test_alias_is_mlx_local_and_preserves_generic_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(
                json.dumps(_assistant_config()), encoding="utf-8"
            )
            with (
                mock.patch(
                    "sglang.srt.arg_groups.speculative_hook.use_mlx",
                    return_value=True,
                ),
                mock.patch(
                    "sglang.srt.utils.hf_transformers_utils.get_config",
                    side_effect=AssertionError(
                        "Transformers must not load this config"
                    ),
                ),
            ):
                self.assertEqual(
                    _resolve_speculative_algorithm_alias(
                        "NEXTN",
                        directory,
                        speculative_draft_model_revision="pinned",
                    ),
                    "FROZEN_KV_MTP",
                )
                with self.assertRaisesRegex(ValueError, "EAGLE3"):
                    _resolve_speculative_algorithm_alias("EAGLE3", directory)

                Path(directory, "config.json").write_text(
                    json.dumps({"model_type": "gemma4_assistant"}),
                    encoding="utf-8",
                )
                self.assertEqual(
                    _resolve_speculative_algorithm_alias("NEXTN", directory),
                    "FROZEN_KV_MTP",
                )

        generic = SimpleNamespace(architectures=["OtherDraftForCausalLM"])
        with (
            mock.patch(
                "sglang.srt.arg_groups.speculative_hook.use_mlx", return_value=False
            ),
            mock.patch(
                "sglang.srt.utils.hf_transformers_utils.get_config",
                return_value=generic,
            ) as get_config,
        ):
            self.assertEqual(
                _resolve_speculative_algorithm_alias(
                    "NEXTN", "generic", speculative_draft_model_revision="rev"
                ),
                "EAGLE",
            )
        self.assertEqual(get_config.call_args.kwargs["revision"], "rev")

    def test_platform_dispatch_preserves_cuda_worker(self):
        with mock.patch("sglang.srt.utils.tensor_bridge.use_mlx", return_value=False):
            worker = SpeculativeAlgorithm.FROZEN_KV_MTP.create_worker(_server_args())
        self.assertEqual(worker.__name__, "FrozenKVMTPWorkerV2")

        if _HAS_MLX:
            with mock.patch(
                "sglang.srt.utils.tensor_bridge.use_mlx", return_value=True
            ):
                worker = SpeculativeAlgorithm.FROZEN_KV_MTP.create_worker(
                    _server_args()
                )
            self.assertEqual(worker.__name__, "MlxFrozenKVMTPWorker")

    def test_handler_normalizes_only_safe_defaults(self):
        args = _server_args(
            max_running_requests=None,
            disable_overlap_schedule=False,
            speculative_eagle_topk=None,
            speculative_num_steps=None,
            speculative_num_draft_tokens=None,
        )
        with (
            mock.patch(
                "sglang.srt.arg_groups.speculative_hook.use_mlx", return_value=True
            ),
            mock.patch(
                "sglang.srt.hardware_backend.mlx.spec_config."
                "load_assistant_config_dict",
                return_value=_assistant_config(),
            ),
            self.assertLogs(level="WARNING"),
        ):
            _handle_frozen_kv_mtp(args)
        self.assertEqual(
            (
                args.max_running_requests,
                args.speculative_eagle_topk,
                args.speculative_num_steps,
                args.speculative_num_draft_tokens,
            ),
            (1, 1, 1, 2),
        )
        self.assertTrue(args.disable_overlap_schedule)
        self.assertEqual(args._mlx_gemma4_mtp_assistant_config, _assistant_config())

    def test_server_guardrails(self):
        loader = (
            "sglang.srt.hardware_backend.mlx.spec_config.load_assistant_config_dict"
        )
        with mock.patch(loader, return_value=_assistant_config()):
            validate_mlx_frozen_kv_mtp_args(_server_args())
            validate_mlx_frozen_kv_mtp_args(_server_args(max_total_tokens=4096))

        cases = [
            ({"speculative_draft_model_path": None}, "draft-model-path"),
            ({"disable_radix_cache": False}, "disable-radix-cache"),
            ({"chunked_prefill_size": 64}, "chunked-prefill-size"),
            ({"speculative_eagle_topk": 2}, "eagle-topk"),
            ({"speculative_num_steps": 2}, "num-steps"),
            ({"speculative_num_draft_tokens": 3}, "num-draft-tokens"),
            ({"speculative_use_rejection_sampling": True}, "rejection sampling"),
            ({"max_running_requests": 2}, "max-running-requests"),
            ({"context_length": 2049}, "2,048"),
            ({"max_total_tokens": 1024}, "max-total-tokens"),
            ({"tp_size": 2}, "tp-size"),
            ({"language_only": True}, "encoder disaggregation"),
        ]
        for changes, message in cases:
            with (
                self.subTest(changes=changes),
                mock.patch(loader, return_value=_assistant_config()),
                self.assertRaisesRegex((ValueError, NotImplementedError), message),
            ):
                validate_mlx_frozen_kv_mtp_args(_server_args(**changes))

        wrong_target = _target_config()
        wrong_target.hf_config.text_config.hidden_size = None
        with mock.patch(loader) as load, self.assertRaisesRegex(ValueError, "E2B"):
            validate_mlx_frozen_kv_mtp_args(
                _server_args(get_model_config=lambda: wrong_target)
            )
        load.assert_not_called()

        wrong_assistant = _assistant_config()
        wrong_assistant["num_centroids"] = 4
        with (
            mock.patch(loader, return_value=wrong_assistant),
            self.assertRaisesRegex(ValueError, "ordered embeddings"),
        ):
            validate_mlx_frozen_kv_mtp_args(_server_args())

    def test_request_whitelist(self):
        valid = SamplingParams(
            temperature=0, top_p=0.5, min_p=0.1, stop=["ordinary stop"]
        )
        self.assertIsNone(validate_mlx_frozen_kv_mtp_request(self._request(valid)))

        cases = [
            (SamplingParams(temperature=0.5), {}, False, "temperature=0"),
            (
                SamplingParams(temperature=0, frequency_penalty=0.1),
                {},
                False,
                "penalties",
            ),
            (
                SamplingParams(temperature=0, logit_bias={"1": 1.0}),
                {},
                False,
                "logit bias",
            ),
            (SamplingParams(temperature=0, regex="a+"), {}, False, "constrained"),
            (
                SamplingParams(temperature=0),
                {"return_logprob": True},
                False,
                "logprobs",
            ),
            (
                SamplingParams(temperature=0),
                {"custom_logit_processor": "x"},
                False,
                "custom logits",
            ),
            (SamplingParams(temperature=0), {"session_id": "s"}, False, "sessions"),
            (
                SamplingParams(temperature=0),
                {"input_embeds": [[0.0]]},
                False,
                "token-ID",
            ),
            (SamplingParams(temperature=0), {}, True, "text-only"),
        ]
        for params, overrides, multimodal, message in cases:
            with self.subTest(message=message):
                error = validate_mlx_frozen_kv_mtp_request(
                    self._request(params, **overrides), has_multimodal=multimodal
                )
                self.assertIn(message, error)

    @staticmethod
    def _request(sampling_params, **overrides):
        values = {
            "sampling_params": sampling_params,
            "return_logprob": False,
            "return_hidden_states": False,
            "return_sampling_mask": False,
            "custom_logit_processor": None,
            "session": None,
            "session_id": None,
            "lora_id": None,
            "input_embeds": None,
            "multimodal_inputs": None,
        }
        values.update(overrides)
        return SimpleNamespace(**values)


if __name__ == "__main__":
    unittest.main()
