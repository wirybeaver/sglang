"""Pinned real-model parity gate for the experimental MLX Gemma 4 MTP path."""

from __future__ import annotations

import importlib.util
import os
import platform
import time
import unittest
from contextlib import contextmanager

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mlx_ci(est_time=8, suite="stage-b-e2e-mlx")

TARGET = "mlx-community/gemma-4-e2b-it-4bit"
TARGET_REVISION = "238767527555cb75a05732a84dff5d6ba0dd6809"
ASSISTANT = "mlx-community/gemma-4-E2B-it-assistant-bf16"
ASSISTANT_REVISION = "a7770799b560135ebdbfae8b7f468947415003bc"
CROSS_WINDOW_PROMPT = "blue green amber violet " * 140

_HAS_RUNTIME = (
    platform.system() == "Darwin"
    and platform.machine() == "arm64"
    and importlib.util.find_spec("mlx") is not None
    and importlib.util.find_spec("mlx_vlm.speculative.drafters.gemma4_assistant")
    is not None
)


@unittest.skipUnless(_HAS_RUNTIME, "requires Apple Silicon, MLX, and mlx-vlm")
class TestGemma4MTPMlxCorrectness(CustomTestCase):
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    @contextmanager
    def _server(cls, *, mtp: bool):
        args = [
            "--revision",
            TARGET_REVISION,
            "--served-model-name",
            TARGET,
            "--disable-radix-cache",
            "--disable-overlap-schedule",
            "--chunked-prefill-size",
            "-1",
            "--max-running-requests",
            "1",
            "--context-length",
            "2048",
            "--max-total-tokens",
            "2048",
            "--log-level",
            "warning",
        ]
        if mtp:
            args.extend(
                [
                    "--speculative-algorithm",
                    "FROZEN_KV_MTP",
                    "--speculative-draft-model-path",
                    try_cached_model(ASSISTANT),
                    "--speculative-draft-model-revision",
                    ASSISTANT_REVISION,
                    "--speculative-num-steps",
                    "1",
                    "--speculative-eagle-topk",
                    "1",
                    "--speculative-num-draft-tokens",
                    "2",
                ]
            )
        env = os.environ.copy()
        env["SGLANG_USE_MLX"] = "1"
        process = popen_launch_server(
            try_cached_model(TARGET),
            cls.base_url,
            timeout=max(DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, 600),
            other_args=args,
            env=env,
        )
        try:
            yield
        finally:
            kill_process_tree(process.pid)

    @classmethod
    def _generate(cls, max_new_tokens: int) -> dict:
        response = requests.post(
            f"{cls.base_url}/generate",
            json={
                "text": CROSS_WINDOW_PROMPT,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=180,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _signature(result: dict) -> tuple[list[int], str, str]:
        return (
            result["output_ids"],
            result["text"],
            result["meta_info"]["finish_reason"]["type"],
        )

    @classmethod
    def _spec_state(cls) -> dict:
        response = requests.get(f"{cls.base_url}/server_info", timeout=30)
        response.raise_for_status()
        states = response.json()["internal_states"]
        assert len(states) == 1
        return states[0]["speculative_worker"]

    @classmethod
    def _wait_for_clean_state(cls, timeout: float = 30) -> dict:
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            last = cls._spec_state()
            if all(
                last[key] == 0
                for key in (
                    "active_request_count",
                    "native_request_count",
                    "assistant_request_binding_count",
                )
            ):
                return last
            time.sleep(0.05)
        raise AssertionError(f"MLX MTP request state did not drain: {last}")

    def test_cross_window_exact_token_parity_and_flush(self):
        with self._server(mtp=False):
            target = self._generate(64)
            target_signature = self._signature(target)
            self.assertEqual(len(target_signature[0]), 64)
            self.assertEqual(target_signature[2], "length")
            self.assertGreater(target["meta_info"]["prompt_tokens"], 512)

        with self._server(mtp=True):
            baseline_state = self._wait_for_clean_state()
            result = self._generate(64)
            self.assertEqual(self._signature(result), target_signature)

            meta = result["meta_info"]
            proposed = int(meta["spec_num_proposed_drafts"])
            accepted = int(meta["spec_num_correct_drafts"])
            self.assertGreater(proposed, 0)
            self.assertLessEqual(0, accepted)
            self.assertLessEqual(accepted, proposed)

            state = self._wait_for_clean_state()
            self.assertEqual(state["implementation"], "mlx_gemma4_frozen_kv_mtp")
            self.assertGreaterEqual(state["proposed_tokens"], state["verified_tokens"])
            self.assertGreater(state["verified_tokens"], 0)
            self.assertLessEqual(
                state["accepted_draft_tokens"], state["verified_tokens"]
            )
            self.assertEqual(
                state["verified_tokens"] - baseline_state["verified_tokens"],
                proposed,
            )
            self.assertEqual(
                state["accepted_draft_tokens"]
                - baseline_state["accepted_draft_tokens"],
                accepted,
            )

            generation = state["assistant_generation"]
            flush = requests.post(f"{self.base_url}/flush_cache", timeout=30)
            flush.raise_for_status()
            self.assertTrue(flush.text.startswith("Cache flushed"))
            self.assertEqual(
                self._wait_for_clean_state()["assistant_generation"], generation
            )

            post_flush = self._generate(8)
            self.assertEqual(post_flush["output_ids"], target_signature[0][:8])
            self._wait_for_clean_state()


if __name__ == "__main__":
    unittest.main()
