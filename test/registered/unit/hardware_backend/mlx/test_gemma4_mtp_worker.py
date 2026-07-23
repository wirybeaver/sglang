from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.mem_cache.allocation import assign_req_to_token_pool_func
from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mlx_ci(est_time=2, suite="stage-a-unit-test-mlx")

_HAS_MLX = importlib.util.find_spec("mlx") is not None

if _HAS_MLX:
    from registered.unit.hardware_backend.mlx.utils import (
        assert_native_cache_equal,
        build_runner,
        cache_logical_length,
        reference_tokens,
        tiny_gemma4,
    )

    from sglang.srt.hardware_backend.mlx.kv_cache.native_transaction import (
        clone_native_cache,
    )
    from sglang.srt.hardware_backend.mlx.speculative_worker import (
        MlxFrozenKVMTPDraftInput,
        MlxFrozenKVMTPWorker,
    )
    from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker
    from sglang.srt.model_executor.forward_batch_info import ForwardMode


class _ScriptedRuntime:
    def __init__(self):
        self.bindings: set[str] = set()

    def release_request(self, request_id: str) -> None:
        self.bindings.discard(request_id)

    def clear_request_bindings(self) -> None:
        self.bindings.clear()

    @property
    def request_binding_count(self) -> int:
        return len(self.bindings)


class _ScriptedProposer:
    def __init__(self, runtime, outcomes, runner):
        self.runtime = runtime
        self.outcomes = list(outcomes)
        self.runner = runner
        self.calls = []

    def propose_one(self, request_id, seed, cache):
        self.runtime.bindings.add(request_id)
        self.calls.append(
            (
                seed.token_id,
                cache_logical_length(cache),
                cache is self.runner._req_caches[request_id],
            )
        )
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _request(prompt, *, finish_on_update=False):
    request = SimpleNamespace(
        rid="request",
        req_pool_idx=0,
        prefix_indices=torch.empty((0,), dtype=torch.long),
        get_fill_ids=lambda: list(prompt),
        mamba_last_track_seqlen=None,
        kv_committed_len=len(prompt),
        is_retracted=False,
        grammar=None,
        spec_verify_ct=0,
        spec_num_correct_drafts=0,
        spec_num_block_accept_tokens=0,
        spec_num_cap_tokens=0,
        update_spec_correct_drafts_histogram=mock.Mock(),
        update_spec_cap_lens_histogram=mock.Mock(),
        output_ids=[],
        finished_reason=None,
        inflight_middle_chunks=0,
        time_stats=mock.Mock(),
        require_reasoning=False,
        return_routed_experts=False,
        return_sampling_mask=False,
        return_hidden_states=False,
    )
    finished = False

    def is_finished():
        return finished

    def update_finish_state():
        nonlocal finished
        if finish_on_update:
            finished = True
            request.finished_reason = FINISH_MATCHED_TOKEN(matched=7)

    request.finished = is_finished
    request.update_finish_state = update_finish_state
    return request


def _prefill_batch(request, prompt):
    return SimpleNamespace(
        reqs=[request],
        forward_mode=ForwardMode.EXTEND,
        input_ids=torch.tensor(prompt, dtype=torch.long),
        out_cache_loc=torch.arange(len(prompt), dtype=torch.long),
        extend_lens=[len(prompt)],
        seq_lens=torch.tensor([len(prompt)], dtype=torch.int32),
        spec_info=None,
        return_logprob=False,
        decoding_reqs=None,
        prefill_stats=mock.Mock(),
        dp_cooperation_info=None,
    )


def _decode_batch(request, prefill_batch, draft_input):
    return SimpleNamespace(
        reqs=[request],
        forward_mode=ForwardMode.DECODE,
        seq_lens=prefill_batch.seq_lens.clone(),
        spec_info=draft_input,
        has_grammar=False,
    )


def _processor(worker):
    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=DisaggregationMode.NULL,
        enable_overlap=False,
        enable_overlap_mlx=False,
        server_args=mock.Mock(),
        model_config=mock.Mock(),
        token_to_kv_pool_allocator=mock.Mock(),
        tree_cache=mock.Mock(),
        hisparse_coordinator=None,
        req_to_token_pool=mock.Mock(),
        decode_offload_manager=None,
        metrics_collector=mock.Mock(),
        metrics_reporter=mock.Mock(),
        draft_worker=worker,
        model_worker=worker,
        logprob_result_processor=mock.Mock(),
        output_streamer=mock.Mock(),
        abort_request=mock.Mock(),
    )


@unittest.skipUnless(_HAS_MLX, "requires MLX")
class TestMlxFrozenKVMTPWorker(unittest.TestCase):
    def _make_worker(self, outcomes):
        model = tiny_gemma4()
        runner = build_runner(model)
        target_worker = MlxTpModelWorker.__new__(MlxTpModelWorker)
        target_worker._mlx_runner = runner
        target_worker._mlx_active_rids = set()
        target_worker._mlx_pool_initialized = True
        target_worker._model_runner = SimpleNamespace()
        runtime = _ScriptedRuntime()
        loader = SimpleNamespace(load=mock.Mock(return_value=runtime))
        server_args = SimpleNamespace(
            speculative_draft_model_path="unused",
            speculative_draft_model_revision="pinned",
        )
        with mock.patch(
            "sglang.srt.hardware_backend.mlx.speculative_worker."
            "Gemma4MTPAssistantLoader",
            return_value=loader,
        ):
            worker = MlxFrozenKVMTPWorker(
                server_args=server_args,
                gpu_id=0,
                ps=SimpleNamespace(),
                nccl_port=0,
                target_worker=target_worker,
            )
        proposer = _ScriptedProposer(runtime, outcomes, runner)
        worker._proposer = proposer
        return model, runner, runtime, proposer, worker

    def test_accept_reject_after_rotation_and_scheduler_owns_commit(self):
        prompt = list(range(1, 18))
        for accepted in (True, False):
            with self.subTest(accepted=accepted):
                model, runner, _runtime, proposer, worker = self._make_worker([])
                expected = reference_tokens(model, prompt, steps=3)
                draft = (
                    expected[1]
                    if accepted
                    else (expected[1] + 1) % model.args.vocab_size
                )
                proposer.outcomes.extend([draft, 0])
                request = _request(prompt)
                prefill_batch = _prefill_batch(request, prompt)
                prefill = worker.forward_batch_generation(prefill_batch)
                self.assertEqual(prefill.next_token_ids.tolist(), expected[:1])
                self.assertIsNone(prefill.accept_lens)
                self.assertIsInstance(
                    prefill.next_draft_input, MlxFrozenKVMTPDraftInput
                )

                decode = worker.forward_batch_generation(
                    _decode_batch(request, prefill_batch, prefill.next_draft_input)
                )
                emitted = expected[1:3] if accepted else expected[1:2]
                padded = emitted + [-1] * (2 - len(emitted))
                self.assertEqual(decode.next_token_ids.tolist(), padded)
                self.assertEqual(decode.accept_lens.tolist(), [len(emitted)])
                self.assertEqual(decode.new_seq_lens.tolist(), [17 + len(emitted)])
                self.assertEqual(decode.speculative_num_draft_tokens, 2)
                self.assertEqual(
                    int(decode.next_draft_input.bonus_tokens[0]), emitted[-1]
                )
                self.assertEqual(
                    proposer.calls,
                    [
                        (expected[0], 17, True),
                        (emitted[-1], 17 + len(emitted), False),
                    ],
                )
                self.assertEqual(
                    runner._req_token_ids[request.rid], prompt + [expected[0]] + emitted
                )
                self.assertEqual(
                    len(runner._req_token_ids[request.rid]),
                    cache_logical_length(runner._req_caches[request.rid]) + 1,
                )

                self.assertEqual(request.kv_committed_len, 17)
                visible = _processor(worker)._resolve_spec_v2_tokens(
                    decode,
                    _decode_batch(request, prefill_batch, prefill.next_draft_input),
                )
                self.assertEqual(visible, [emitted])
                self.assertEqual(request.kv_committed_len, 17 + len(emitted))
                self.assertEqual(request.spec_verify_ct, 1)
                self.assertEqual(request.spec_num_correct_drafts, len(emitted) - 1)

    def test_proposal_failure_does_not_publish_prepared_cache(self):
        prompt = list(range(1, 18))
        model, runner, runtime, proposer, worker = self._make_worker([])
        expected = reference_tokens(model, prompt, steps=3)
        proposer.outcomes.extend([expected[1], RuntimeError("proposal failed")])
        request = _request(prompt)
        prefill_batch = _prefill_batch(request, prompt)
        prefill = worker.forward_batch_generation(prefill_batch)
        reference_cache = clone_native_cache(runner._req_caches[request.rid])
        reference_history = list(runner._req_token_ids[request.rid])

        with self.assertRaisesRegex(RuntimeError, "proposal failed"):
            worker.forward_batch_generation(
                _decode_batch(request, prefill_batch, prefill.next_draft_input)
            )
        self.assertEqual(proposer.calls[-1][1:], (19, False))
        assert_native_cache_equal(
            self, runner._req_caches[request.rid], reference_cache
        )
        self.assertEqual(runner._req_token_ids[request.rid], reference_history)
        self.assertEqual(runtime.request_binding_count, 0)

        proposer.outcomes.append(0)
        retried = worker.forward_batch_generation(
            _decode_batch(request, prefill_batch, prefill.next_draft_input)
        )
        self.assertEqual(retried.next_token_ids.tolist(), expected[1:3])

    def test_prefill_natural_finish_releases_all_request_state(self):
        prompt = [1, 2, 3]
        _model, runner, runtime, _proposer, worker = self._make_worker([0])
        request = _request(prompt, finish_on_update=True)
        prefill = worker.forward_batch_generation(_prefill_batch(request, prompt))
        self.assertTrue(runner.has_request(request.rid))
        self.assertEqual(runtime.request_binding_count, 1)

        with (
            mock.patch.object(
                worker, "note_request_finished", wraps=worker.note_request_finished
            ) as note_finished,
            mock.patch.object(
                worker,
                "prepare_for_kv_cache_release",
                wraps=worker.prepare_for_kv_cache_release,
            ) as prepare_release,
            mock.patch(
                "sglang.srt.managers.scheduler_components.batch_result_processor."
                "release_kv_cache"
            ) as release,
        ):
            _processor(worker).process_batch_result_prefill(
                _prefill_batch(request, prompt), prefill
            )

        note_finished.assert_called_once_with(rid=request.rid, natural_stop=True)
        prepare_release.assert_called_once_with(request)
        release.assert_called_once()
        self.assertFalse(runner.has_request(request.rid))
        self.assertEqual(runtime.request_binding_count, 0)
        self.assertNotIn(request.rid, worker._active_rids)


class TestMlxCpuBookkeeping(unittest.TestCase):
    def test_cpu_req_to_token_assignment_does_not_launch_triton(self):
        req_to_token = torch.zeros((3, 8), dtype=torch.int32)
        with mock.patch("sglang.srt.mem_cache.allocation._is_cpu", False):
            assign_req_to_token_pool_func(
                req_pool_indices=torch.tensor([1, 2], dtype=torch.int32),
                req_to_token=req_to_token,
                start_offset=torch.tensor([2, 1], dtype=torch.int32),
                end_offset=torch.tensor([4, 4], dtype=torch.int32),
                out_cache_loc=torch.tensor([10, 11, 20, 21, 22]),
                batch_size=2,
            )
        self.assertEqual(req_to_token[0].tolist(), [0] * 8)
        self.assertEqual(req_to_token[1, 2:4].tolist(), [10, 11])
        self.assertEqual(req_to_token[2, 1:4].tolist(), [20, 21, 22])


if __name__ == "__main__":
    unittest.main()
