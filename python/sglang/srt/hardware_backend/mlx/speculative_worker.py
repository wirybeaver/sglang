"""Synchronous MLX-native Gemma 4 Frozen-KV MTP worker."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.hardware_backend.mlx.gemma4_mtp import (
    Gemma4MTPAssistantLoader,
    Gemma4MTPAssistantRuntime,
)
from sglang.srt.hardware_backend.mlx.spec_decode import (
    build_verify_queries,
    verify_one_draft,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.mlx.model_adapter import MlxTargetSeed
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.server_args import ServerArgs


class MlxFrozenKVMTPDraftInput(SpecInput):
    """One-draft CPU relay; hidden states and target KV remain worker-owned."""

    def __init__(
        self,
        *,
        request_ids: tuple[str, ...],
        draft_token_ids: torch.Tensor,
        bonus_tokens: torch.Tensor,
    ) -> None:
        super().__init__(SpecInputType.FROZEN_KV_MTP_DRAFT)
        size = len(request_ids)
        draft_token_ids = draft_token_ids.to(dtype=torch.long, device="cpu")
        bonus_tokens = bonus_tokens.to(dtype=torch.long, device="cpu")
        for name, value in (
            ("draft_token_ids", draft_token_ids),
            ("bonus_tokens", bonus_tokens),
        ):
            if value.ndim != 1 or len(value) != size:
                raise ValueError(f"{name} must be a flat tensor of length {size}")
        if bool(torch.any(draft_token_ids < 0)) or bool(torch.any(bonus_tokens < 0)):
            raise ValueError("MLX MTP draft and bonus tokens must be non-negative")

        self.request_ids = tuple(request_ids)
        self.draft_token_ids = draft_token_ids
        self.bonus_tokens = bonus_tokens
        self.num_tokens_per_req = 1
        self.num_tokens_for_logprob_per_req = 1
        self.future_indices = None
        self.dsa_topk_indices = None

    @property
    def draft_token(self) -> torch.Tensor:
        return self.draft_token_ids

    def filter_batch(
        self,
        new_indices: torch.Tensor,
        has_been_filtered: bool = True,
        new_indices_cpu: Optional[list[int]] = None,
    ) -> None:
        del has_been_filtered
        indices_cpu = (
            list(new_indices_cpu)
            if new_indices_cpu is not None
            else [int(index) for index in new_indices.to("cpu").tolist()]
        )
        cpu_indices = torch.tensor(indices_cpu, dtype=torch.long, device="cpu")
        self.request_ids = tuple(self.request_ids[index] for index in indices_cpu)
        self.draft_token_ids = self.draft_token_ids[cpu_indices]
        self.bonus_tokens = self.bonus_tokens[cpu_indices]
        if self.future_indices is not None:
            self.future_indices = self.future_indices[new_indices]

    def merge_batch(self, other: MlxFrozenKVMTPDraftInput) -> None:
        if not isinstance(other, MlxFrozenKVMTPDraftInput):
            raise TypeError("cannot merge a non-MLX Frozen-KV draft input")
        self.request_ids += other.request_ids
        self.draft_token_ids = torch.cat((self.draft_token_ids, other.draft_token_ids))
        self.bonus_tokens = torch.cat((self.bonus_tokens, other.bonus_tokens))
        if self.future_indices is not None or other.future_indices is not None:
            if self.future_indices is None or other.future_indices is None:
                raise ValueError("future-index relay state must agree when merging")
            self.future_indices = torch.cat((self.future_indices, other.future_indices))


class MlxGemma4MTPProposer:
    def __init__(self, runtime: Gemma4MTPAssistantRuntime):
        self.runtime = runtime

    @staticmethod
    def needs_target_hidden_states() -> bool:
        return True

    def propose_one(self, request_id: str, seed: MlxTargetSeed, cache) -> int:
        self.runtime.release_request(request_id)
        view = self.runtime.bind_request(request_id, cache)
        try:
            return self.runtime.propose_one(seed, view)
        except BaseException:
            self.runtime.release_request(request_id)
            raise


class MlxFrozenKVMTPWorker(BaseSpecWorker):
    """One-proposal, greedy, BS=1 native-cache spec-v2 orchestration."""

    def __init__(self, server_args: ServerArgs, gpu_id, ps, nccl_port, target_worker):
        del gpu_id, ps, nccl_port
        if not hasattr(target_worker, "_mlx_runner"):
            raise TypeError("MlxFrozenKVMTPWorker requires MlxTpModelWorker")
        if not target_worker._mlx_runner.native_cache_fallback:
            raise ValueError("MLX Frozen-KV MTP requires native Gemma 4 target caches")

        self.server_args = server_args
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self._draft_worker = None
        self.speculative_num_draft_tokens = 2
        self._native_runner = target_worker._mlx_runner
        self._target_adapter = self._native_runner._get_target_adapter()
        self._assistant_loader = Gemma4MTPAssistantLoader(self._native_runner.model)
        self._assistant_runtime = self._assistant_loader.load(
            server_args.speculative_draft_model_path,
            revision=getattr(server_args, "speculative_draft_model_revision", None),
            config=getattr(server_args, "_mlx_gemma4_mtp_assistant_config", None),
        )
        self._proposer = MlxGemma4MTPProposer(self._assistant_runtime)
        self._active_rids: set[str] = set()

    @property
    def draft_worker(self):
        return None

    def carries_draft_hidden_states(self) -> bool:
        return False

    def get_draft_kv_pool(self):
        return None

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ) -> None:
        del memory_pool_config
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator

    def init_attention_backends(self) -> None:
        return None

    def init_cuda_graphs(self) -> None:
        return None

    def _cleanup_departed(self, current_rids: set[str]) -> None:
        for rid in self._active_rids - current_rids:
            self._assistant_runtime.release_request(rid)
            if self._native_runner.has_request(rid):
                self._native_runner.remove_request(rid)
            self._target_worker._mlx_active_rids.discard(rid)
        self._active_rids.intersection_update(current_rids)

    @staticmethod
    def _padded_tokens(emitted: tuple[int, ...]) -> torch.Tensor:
        if len(emitted) not in (1, 2) or any(token < 0 for token in emitted):
            raise ValueError("MLX MTP emitted width must be one or two")
        return torch.tensor(
            list(emitted) + [-1] * (2 - len(emitted)),
            dtype=torch.long,
            device="cpu",
        )

    @staticmethod
    def _make_draft_input(
        *,
        request_id: str,
        proposal: int,
        bonus_token: int,
    ) -> MlxFrozenKVMTPDraftInput:
        return MlxFrozenKVMTPDraftInput(
            request_ids=(request_id,),
            draft_token_ids=torch.tensor([proposal], dtype=torch.long),
            bonus_tokens=torch.tensor([bonus_token], dtype=torch.long),
        )

    def _propose_from_output(
        self,
        request_id: str,
        target_output,
        hidden_row_index: int,
        emitted_token_id: int,
        cache,
    ) -> int:
        seed = self._target_adapter.make_seed(
            target_output,
            hidden_row_index=hidden_row_index,
            emitted_token_id=emitted_token_id,
        )
        proposal = int(self._proposer.propose_one(request_id, seed, cache))
        if proposal < 0 or proposal >= self._target_adapter.vocab_size:
            raise ValueError("assistant proposal is outside the target vocabulary")
        return proposal

    def _forward_prefill(self, batch: ScheduleBatch) -> GenerationBatchResult:
        if len(batch.reqs) != 1:
            raise ValueError("MLX Frozen-KV MTP MVP supports batch size one")
        target_result, target_output = (
            self._target_worker.forward_batch_generation_mtp_prefill(batch)
        )
        request = batch.reqs[0]
        token = int(target_result.next_token_ids[0])
        try:
            proposal = self._propose_from_output(
                request.rid,
                target_output,
                hidden_row_index=target_output.hidden_states.shape[1] - 1,
                emitted_token_id=token,
                cache=self._native_runner._req_caches[request.rid],
            )
        except BaseException:
            self._assistant_runtime.release_request(request.rid)
            self._native_runner.remove_request(request.rid)
            self._target_worker._mlx_active_rids.discard(request.rid)
            raise

        new_seq_lens = batch.seq_lens.clone()
        target_result.next_draft_input = self._make_draft_input(
            request_id=request.rid,
            proposal=proposal,
            bonus_token=token,
        )
        target_result.new_seq_lens = new_seq_lens
        target_result.speculative_num_draft_tokens = 2
        return target_result

    def _forward_decode(self, batch: ScheduleBatch) -> GenerationBatchResult:
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        if len(batch.reqs) != 1:
            raise ValueError("MLX Frozen-KV MTP MVP supports batch size one")
        draft_input = batch.spec_info
        if not isinstance(draft_input, MlxFrozenKVMTPDraftInput):
            raise TypeError("MLX Frozen-KV MTP decode requires its native draft input")
        request = batch.reqs[0]
        if draft_input.request_ids != (request.rid,):
            raise ValueError("MLX MTP draft handoff request ordering is stale")
        root = int(self._native_runner._req_token_ids[request.rid][-1])
        if int(draft_input.bonus_tokens[0]) != root:
            raise ValueError("MLX MTP draft handoff bonus token is stale")
        draft = int(draft_input.draft_token_ids[0])

        pending = None
        try:
            pending = self._native_runner.verify_start(
                request.rid, build_verify_queries(root, draft)
            )
            target_ids = self._native_runner.verify_materialize(pending)
            decision = verify_one_draft(request.rid, draft, target_ids)
            candidate_cache = self._native_runner.verify_prepare(pending, decision)
            proposal = self._propose_from_output(
                request.rid,
                pending.target_output,
                hidden_row_index=decision.seed_hidden_row_index,
                emitted_token_id=decision.emitted_token_ids[-1],
                cache=candidate_cache,
            )
            self._native_runner.verify_commit(pending, decision)
        except BaseException:
            if pending is not None:
                self._native_runner.verify_abort(pending)
            self._assistant_runtime.release_request(request.rid)
            raise

        emitted = decision.emitted_token_ids
        padded = self._padded_tokens(emitted)
        accept_lens = torch.tensor([len(emitted)], dtype=torch.int32, device="cpu")
        new_seq_lens = batch.seq_lens + accept_lens.to(batch.seq_lens.device)
        return GenerationBatchResult(
            logits_output=LogitsProcessorOutput(next_token_logits=None),
            next_token_ids=padded,
            can_run_cuda_graph=False,
            accept_lens=accept_lens,
            new_seq_lens=new_seq_lens,
            next_draft_input=self._make_draft_input(
                request_id=request.rid,
                proposal=proposal,
                bonus_token=emitted[-1],
            ),
            speculative_num_draft_tokens=2,
        )

    @staticmethod
    def _forward_idle(batch: ScheduleBatch) -> GenerationBatchResult:
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        empty_long = torch.empty((0,), dtype=torch.long, device="cpu")
        empty_int = torch.empty((0,), dtype=torch.int32, device="cpu")
        empty_seq = batch.seq_lens.clone()
        return GenerationBatchResult(
            logits_output=LogitsProcessorOutput(next_token_logits=None),
            next_token_ids=empty_long,
            accept_lens=empty_int,
            new_seq_lens=empty_seq,
            next_draft_input=MlxFrozenKVMTPDraftInput(
                request_ids=(),
                draft_token_ids=empty_long,
                bonus_tokens=empty_long,
            ),
            speculative_num_draft_tokens=2,
            can_run_cuda_graph=False,
        )

    def forward_batch_generation(self, batch: ScheduleBatch, on_publish=None, **kwargs):
        del kwargs
        current_rids = {request.rid for request in batch.reqs}
        self._cleanup_departed(current_rids)
        if batch.forward_mode.is_idle():
            result = self._forward_idle(batch)
        elif batch.forward_mode.is_extend():
            result = self._forward_prefill(batch)
        elif batch.forward_mode.is_decode():
            result = self._forward_decode(batch)
        else:
            raise ValueError(f"MLX Frozen-KV MTP does not support {batch.forward_mode}")
        self._active_rids = current_rids
        if on_publish is not None and result.new_seq_lens is not None:
            on_publish(result.new_seq_lens)
        return result

    def note_request_finished(self, *, rid: str, natural_stop: bool) -> None:
        del natural_stop
        self._assistant_runtime.release_request(rid)
        self._active_rids.discard(rid)

    def prepare_for_kv_cache_release(self, req) -> None:
        self._assistant_runtime.release_request(req.rid)
        self._active_rids.discard(req.rid)
        self._target_worker.prepare_for_kv_cache_release(req)

    def clear_cache_pool(self) -> None:
        self._target_worker.clear_cache_pool()
        self._assistant_runtime.clear_request_bindings()
        self._active_rids.clear()

    @staticmethod
    def _unsupported_weight_update():
        return (
            False,
            "MLX Frozen-KV MTP requires an idle restart to update pinned target "
            "and assistant checkpoints.",
        )

    def update_weights_from_disk(self, recv_req):
        del recv_req
        return self._unsupported_weight_update()

    def update_weights_from_ipc(self, recv_req):
        del recv_req
        return self._unsupported_weight_update()

    def update_weights_from_tensor(self, recv_req):
        del recv_req
        return self._unsupported_weight_update()
