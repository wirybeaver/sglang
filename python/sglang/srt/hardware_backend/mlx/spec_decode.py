"""Pure one-token verification for MLX Frozen-KV MTP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class MlxVerifyDecision:
    """Lossless greedy decision for one proposed token."""

    request_id: str
    emitted_token_ids: tuple[int, ...]
    accepted_draft: bool

    @property
    def committed_query_count(self) -> int:
        return len(self.emitted_token_ids)

    @property
    def seed_hidden_row_index(self) -> int:
        # The sampled mismatch/bonus has not been queried yet. The next seed
        # therefore uses the last query committed to target KV.
        return self.committed_query_count - 1


def build_verify_queries(root_token: int, draft_token: int) -> tuple[int, int]:
    """Return the pending target token followed by the assistant proposal."""

    if root_token < 0 or draft_token < 0:
        raise ValueError("verification token IDs must be non-negative")
    return root_token, draft_token


def verify_one_draft(
    request_id: str,
    draft_token: int,
    target_token_ids: Sequence[int],
) -> MlxVerifyDecision:
    """Accept an exact draft match, otherwise emit the target mismatch."""

    targets = tuple(int(token) for token in target_token_ids)
    if draft_token < 0 or len(targets) != 2 or any(token < 0 for token in targets):
        raise ValueError(
            "one-draft verification requires one non-negative draft and two "
            "non-negative target token IDs"
        )

    accepted = draft_token == targets[0]
    emitted = (draft_token, targets[1]) if accepted else (targets[0],)
    return MlxVerifyDecision(
        request_id=request_id,
        emitted_token_ids=emitted,
        accepted_draft=accepted,
    )
