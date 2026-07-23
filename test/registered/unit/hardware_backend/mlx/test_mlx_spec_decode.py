from __future__ import annotations

import unittest

from sglang.srt.hardware_backend.mlx.spec_decode import (
    build_verify_queries,
    verify_one_draft,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mlx_ci(est_time=1, suite="stage-a-unit-test-mlx")


class TestMlxGreedyVerifier(unittest.TestCase):
    def test_accept_and_reject_are_lossless(self):
        cases = (
            (5, (5, 9), (5, 9), True),
            (5, (6, 9), (6,), False),
        )
        for draft, targets, emitted, accepted in cases:
            with self.subTest(targets=targets):
                decision = verify_one_draft("request", draft, targets)
                self.assertEqual(decision.emitted_token_ids, emitted)
                self.assertEqual(decision.accepted_draft, accepted)
                self.assertEqual(decision.seed_hidden_row_index, len(emitted) - 1)

    def test_query_and_result_shapes_are_strict(self):
        self.assertEqual(build_verify_queries(4, 5), (4, 5))
        for draft, targets in ((-1, (1, 2)), (1, (2,)), (1, (2, -1))):
            with self.subTest(draft=draft, targets=targets):
                with self.assertRaises(ValueError):
                    verify_one_draft("request", draft, targets)


if __name__ == "__main__":
    unittest.main()
