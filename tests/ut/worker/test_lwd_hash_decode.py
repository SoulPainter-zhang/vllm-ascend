# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU regression tests for graph-buffer reuse across prompt/decode batches."""

import ast
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / "vllm_ascend/worker/lwd_hash/lwd_hash_routing.py"
SPEC = importlib.util.spec_from_file_location("lwd_hash_routing", PATH)
routing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(routing)


class TestHashDecodeStaging(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import the actual method without loading the NPU plugin.
        path = ROOT / "vllm_ascend/worker/lwd_cloud/lwd_cloud_model_runner.py"
        tree = ast.parse(path.read_text())
        runner = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        method = next(n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name == "_lwd_prepare_hash_batch")
        ns = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), ns)
        cls.prepare = staticmethod(ns[method.name])

    def setUp(self):
        self.state = routing.LwdHashRoutingState(3, 2, 8)
        self.remote = torch.full((4, 3, 2), 7, dtype=torch.int32)
        self.state.add_chunk("p", 4, 0, self.remote)
        self.runner = SimpleNamespace(
            lwd_hash_state=self.state,
            _lwd_hash_experts=torch.zeros(8, 3, 2, dtype=torch.int32),
            _lwd_hash_prompt_mask=torch.zeros(8, dtype=torch.bool),
        )

    def prepare_batch(self, ids, counts, computed, lengths):
        self.runner.input_batch = SimpleNamespace(
            num_reqs=len(ids),
            req_ids=ids,
            num_computed_tokens_cpu=computed,
            num_prompt_tokens=lengths,
        )
        self.prepare(self.runner, counts)

    def test_prefill_to_decode_preserves_buffers_and_uses_local_routes(self):
        self.prepare_batch(["p"], [4], [0], [4])
        r = self.runner
        pointers = (r._lwd_hash_experts.data_ptr(), r._lwd_hash_prompt_mask.data_ptr())
        experts = r._lwd_hash_experts.clone()
        with patch.object(self.state, "build_batch", side_effect=AssertionError("decode staging")):
            self.prepare_batch(["p"], [2], [4], [4])
        self.assertFalse(r._lwd_hash_prompt_mask.any())
        self.assertTrue(torch.equal(experts, r._lwd_hash_experts))
        self.assertEqual(r._lwd_hash_step_tokens, 2)
        self.assertEqual(pointers, (r._lwd_hash_experts.data_ptr(), r._lwd_hash_prompt_mask.data_ptr()))
        table = torch.arange(16, dtype=torch.int32).reshape(8, 2) % 8
        ids = torch.tensor([2, 3, 0, 0])  # includes graph padding
        for layer in range(3):
            actual = routing.merge_hash_expert_ids(
                table,
                ids,
                r._lwd_hash_experts[:4, layer],
                r._lwd_hash_prompt_mask[:4],
            )
            self.assertTrue(torch.equal(actual, table[ids]))

    def test_decode_to_mixed_batch_restores_remote_routes(self):
        self.prepare_batch(["d"], [2], [8], [4])
        self.prepare_batch(["d", "p"], [2, 2], [8, 2], [4, 4])
        r = self.runner
        self.assertEqual(r._lwd_hash_prompt_mask.tolist(), [False, False, True, True] + [False] * 4)
        self.assertTrue(torch.equal(r._lwd_hash_experts[2:4], self.remote[2:4]))
        self.assertEqual(r._lwd_hash_step_tokens, 4)

    def test_unscheduled_prompt_does_not_require_remote_data(self):
        with patch.object(self.state, "build_batch", side_effect=AssertionError("decode staging")):
            self.prepare_batch(["missing", "d"], [0, 2], [0, 8], [4, 4])
        self.assertEqual(self.runner._lwd_hash_step_tokens, 2)

    def test_missing_scheduled_prompt_still_fails(self):
        with self.assertRaisesRegex(ValueError, "Missing LWD Hash MoE"):
            self.prepare_batch(["missing"], [2], [0], [4])


if __name__ == "__main__":
    unittest.main()
