# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU tensor regression tests; runnable directly without the NPU plugin."""

import ast
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import torch

ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / "vllm_ascend/worker/lwd_cloud/lwd_draft_cache.py"
SPEC = importlib.util.spec_from_file_location("lwd_draft_cache", MODULE)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
LwdDraftCache = module.LwdDraftCache


def output(scheduled, specs=None, finished=(), preempted=(), resumed=(), new=(), computed=None):
    computed = computed or {}
    return NS(
        num_scheduled_tokens=scheduled,
        scheduled_spec_decode_tokens=specs or {},
        finished_req_ids=set(finished),
        preempted_req_ids=set(preempted),
        scheduled_new_reqs=[NS(req_id=r) for r in new],
        scheduled_cached_reqs=NS(
            req_ids=list(computed), num_computed_tokens=list(computed.values()), resumed_req_ids=set(resumed)
        ),
    )


class TestLwdDraftCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Exercise the actual restore method without importing torch_npu/vLLM.
        path = ROOT / "vllm_ascend/worker/lwd_cloud/lwd_cloud_model_runner.py"
        tree = ast.parse(path.read_text())
        runner = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        method = next(n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name == "_lwd_restore_drafts")
        ns = {"torch": torch}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), ns)
        cls.restore = staticmethod(ns[method.name])

    def setUp(self):
        self.cache = LwdDraftCache()
        self.requests = {r: NS(num_computed_tokens=20) for r in ("A", "B", "C")}

    def runner(self, ids, prev, tokens):
        return NS(
            _lwd_draft_cache=self.cache,
            requests=self.requests,
            input_batch=NS(req_ids=ids),
            prev_positions=NS(np=prev),
            input_ids=NS(gpu=torch.tensor(tokens, dtype=torch.int32)),
            pin_memory=False,
            device="cpu",
        )

    def test_logged_failure_across_two_prefills_and_reordered_batch(self):
        drafts = torch.tensor([[24382], [99], [777]])  # includes padded row
        self.cache.preserve(output({"B": 3504}), self.requests, {"A": 0, "C": 1}, drafts)
        drafts.zero_()  # simulate graph replay/staging buffer reuse
        self.cache.preserve(output({"C": 3504}), self.requests, {"B": 0}, torch.tensor([[55]]))
        r = self.runner(["C", "A", "B"], [0, -1, -1], [10, 11, 270, 0, 20, 0])
        self.restore(r, output({"C": 2, "A": 2, "B": 2}, {r: [-1] for r in ("A", "B", "C")}), 3, [2, 4, 6])
        self.assertEqual(r.input_ids.gpu.tolist(), [10, 11, 270, 24382, 20, 55])
        self.assertFalse(self.cache.entries)

    def test_continuous_decode_does_not_clone_or_change_inputs(self):
        class NoClone:
            def detach(self):
                raise AssertionError("continuous decode must not snapshot")

        self.cache.preserve(output({"A": 2}), self.requests, {"A": 0}, NoClone())
        r = self.runner(["A"], [0], [270, 24382])
        self.restore(r, output({"A": 2}, {"A": [-1]}), 1, [2])
        self.assertEqual(r.input_ids.gpu.tolist(), [270, 24382])

    def test_trimmed_multi_draft_and_single_use(self):
        self.cache.preserve(output({}), self.requests, {"A": 0}, torch.tensor([[1, 2, 3]]))
        r = self.runner(["A"], [-1], [99, 0, 0])
        self.restore(r, output({"A": 3}, {"A": [-1, -1]}), 1, [3])
        self.assertEqual(r.input_ids.gpu.tolist(), [99, 1, 2])
        self.assertFalse(self.cache.entries)

    def test_scheduler_drops_drafts(self):
        self.cache.preserve(output({}), self.requests, {"A": 0}, torch.tensor([[1]]))
        r = self.runner(["A"], [-1], [99])
        self.restore(r, output({"A": 1}), 1, [1])
        self.assertEqual(r.input_ids.gpu.tolist(), [99])
        self.assertFalse(self.cache.entries)

    def test_lifecycle_invalidations(self):
        cases = [
            dict(finished=["A"]),
            dict(preempted=["A"]),
            dict(resumed=["A"]),
            dict(new=["A"]),
            dict(computed={"A": 0}),
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                self.cache.preserve(output({}), self.requests, {"A": 0}, torch.tensor([[1]]))
                self.cache.preserve(output({}, **kwargs), self.requests, {"A": 0}, torch.tensor([[2]]))
                self.assertNotIn("A", self.cache.entries)

    def test_request_id_reuse_and_orphan(self):
        self.cache.preserve(output({}), self.requests, {"A": 0}, torch.tensor([[1]]))
        self.assertIsNone(self.cache.take("A", NS(num_computed_tokens=20), 1))
        self.cache.preserve(output({}), self.requests, {"A": 0}, torch.tensor([[1]]))
        del self.requests["A"]
        self.cache.preserve(output({}), self.requests, {}, None)
        self.assertFalse(self.cache.entries)

    def test_invalid_length_does_not_fill_placeholder(self):
        self.cache.preserve(output({}), self.requests, {"A": 0}, torch.tensor([[1]]))
        r = self.runner(["A"], [-1], [99, 0, 0])
        with self.assertRaisesRegex(RuntimeError, "snapshot is invalid"):
            self.restore(r, output({"A": 3}, {"A": [-1, -1]}), 1, [3])
        self.assertEqual(r.input_ids.gpu.tolist(), [99, 0, 0])

    def test_runner_hooks_snapshot_before_removal_and_restore_after_native_copy(self):
        path = ROOT / "vllm_ascend/worker/lwd_cloud/lwd_cloud_model_runner.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        cls.body = [
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef)
            and n.name in {"_update_states", "_prepare_input_ids", "_lwd_restore_drafts"}
        ]

        class Base:
            def _update_states(self, scheduler_output):
                self.input_batch.prev_req_id_to_index.clear()
                self._draft_token_ids.zero_()
                return "updated"

            def _prepare_input_ids(self, *args):
                self.input_ids.gpu.copy_(torch.tensor([270, 0], dtype=torch.int32))

        ns = {"NPUModelRunner": Base, "torch": torch}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), ns)
        r = ns[cls.name]()
        r.__dict__.update(self.runner(["A"], [-1], [0, 0]).__dict__)
        r.input_batch.prev_req_id_to_index = {"A": 0}
        r._draft_token_ids = torch.tensor([[24382]])
        self.assertEqual(r._update_states(output({"B": 3504})), "updated")
        r._prepare_input_ids(output({"A": 2}, {"A": [-1]}), 1, 2, [2])
        self.assertEqual(r.input_ids.gpu.tolist(), [270, 24382])

    def test_snapshot_cache_disables_legacy_stash_without_disabling_counts(self):
        path = ROOT / "vllm_ascend/worker/model_runner_v1.py"
        tree = ast.parse(path.read_text())
        propose = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)
            and n.name == "propose_draft_token_ids"
            and any(isinstance(child, ast.If) and "_lwd_draft_stash" in ast.unparse(child) for child in n.body)
        )
        stash = next(n for n in propose.body if isinstance(n, ast.If) and "_lwd_draft_stash" in ast.unparse(n))
        code = compile(ast.Module(body=[stash], type_ignores=[]), str(path), "exec")
        for cache in (LwdDraftCache(), None):
            with self.subTest(snapshot_enabled=cache is not None):
                drafts = torch.tensor([[123]])
                runner = NS(
                    _lwd_spec_persist_enabled=True,
                    _lwd_draft_cache=cache,
                    _draft_token_ids=drafts,
                    _lwd_draft_stash={},
                    input_batch=NS(req_ids=["A"]),
                    requests={"A": object()},
                )
                exec(code, {"self": runner, "torch": torch})
                self.assertTrue(runner._lwd_spec_persist_enabled)
                if cache is not None:
                    self.assertFalse(runner._lwd_draft_stash)
                else:
                    self.assertEqual(runner._lwd_draft_stash["A"].tolist(), [123])


if __name__ == "__main__":
    unittest.main()
