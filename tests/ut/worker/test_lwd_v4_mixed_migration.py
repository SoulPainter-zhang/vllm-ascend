# SPDX-License-Identifier: Apache-2.0
"""CPU checks for V4 transport and mixed batches, without importing the NPU plugin.

Run this file directly with a Python environment containing torch and numpy.
These checks do not validate HCCL streams or NPU graph execution.
"""

import ast
import importlib.util
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3] / "vllm_ascend"
SPEC = importlib.util.spec_from_file_location("routing", ROOT / "worker/lwd_hash/lwd_hash_routing.py")
routing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(routing)


def load_methods(path, names, base=object, **globals_):
    """Compile real methods with a stub base to isolate hardware dependencies."""
    tree = ast.parse((ROOT / path).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    cls.bases = [ast.Name(id="Base", ctx=ast.Load())]
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(Base=base, torch=torch, np=np, **globals_)
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    exec(compile(module, str(ROOT / path), "exec"), namespace)
    return namespace[cls.name]


RUNNER = "worker/lwd_cloud/lwd_cloud_model_runner.py"


class TestHashTransport(unittest.TestCase):
    def test_lossless_ids_above_bf16_integer_precision(self):
        embeds = torch.arange(12).reshape(3, 4).to(torch.bfloat16)
        ids = torch.tensor([0, 257, 511, 1023, 4095, 65535]).reshape(3, 1, 2)
        payload = routing.pack_hash_payload(embeds, ids)
        actual_embeds, actual_ids = routing.unpack_hash_payload(payload, 3, 4, 1, 2)
        self.assertTrue(torch.equal(actual_embeds, embeds))
        self.assertTrue(torch.equal(actual_ids, ids))
        with self.assertRaises(ValueError):
            routing.unpack_hash_payload(payload[:-1], 3, 4, 1, 2)

    def test_reordered_mixed_requests_and_chunk_offsets(self):
        state = routing.LwdHashRoutingState(1, 2, 1024)
        a = torch.tensor([[[257, 511]], [[7, 8]]])
        b = torch.tensor([[[9, 10]], [[11, 12]]])
        state.add_chunk("a", 4, 2, a)
        state.add_chunk("b", 2, 0, b)
        batch, mask = state.build_batch(["decode", "b", "a"], [2, 2, 2], [9, 0, 2], [4, 2, 4])
        self.assertEqual(mask.tolist(), [False, False, True, True, True, True])
        self.assertTrue(torch.equal(batch[2:4], b))
        self.assertTrue(torch.equal(batch[4:6], a))
        table = torch.arange(32).reshape(16, 2).int()
        # Prompt placeholders must never be used as vocabulary indices.
        actual = routing.merge_hash_expert_ids(table, torch.tensor([2, 3, 99999, -77, -1, 99999]), batch[:, 0], mask)
        self.assertTrue(torch.equal(actual[:2], table[[2, 3]]))
        self.assertTrue(torch.equal(actual[2:], batch[2:, 0]))
        state.discard("a")
        with self.assertRaisesRegex(ValueError, "Missing LWD"):
            state.build_batch(["a"], [2], [2], [4])

    def test_invalid_or_missing_chunks_fail(self):
        state = routing.LwdHashRoutingState(1, 2, 8)
        ids = torch.zeros(2, 1, 2, dtype=torch.int32)
        for offset, chunk in [(3, ids), (0, ids + 8), (0, ids[:, :, :1])]:
            with self.subTest(offset=offset, shape=chunk.shape), self.assertRaises(ValueError):
                state.add_chunk("p", 4, offset, chunk)
        state.add_chunk("p", 4, 2, ids)
        with self.assertRaisesRegex(ValueError, "Missing LWD"):
            state.build_batch(["p"], [2], [0], [4])


class TestMixedEmbeddingMasks(unittest.TestCase):
    def test_device_mask_uses_corrected_positions_and_preserves_prompt(self):
        cls = load_methods(RUNNER, {"_lwd_refresh_decode_embedding_mask", "_lwd_preprocess_prompt_embeds"})
        runner = cls()
        runner.input_batch = NS(
            req_ids=["decode", "prompt"],
            num_prompt_tokens_cpu_tensor=torch.tensor([4, 3]),
        )
        runner._lwd_prompt_lens_gpu = torch.zeros(2, dtype=torch.int32)
        runner.req_indices = NS(gpu=torch.tensor([0, 0, 1, 1, 1]))
        runner.positions = torch.tensor([4, 5, 0, 1, 2, 99])
        runner.is_token_ids = NS(gpu=torch.zeros(6, dtype=torch.bool))
        runner.input_ids = NS(gpu=torch.tensor([2, 3, 99999, -77, -1, 0]))
        runner.inputs_embeds = NS(gpu=torch.full((6, 2), 42.0))
        table = torch.arange(16, dtype=torch.float32).reshape(8, 2)
        runner.model = NS(embed_input_ids=lambda input_ids: table[input_ids])
        runner.uses_mrope = False
        runner.uses_xdrope_dim = 0
        runner._init_model_kwargs = dict
        output = NS(total_num_scheduled_tokens=5)
        runner._lwd_refresh_decode_embedding_mask(output)
        result = runner._lwd_preprocess_prompt_embeds(output, 6)
        self.assertEqual(runner.is_token_ids.gpu[:5].tolist(), [True, True, False, False, False])
        self.assertTrue(torch.equal(result[1][:2], table[[2, 3]]))
        self.assertTrue(torch.equal(result[1][2:5], torch.full((3, 2), 42.0)))
        self.assertEqual(result[2][-1], 0)

    def test_cpu_fallback_handles_optimistic_decode_positions(self):
        cls = load_methods(RUNNER, {"_lwd_rebuild_is_token_ids_mask"})
        runner = cls()
        runner.input_batch = NS(
            req_prompt_embeds={1: object()},
            num_computed_tokens_cpu=np.array([100, 0, 2]),
            num_prompt_tokens=np.array([4, 3, 4]),
        )
        runner.is_token_ids = NS(np=np.zeros(9, dtype=bool))
        runner._lwd_rebuild_is_token_ids_mask(np.array([2, 3, 3]))
        self.assertEqual(runner.is_token_ids.np[:8].tolist(), [True, True, False, False, False, False, False, True])


class TestMixedInjection(unittest.TestCase):
    def setUp(self):
        class Base:
            def _prepare_inputs(self, scheduler_output, counts):
                self.positions_seen_by_base = {
                    rid: state.mrope_positions.clone() for rid, state in self.requests.items()
                }
                # The base input preparation must not overwrite remote embeds
                # after injection, and must already see the current mrope chunk.
                self.inputs_embeds.gpu.fill_(-42)
                return "prepared"

        cls = load_methods(
            RUNNER,
            {
                "_lwd_inject_remote_embeds",
                "_lwd_harvest_up_batches",
                "_lwd_inject_mrope_positions",
                "_prepare_inputs",
                "_lwd_prepare_hash_batch",
            },
            base=Base,
            time=time,
            logger=NS(info=lambda *args: None),
            hash_layer_count=routing.hash_layer_count,
            hash_payload_numel=routing.hash_payload_numel,
            unpack_hash_payload=routing.unpack_hash_payload,
        )
        self.runner = cls()
        self.embeds = torch.arange(8).reshape(4, 2).to(torch.bfloat16)
        self.ids = torch.arange(8).reshape(4, 1, 2).int()
        payload = routing.pack_hash_payload(self.embeds, self.ids)
        self.meta = NS(req_ids=["b", "a"], token_ids=[[0, 0], [0, 0]], prompt_offsets=[0, 2], has_mrope=[False, False])
        self.result = NS(tensor=payload, aux_tensor=None)
        self.runner.worker = NS(
            rank=1,
            parallel_config=NS(lwd_config=NS(edge_npu_count=1)),
            _lwd_up_recv_futures={7: (NS(wait=lambda timeout: self.result), self.meta)},
        )
        self.runner.model_config = NS(
            get_hidden_size=lambda: 2,
            hf_config=NS(model_type="deepseek_v4", num_hash_layers=1, num_experts_per_tok=2),
        )
        self.runner.input_batch = NS(
            num_reqs=3,
            req_ids=["a", "decode", "b"],
            req_id_to_index={"a": 0, "decode": 1, "b": 2},
            num_prompt_tokens=[4, 4, 2],
            num_computed_tokens_cpu=[2, 9, 0],
            req_prompt_embeds={0: torch.zeros(4, 2), 2: torch.zeros(2, 2)},
            is_token_ids=np.ones((3, 16), dtype=bool),
        )
        self.runner.inputs_embeds = NS(gpu=torch.full((6, 2), -42.0))
        self.runner.lwd_hash_state = routing.LwdHashRoutingState(1, 2, 8)
        self.runner._lwd_hash_experts = torch.zeros(6, 1, 2, dtype=torch.int32)
        self.runner._lwd_hash_prompt_mask = torch.zeros(6, dtype=torch.bool)
        self.runner.requests = {
            "a": NS(mrope_positions=torch.zeros(3, 4, dtype=torch.int64)),
            "b": NS(mrope_positions=torch.zeros(3, 2, dtype=torch.int64)),
        }
        self.runner._lwd_enabled = lambda: True
        self.runner._lwd_rebuild_is_token_ids_mask = lambda counts: None
        self.runner._lwd_release_consumed_prompt_embeds = lambda: None

    def inject(self, tp_group=None):
        parallel = NS(get_tp_group=lambda: tp_group or NS(world_size=1))
        with (
            patch.dict(sys.modules, {"vllm.distributed.parallel_state": parallel}),
            patch.object(torch, "npu", NS(current_stream=lambda: None), create=True),
            patch.object(torch.Tensor, "record_stream", return_value=None),
        ):
            self.assertEqual(self.runner._prepare_inputs(NS(), [2, 2, 2]), "prepared")

    def test_transport_order_differs_from_runner_order(self):
        self.inject()
        actual = self.runner.inputs_embeds.gpu
        self.assertTrue(torch.equal(actual[:2], self.embeds[2:]))
        self.assertTrue(torch.equal(actual[2:4], torch.full((2, 2), -42.0)))
        self.assertTrue(torch.equal(actual[4:], self.embeds[:2]))
        batch, mask = self.runner.lwd_hash_state.build_batch(["a", "decode", "b"], [2, 2, 2], [2, 9, 0], [4, 4, 2])
        self.assertTrue(torch.equal(batch[:2], self.ids[2:]))
        self.assertTrue(torch.equal(batch[4:], self.ids[:2]))
        self.assertEqual(mask.tolist(), [True, True, False, False, True, True])

    def test_missing_offsets_fail(self):
        self.meta.prompt_offsets = []
        with self.assertRaisesRegex(ValueError, "one prompt offset"):
            self.inject()

    def test_wrong_absolute_offset_fails(self):
        self.meta.prompt_offsets = [1, 2]
        with self.assertRaisesRegex(ValueError, "chunk offset mismatch"):
            self.inject()

    def test_mm_aux_is_visible_before_base_preparation(self):
        self.runner.model_config.hf_config = NS(model_type="qwen2_5_vl")
        self.runner.lwd_hash_state = None
        self.result.tensor = self.embeds.flatten()
        self.result.aux_tensor = torch.tensor([[3, 4, 5], [6, 7, 8]])
        self.meta.has_mrope = [True, False]
        self.inject()
        self.assertTrue(
            torch.equal(
                self.runner.positions_seen_by_base["b"],
                self.result.aux_tensor.t(),
            )
        )
        self.assertFalse(self.runner.positions_seen_by_base["a"].any())
        self.assertEqual(self.runner.requests["b"].mrope_position_delta, 7)
        self.assertTrue(torch.equal(self.runner.inputs_embeds.gpu[:2], self.embeds[2:]))
        self.assertTrue(torch.equal(self.runner.inputs_embeds.gpu[4:], self.embeds[:2]))

    def test_chunk_rows_must_match_scheduled_window(self):
        self.meta.token_ids[0] = [0]
        with self.assertRaisesRegex(RuntimeError, "window mismatch"):
            self.inject()

    def test_hash_staging_runs_after_injection(self):
        self.inject()
        self.assertTrue(torch.equal(self.runner._lwd_hash_experts[:2], self.ids[2:]))
        self.assertTrue(torch.equal(self.runner._lwd_hash_experts[4:], self.ids[:2]))
        self.assertEqual(self.runner._lwd_hash_prompt_mask.tolist(), [True, True, False, False, True, True])

    def test_tp_receiver_allocates_complete_hash_payload(self):
        self.runner.worker.rank = 2
        self.runner.worker._lwd_up_recv_futures[7] = (None, self.meta)
        native_empty = torch.empty
        sizes = []

        def allocate(size, **kwargs):
            sizes.append(size)
            return native_empty(size, dtype=kwargs["dtype"])

        def broadcast(tensor, **kwargs):
            self.assertEqual(kwargs["src"], 1)
            tensor.copy_(self.result.tensor)
            return NS(wait=lambda: None)

        with (
            patch.object(torch, "empty", side_effect=allocate),
            patch("torch.distributed.broadcast", side_effect=broadcast),
        ):
            self.inject(NS(world_size=2, ranks=[1, 2], device_group=object()))
        self.assertEqual(sizes, [routing.hash_payload_numel(4, 2, 1, 2)])
        self.assertTrue(torch.equal(self.runner._lwd_hash_experts[:2], self.ids[2:]))

    def test_tp_receiver_preserves_separate_mm_aux_frame(self):
        self.runner.worker.rank = 2
        self.runner.worker._lwd_up_recv_futures[7] = (None, self.meta)
        self.runner.model_config.hf_config = NS(model_type="qwen2_5_vl")
        self.runner.lwd_hash_state = None
        self.meta.has_mrope = [False, True]
        aux = torch.tensor([[3, 4, 5], [6, 7, 8]])
        frames = iter([self.embeds.flatten(), aux.flatten()])
        events = []
        native_empty = torch.empty

        def allocate(size, **kwargs):
            return native_empty(size, dtype=kwargs["dtype"])

        def broadcast(tensor, **kwargs):
            frame = next(frames)
            self.assertEqual(tensor.dtype, frame.dtype)
            tensor.copy_(frame)
            events.append(("launch", tensor.numel()))
            return NS(wait=lambda: events.append(("wait", tensor.numel())))

        with (
            patch.object(torch, "empty", side_effect=allocate),
            patch("torch.distributed.broadcast", side_effect=broadcast),
        ):
            self.inject(NS(world_size=2, ranks=[1, 2], device_group=object()))
        self.assertEqual(events, [("launch", 8), ("wait", 8), ("launch", 6), ("wait", 6)])
        self.assertTrue(torch.equal(self.runner.positions_seen_by_base["a"][:, 2:], aux.t()))
        self.assertFalse(self.runner.positions_seen_by_base["b"].any())


class TestExternalEmbeddingSP(unittest.TestCase):
    def test_main_and_mtp_shard_only_embedding_rows(self):
        class Base:
            def _model_forward(self, *args, **kwargs):
                return args

            def _run_merged_draft(self, *args):
                return args

        context = NS(flash_comm_v1_enabled=True, pad_size=1)
        group = NS(world_size=2, rank_in_group=1)
        globals_ = dict(get_forward_context=lambda: context, get_tp_group=lambda: group)
        main = load_methods(RUNNER, {"_model_forward"}, Base, **globals_)()
        main.supports_mm_inputs = False
        main.lwd_hash_state = None
        mtp = load_methods("worker/lwd_cloud/lwd_mtp_proposer.py", {"_run_merged_draft"}, Base, **globals_)()
        mtp.is_multimodal_model = False
        embeds = torch.arange(6).reshape(3, 2).float()
        positions = torch.arange(3)
        main_result = main._model_forward(3, positions=positions, inputs_embeds=embeds)
        mtp_result = mtp._run_merged_draft(3, 1, None, positions, embeds, None, 3)
        expected = torch.tensor([[4.0, 5.0], [0.0, 0.0]])
        self.assertTrue(torch.equal(main_result[4], expected))
        self.assertTrue(torch.equal(mtp_result[4], expected))
        self.assertIs(main_result[2], positions)
        self.assertIs(mtp_result[3], positions)
        with self.assertRaisesRegex(ValueError, "full-token"):
            main._model_forward(3, inputs_embeds=embeds[:2])
        with self.assertRaisesRegex(ValueError, "full-token"):
            mtp._run_merged_draft(3, 1, None, positions, embeds[:2], None, 3)


if __name__ == "__main__":
    unittest.main()
