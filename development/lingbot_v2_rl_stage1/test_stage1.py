import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import save_file

from stage1 import V2ODEBridge, checkpoint_inventory, derive_indices, raw_observations, select_valid_dims, strict_load


class Stage1Tests(unittest.TestCase):
    def setUp(self):
        self.training = {"data": {"joints": ["{'arm.position': 14}",
                            "{'end.position': 14}", "{'effector.position': 2}"]}}
        self.robot = {}
        for category, prefix, key in [("actions", "action.", "action"),
                                      ("states", "observation.state.", "observation.state")]:
            self.robot[category] = [
                {prefix + "arm.position": {"origin_keys": [
                    {key: {"start": 0, "end": 6}}, {key: {"start": 7, "end": 13}}]}},
                {prefix + "effector.position": {"origin_keys": [
                    {key: {"start": 6, "end": 7}}, {key: {"start": 13, "end": 14}}]}}]

    def test_joint_and_gripper_sentinels_roundtrip(self):
        idx = derive_indices(self.training, self.robot, "actions", 55)
        # Each env axis has a distinct sentinel; padding must never become an actuator.
        env = torch.arange(14.0)
        canonical = torch.full((2, 50, 55), -999.0)
        canonical[..., idx] = env
        torch.testing.assert_close(select_valid_dims(canonical, idx), env.expand(2, 50, 14))
        self.assertEqual(idx[6], 28)
        self.assertEqual(idx[13], 29)
        self.assertEqual(idx, derive_indices(self.training, self.robot, "states", 55))

    def test_group_order_is_derived(self):
        self.training["data"]["joints"].reverse()
        idx = derive_indices(self.training, self.robot, "actions", 55)
        self.assertEqual((idx[6], idx[13]), (0, 1))
        self.assertEqual(idx[:6], list(range(16, 22)))

    def test_relative_actions_rejected(self):
        self.robot["actions"][0]["action.arm.position"]["subtract_state"] = True
        with self.assertRaises(ValueError):
            derive_indices(self.training, self.robot, "actions", 55)

    def test_missing_actuator_rejected(self):
        self.robot["actions"].pop()
        with self.assertRaises(ValueError):
            derive_indices(self.training, self.robot, "actions", 55)

    def test_dimension_overflow_rejected(self):
        with self.assertRaises(ValueError):
            derive_indices(self.training, self.robot, "actions", 20)

    @staticmethod
    def env():
        return {"main_images": np.full((2, 4, 4, 3), 10, np.uint8),
                "wrist_images": np.stack([np.full((2, 4, 4, 3), 20, np.uint8),
                                           np.full((2, 4, 4, 3), 30, np.uint8)], axis=1),
                "states": np.arange(28, dtype=np.float32).reshape(2, 14),
                "task_descriptions": ["task A", "task B"]}

    def test_views_and_instructions_preserved_without_aliases(self):
        env = self.env()
        result = raw_observations(env)
        self.assertEqual(result[1]["task"], "task B")
        self.assertEqual(result[0]["observation.images.cam_left_wrist"][0, 0, 0], 20)
        self.assertEqual(result[0]["observation.images.cam_right_wrist"][0, 0, 0], 30)
        result[0]["observation.state"][0] = -1
        self.assertEqual(env["states"][0, 0], 0)

    def test_bad_observations_rejected(self):
        for key, value in [("main_images", self.env()["main_images"].astype(float)),
                           ("states", np.full((2, 14), np.nan)),
                           ("wrist_images", np.zeros((2, 1, 4, 4, 3), np.uint8)),
                           ("task_descriptions", "not a batch")]:
            env = self.env()
            env[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                raw_observations(env)

    def write_checkpoint(self, directory, tensors):
        root = Path(directory)
        save_file(tensors, root / "model.safetensors")
        (root / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {k: "model.safetensors" for k in tensors}}))

    def test_valid_weights_load_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected = {"weight": torch.arange(6.0).reshape(2, 3), "bias": torch.tensor([7., 8.])}
            self.write_checkpoint(tmp, expected)
            policy = torch.nn.Linear(3, 2)
            result = strict_load(policy, tmp)
            self.assertEqual(result["loaded_keys"], 2)
            for k, v in expected.items():
                torch.testing.assert_close(policy.state_dict()[k], v)

    def test_invalid_weights_fail_before_key_shape_mutation(self):
        for tensors in [{"weight": torch.zeros(2, 3)},
                        {"weight": torch.zeros(3, 2), "bias": torch.zeros(2)},
                        {"weight": torch.zeros(2, 3), "bias": torch.zeros(2), "unknown": torch.zeros(1)}]:
            with tempfile.TemporaryDirectory() as tmp:
                self.write_checkpoint(tmp, tensors)
                policy = torch.nn.Linear(3, 2)
                before = copy.deepcopy(policy.state_dict())
                with self.assertRaises(ValueError):
                    strict_load(policy, tmp)
                for k, v in before.items():
                    torch.testing.assert_close(policy.state_dict()[k], v)

    def test_nonfinite_weight_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_checkpoint(tmp, {"weight": torch.full((2, 3), float("nan")), "bias": torch.zeros(2)})
            with self.assertRaises(ValueError):
                strict_load(torch.nn.Linear(3, 2), tmp)

    def test_truncated_payload_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_checkpoint(tmp, {"weight": torch.zeros(2, 3)})
            p = Path(tmp) / "model.safetensors"
            p.write_bytes(p.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                checkpoint_inventory(tmp)

    def test_exclusions_must_be_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_checkpoint(tmp, {"weight": torch.zeros(2, 3), "bias": torch.zeros(2),
                                        "teacher.weight": torch.zeros(3)})
            with self.assertRaises(ValueError):
                strict_load(torch.nn.Linear(3, 2), tmp)
            result = strict_load(torch.nn.Linear(3, 2), tmp, allowed_extra_prefixes=("teacher.",))
            self.assertEqual(result["excluded_keys"], 1)

    def test_bridge_preserves_noise_replay_and_execution_horizon(self):
        class InPlaceSampler(torch.nn.Module):
            def sample_actions(self, *, noise, **inputs):
                return noise.add_(2)  # Original V2 Euler sampler also mutates its input noise.

        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.model = InPlaceSampler()
                self.config = SimpleNamespace(vlm_family="qwen3_vl", n_action_steps=50, max_action_dim=55)

        class Server:
            use_bf16 = False
            vla = Policy()

            @staticmethod
            def _prepare_model_input(raw):
                return {"images": torch.zeros(3, 3, 4, 4), "img_masks": torch.ones(3, dtype=torch.bool),
                        "state": torch.zeros(55), "lang_tokens": torch.ones(3, dtype=torch.long),
                        "lang_masks": torch.ones(3, dtype=torch.bool),
                        "image_grid_thw": torch.ones(3, 3, dtype=torch.long)}

            _pad_and_stack_tensors = staticmethod(torch.stack)

            @staticmethod
            def _unapply_batched_actions(transformed, normalized):
                return {"action": normalized[..., :14].numpy()}

        bridge = V2ODEBridge(Server(), execute_steps=25)
        noise = torch.randn(2, 50, 55)
        original = noise.clone()
        one = bridge.predict_action_batch(self.env(), noise=noise)
        two = bridge.predict_action_batch(self.env(), noise=noise)
        torch.testing.assert_close(noise, original)
        torch.testing.assert_close(one["normalized_actions"], original + 2)
        np.testing.assert_array_equal(one["actions"], two["actions"])
        self.assertEqual(one["actions"].shape, (2, 25, 14))


if __name__ == "__main__":
    unittest.main()
