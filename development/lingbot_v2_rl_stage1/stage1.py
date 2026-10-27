"""Preparation for LingBot V2 RL. No policy-gradient training is implemented here."""
from __future__ import annotations

import argparse
import ast
import json
import math
import struct
from pathlib import Path


def checkpoint_inventory(directory):
    """Validate safetensors headers/index without allocating the model or reading tensor bodies.

    This detects structural errors, not corruption in tensor payloads. It is not SHA verification.
    """
    root = Path(directory)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    tensors = {}
    dtype_bytes = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2,
                   "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
    for filename in sorted(set(index["weight_map"].values())):
        if Path(filename).name != filename:
            raise ValueError("Shard names must be plain filenames")
        path = root / filename
        size = path.stat().st_size
        with path.open("rb") as f:
            prefix = f.read(8)
            if len(prefix) != 8:
                raise ValueError(f"Short header: {filename}")
            header_size, = struct.unpack("<Q", prefix)
            if header_size > min(size - 8, 100_000_000):
                raise ValueError(f"Invalid header size: {filename}")
            header = json.loads(f.read(header_size))
        ranges = []
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            if key in tensors or index["weight_map"].get(key) != filename:
                raise ValueError(f"Duplicate or misindexed tensor: {key}")
            shape, dtype = meta["shape"], meta["dtype"]
            if dtype not in dtype_bytes or any(type(n) is not int or n < 0 for n in shape):
                raise ValueError(f"Unsupported dtype or shape: {key}")
            start, end = meta["data_offsets"]
            if not (0 <= start <= end <= size - 8 - header_size):
                raise ValueError(f"Invalid data range: {key}")
            if end - start != math.prod(shape) * dtype_bytes[dtype]:
                raise ValueError(f"Shape/byte mismatch: {key}")
            ranges.append((start, end))
            tensors[key] = {"shape": shape, "dtype": dtype, "shard": filename}
        offset = 0
        for start, end in sorted(ranges):
            if start != offset:
                raise ValueError(f"Overlapping ranges or data hole: {filename}")
            offset = end
        if offset != size - 8 - header_size:
            raise ValueError(f"Trailing/missing bytes: {filename}")
    if set(tensors) != set(index["weight_map"]):
        raise ValueError("Index/header key mismatch")
    return tensors


def derive_indices(training, robot, category, max_dim):
    """Derive env-order indices from the same joint-order/slice rules as FeatureTransform.

    Supports the present RoboTwin absolute joint-position schema only; reject other schemas.
    This performs index mapping only, never normalization or conversion of relative actions.
    """
    if category not in ("states", "actions"):
        raise ValueError(category)
    offset, groups = 0, {}
    for entry in training["data"]["joints"]:
        entry = ast.literal_eval(entry) if isinstance(entry, str) else entry
        if len(entry) != 1:
            raise ValueError("Expected one feature per joint entry")
        name, width = next(iter(entry.items()))
        if type(width) is not int or width < 0 or name in groups:
            raise ValueError("Invalid or duplicate joint group")
        if width:
            groups[name] = (offset, width)
            offset += width
    if offset > max_dim:
        raise ValueError("Joint layout exceeds model dimension")
    prefix = "action." if category == "actions" else "observation.state."
    source = "action" if category == "actions" else "observation.state"
    mapping = {}
    for entry in robot[category]:
        target, spec = next(iter(entry.items()))
        if not target.startswith(prefix) or target[len(prefix):] not in groups:
            raise ValueError(f"Unknown target {target}")
        if spec.get("subtract_state", False) or spec.get("convert_from_state", False):
            raise ValueError("Relative/derived actions require upstream FeatureTransform")
        base, width = groups[target[len(prefix):]]
        used = 0
        for origin in spec["origin_keys"]:
            key, span = next(iter(origin.items()))
            if key != source or not (0 <= span["start"] < span["end"]):
                raise ValueError("Unsupported origin")
            for i in range(span["start"], span["end"]):
                if i in mapping or used >= width:
                    raise ValueError("Duplicate origin or group overflow")
                mapping[i] = base + used
                used += 1
    if sorted(mapping) != list(range(14)):
        raise ValueError("Expected complete RoboTwin 14-D mapping")
    return [mapping[i] for i in range(14)]


def select_valid_dims(tensor, indices):
    """Preserve leading axes; also usable on per-dimension logprobs later."""
    if len(set(indices)) != 14 or min(indices) < 0 or max(indices) >= tensor.shape[-1]:
        raise ValueError("Invalid indices")
    return tensor[..., indices]


def strict_load(policy, directory, *, allowed_extra_prefixes=()):
    """Validate ALL key/shape sets before streaming shards into an existing V2 policy.

    Exclusions default empty. Explicit teacher-only exclusions are a caller decision.
    Model construction, Qwen config merge and tokenizer remain the upstream loader's job.
    """
    import torch
    from safetensors.torch import load_file

    inv = checkpoint_inventory(directory)
    expected = policy.state_dict()
    if any(v.device.type == "meta" for v in expected.values()):
        raise ValueError("Materialize model before loading")
    missing = sorted(set(expected) - set(inv))
    extra = sorted(k for k in set(inv) - set(expected)
                   if not any(k.startswith(p) for p in allowed_extra_prefixes))
    mismatch = sorted(k for k in set(expected) & set(inv)
                      if list(expected[k].shape) != inv[k]["shape"])
    if missing or extra or mismatch:
        raise ValueError(json.dumps({"missing": missing, "extra": extra,
                                     "shape_mismatch": mismatch}))
    for name in sorted({v["shard"] for v in inv.values()}):
        values = load_file(str(Path(directory) / name), device="cpu")
        values = {k: v for k, v in values.items() if k in expected}
        for k, v in values.items():
            if v.is_floating_point() and not torch.isfinite(v).all():
                raise ValueError(f"Nonfinite weight: {k}")
        policy.load_state_dict(values, strict=False)  # Complete key/shape sets checked above.
    return {"loaded_keys": len(expected), "excluded_keys": len(set(inv) - set(expected)),
            "parameter_dtypes": sorted({str(v.dtype) for v in expected.values()})}


def raw_observations(env):
    """Translate RLinf B,H,W,C uint8 RGB observations to upstream raw keys.

    Calls no tokenizer, resizer or normalizer: upstream FeatureTransform must handle those.
    """
    import numpy as np

    def array(value):
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

    main, wrists, states = (array(env[k]) for k in ("main_images", "wrist_images", "states"))
    tasks = env["task_descriptions"]
    b = len(main)
    if main.ndim != 4 or main.shape[-1] != 3 or main.dtype != np.uint8:
        raise ValueError("main_images must be uint8 RGB [B,H,W,3]")
    if (wrists.ndim != 5 or wrists.shape[:2] != (b, 2) or wrists.shape[-1] != 3
            or wrists.dtype != np.uint8):
        raise ValueError("wrist_images must be uint8 RGB [B,2,H,W,3], left then right")
    if states.shape != (b, 14) or not np.isfinite(states).all():
        raise ValueError("states must be finite [B,14]")
    if isinstance(tasks, str) or len(tasks) != b or not all(isinstance(t, str) for t in tasks):
        raise ValueError("One task description per environment is required")
    return [{"observation.images.cam_high": main[i].copy(),
             "observation.images.cam_left_wrist": wrists[i, 0].copy(),
             "observation.images.cam_right_wrist": wrists[i, 1].copy(),
             "observation.state": states[i].copy(), "task": tasks[i]} for i in range(b)]


class V2ODEBridge:
    """Inference-only bridge over an already loaded original LingbotVLAv2Server.

    This is NOT a BasePolicy implementation, and exposes no invented RL logprobs.
    Original preprocessing, sampler and action unnormalization are used unchanged.
    """
    def __init__(self, server, execute_steps=25):
        if not hasattr(server, "vla") or server.vla.config.vlm_family != "qwen3_vl":
            raise ValueError("Expected original V2 server")
        if not 0 < execute_steps <= server.vla.config.n_action_steps:
            raise ValueError("Invalid execution horizon")
        self.server, self.execute_steps = server, execute_steps

    def predict_action_batch(self, env, *, noise):
        import torch

        server = self.server
        transformed = [server._prepare_model_input(x) for x in raw_observations(env)]
        batch = {k: server._pad_and_stack_tensors([x[k] for x in transformed])
                 for k in ("images", "img_masks", "lang_tokens", "lang_masks", "state", "image_grid_thw")}
        policy = server.vla
        device = next(policy.parameters()).device
        dtype = torch.bfloat16 if server.use_bf16 else torch.float32
        expected = (len(transformed), policy.config.n_action_steps, policy.config.max_action_dim)
        if tuple(noise.shape) != expected or not torch.isfinite(noise).all():
            raise ValueError(f"Expected finite noise {expected}")
        inputs = {k: v.to(device=device, dtype=(dtype if k in ("images", "state") else v.dtype))
                  for k, v in batch.items()}
        inputs["image_grid_thw"] = inputs["image_grid_thw"].long()
        policy.eval()
        with torch.no_grad():
            # The upstream Euler loop updates x_t in place; preserve caller noise for replay.
            normalized = policy.model.sample_actions(**inputs, noise=noise.to(device=device, dtype=dtype).clone())
        if tuple(normalized.shape) != expected or not torch.isfinite(normalized).all():
            raise ValueError("Invalid V2 sampler output")
        actions = server._unapply_batched_actions(transformed, normalized.detach().float().cpu())["action"]
        if actions.shape != (len(transformed), expected[1], 14):
            raise ValueError("Expected unnormalized [B,H,14]")
        return {"actions": actions[:, :self.execute_steps],
                "normalized_actions": normalized.detach(),
                "executed_horizon": self.execute_steps}


def main():
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--robot-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads((args.checkpoint / "config.json").read_text())
    training = yaml.safe_load((args.checkpoint / "lingbotvla_cli.yaml").read_text())
    robot = yaml.safe_load(args.robot_config.read_text())
    if config.get("architectures") != ["LingbotVlaV2Policy"] or config.get("vlm_family") != "qwen3_vl":
        raise ValueError("Not the expected V2 checkpoint")
    inventory = checkpoint_inventory(args.checkpoint)
    result = {"status": "static_checks_passed_runtime_unverified",
              "checkpoint": str(args.checkpoint.resolve()), "tensor_count": len(inventory),
              "numel": sum(math.prod(v["shape"]) for v in inventory.values()),
              "dtypes": sorted({v["dtype"] for v in inventory.values()}),
              "state_indices": derive_indices(training, robot, "states", config["max_state_dim"]),
              "action_indices": derive_indices(training, robot, "actions", config["max_action_dim"]),
              "predicted_horizon": config["n_action_steps"],
              "model_action_dim": config["max_action_dim"],
              "payload_hash_verified_this_run": False,
              "model_load_verified": False, "ode_equivalence_verified": False,
              "rl_update_implemented": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
