"""GPU validation gate; needs existing LingBot dependencies and real raw observation NPZ.

NPZ keys: main_images (uint8 B,H,W,3), wrist_images (uint8 B,2,H,W,3),
states (B,14), task_descriptions (unicode B). No pickle is loaded.
"""
import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    for key in ("source", "checkpoint", "qwen_config", "norm_stats", "observations", "output"):
        parser.add_argument("--" + key.replace("_", "-"), required=True, type=Path)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--execute-steps", type=int, default=25)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--disable-fast-moe", action="store_true",
                        help="Use existing fused-layout fallback instead of inference-only robby_moe kernel")
    args = parser.parse_args()
    for key in ("source", "checkpoint", "qwen_config", "norm_stats", "observations", "output"):
        setattr(args, key, getattr(args, key).resolve())
    import numpy as np
    import torch
    from stage1 import V2ODEBridge, raw_observations, strict_load

    if not torch.cuda.is_available():
        raise RuntimeError("Linux/CUDA runtime required; no CPU equivalence claim")
    os.environ["QWEN3VL_PATH"] = str(args.qwen_config)
    os.environ["LINGBOTVLA_TRAINING_CONFIG"] = str(args.checkpoint / "lingbotvla_cli.yaml")
    sys.path.insert(0, str(args.source))
    os.chdir(args.source)
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
    if args.disable_fast_moe:
        import lingbotvla.models.vla.lingbot_vla.qwen2_action_expert as moe_module
        moe_module.robby_moe_forward = None

    class CheckedServer(LingbotVLAv2Server):
        def load_model_weights(self, directory, strict=True):
            self.weight_report = strict_load(self.vla, directory, allowed_extra_prefixes=(
                "model.current_video_align_", "model.depth_align_", "model.future_depth_align_",
                "model.future_video_align_", "model.current_shared_task_proj", "model.future_shared_task_proj"))

    server = CheckedServer(path_to_pi_model=str(args.checkpoint), robot_norm_path=str(args.norm_stats),
                           use_bf16=args.precision == "bf16", use_fp32=args.precision == "fp32",
                           use_compile=False, use_length=args.execute_steps, chunk_ret=True)
    server.reset("robotwin")
    with np.load(args.observations, allow_pickle=False) as data:
        env = {key: data[key] for key in ("main_images", "wrist_images", "states", "task_descriptions")}
    raw = raw_observations(env)
    shape = (len(raw), server.vla.config.n_action_steps, server.vla.config.max_action_dim)
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    torch.manual_seed(args.seed)
    noise = torch.randn(shape, device="cuda", dtype=dtype)
    # Inject identical noise directly; seed reset alone also depends on preprocessing RNG use.
    original_sampler = server.sample_actions_fn
    def fixed_noise_sampler(*positional, **kwargs):
        return original_sampler(*positional, **kwargs, noise=noise.clone())
    server.sample_actions_fn = fixed_noise_sampler
    reference, reference_normalized = server._infer_batch(raw, return_normalized=True)
    repeat_reference, repeat_normalized = server._infer_batch(raw, return_normalized=True)
    server.sample_actions_fn = original_sampler
    result = V2ODEBridge(server, args.execute_steps).predict_action_batch(env, noise=noise)
    diagnostic = {"baseline_repeat_max_abs_error": float((reference_normalized.float() - repeat_normalized.float()).abs().max()),
                  "bridge_max_abs_error_per_sample": (result["normalized_actions"].cpu().float() - reference_normalized.float()).abs().flatten(1).amax(1).tolist(),
                  "explicit_identical_noise": True, "precision": args.precision,
                  "inference_fast_moe_disabled": args.disable_fast_moe}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".diagnostic.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
    print(json.dumps(diagnostic, indent=2), flush=True)
    torch.testing.assert_close(reference_normalized, repeat_normalized, atol=args.atol, rtol=args.rtol)
    torch.testing.assert_close(result["normalized_actions"].cpu(), reference_normalized,
                               atol=args.atol, rtol=args.rtol, check_dtype=False)
    np.testing.assert_allclose(result["actions"], reference["action"][:, :args.execute_steps],
                               atol=args.atol, rtol=args.rtol)
    if not np.isfinite(result["actions"]).all():
        raise ValueError("Nonfinite environment action")
    report = {"status": "passed", "scope": "strict_load_and_original_ode_equivalence_only",
              "precision": args.precision, "atol": args.atol, "rtol": args.rtol,
              "seed": args.seed, "device": torch.cuda.get_device_name(),
              "inference_fast_moe_disabled": args.disable_fast_moe,
              "weight_report": server.weight_report,
              "normalized_max_abs_error": float((result["normalized_actions"].cpu().float() - reference_normalized.float()).abs().max()),
              "rl_update_verified": False, "environment_rollout_verified": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
