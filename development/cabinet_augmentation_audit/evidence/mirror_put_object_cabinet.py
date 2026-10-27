#!/usr/bin/env python3
"""Generate simulator-validated left/right symmetric put_object_cabinet demos.

The source episodes are the official demo_clean seeds.  For every seed, the
task scene is reconstructed, the object and cabinet root poses are reflected
across the world x=0 plane, and RoboTwin's existing expert replans the task.
Only episodes passing both planning and task success checks are exported.
"""

import argparse
import json
import os
import re
import sys
import traceback
import types
from copy import deepcopy

import numpy as np
import sapien.core as sapien
import transforms3d as t3d
import yaml

ROB0TWIN_ROOT = "/RoboTwin"
sys.path.insert(0, ROB0TWIN_ROOT)

from description.utils.generate_episode_instructions import generate_episode_descriptions
from envs._GLOBAL_CONFIGS import CONFIGS_PATH
from envs.put_object_cabinet import put_object_cabinet
from envs.utils import ArmTag

# Measured from Aloha-AgileX neutral TCPs in the clean environment:
# left_x + right_x. The physical symmetry plane is x = value / 2, not x = 0.
LEFT_NEUTRAL_TCP = np.array([
    -0.2979237139, -0.3138038814, 0.9419986010,
    0.7000007076, -0.0000070933, -0.0000037938, 0.7141421493,
])
RIGHT_NEUTRAL_TCP = np.array([
    0.3064331412, -0.3127997518, 0.9409983754,
    0.7035625829, -0.0000070778, -0.0000037937, 0.7106333034,
])
REFLECT_X = np.diag([-1.0, 1.0, 1.0])


def map_pose_between_arms(pose: sapien.Pose, source_home7, target_home7) -> sapien.Pose:
    """Move an actor to the opposite side without reflecting its mesh frame.

    RoboTwin's evaluation sampler changes the object's x side but retains the
    ordinary object orientation distribution. Reflecting the orientation frame
    would not reflect the actual mesh and makes contact-point grasps invalid.
    """
    source_p = np.asarray(source_home7[:3], dtype=np.float64)
    target_p = np.asarray(target_home7[:3], dtype=np.float64)
    p = target_p + REFLECT_X @ (np.asarray(pose.p, dtype=np.float64) - source_p)

    # Asset meshes are not reflected geometrically. Preserve their sampled
    # orientation so support faces and the cabinet front remain physically
    # valid; only map the task-side position.
    return sapien.Pose(p=p, q=np.asarray(pose.q, dtype=np.float64))


class symmetric_put_object_cabinet(put_object_cabinet):
    """Original task with its task actors mirrored after deterministic loading."""

    def setup_demo(self, **kwargs):
        super().setup_demo(**kwargs)
        if kwargs.get("need_plan", True):
            self._install_clik_fallback()

    def _install_clik_fallback(self):
        """Fallback for contact targets rejected by collision-aware MPLib IK.

        The normal planner remains first choice. On failure, solve the same
        target with continuous kinematic IK and create a dense joint path; the
        resulting motion is accepted only if the simulator task succeeds.
        """
        robot = self.robot
        robot.left_mplib_planner.plan_step_lim = 5000
        robot.right_mplib_planner.plan_step_lim = 5000

        def install(arm):
            original = robot.left_plan_path if arm == "left" else robot.right_plan_path
            wrapper = robot.left_mplib_planner if arm == "left" else robot.right_mplib_planner
            entity = robot.left_entity if arm == "left" else robot.right_entity

            def plan_with_fallback(this, target_pose, constraint_pose=None,
                                   use_point_cloud=False, use_attach=False,
                                   last_qpos=None):
                result = original(
                    target_pose, constraint_pose=constraint_pose,
                    use_point_cloud=use_point_cloud, use_attach=use_attach,
                    last_qpos=last_qpos,
                )
                setattr(this, f"_debug_last_{arm}_target", np.asarray(target_pose, dtype=float).tolist())
                if result.get("status") == "Success":
                    return result
                planner = wrapper.planner
                start = np.asarray(entity.get_qpos() if last_qpos is None else last_qpos, dtype=np.float64)
                link_pose = this._trans_from_gripper_to_endlink(target_pose, arm_tag=arm)
                goal = planner._transform_goal_to_wrt_base(link_pose)
                link_idx = planner.link_name_2_idx[planner.move_group]
                q_goal, converged, _ = planner.pinocchio_model.compute_IK_CLIK(
                    link_idx, goal, start, []
                )
                if not converged or not planner.wrap_joint_limit(q_goal):
                    ik_status, sampled_q = planner.IK(
                        goal, start, n_init_qpos=100, threshold=0.003,
                        return_closest=True, verbose=False,
                    )
                    if ik_status == "Success" and sampled_q is not None:
                        q_goal = sampled_q
                        converged = True
                        print(f"SAMPLED_IK_FALLBACK_SUCCESS arm={arm}")
                    else:
                        print(
                            f"CLIK_FALLBACK_FAIL arm={arm} converged={converged} "
                            f"sampled_status={ik_status}"
                        )
                        return result
                planner.pinocchio_model.compute_forward_kinematics(q_goal)
                reached = planner.pinocchio_model.get_link_pose(link_idx)
                error = float(goal.distance(reached))
                if error > 0.003:
                    print(f"CLIK_FALLBACK_FAIL arm={arm} error={error:.6f}")
                    return result
                idx = np.asarray(planner.move_group_joint_indices, dtype=int)
                active = list(entity.get_active_joints())
                arm_joints = this.left_arm_joints if arm == "left" else this.right_arm_joints
                arm_idx = np.asarray([active.index(j) for j in arm_joints], dtype=int)
                q_goal = np.asarray(q_goal, dtype=np.float64).copy()
                # CLIK may return a periodic FK-equivalent solution several
                # revolutions away (e.g. -6 to -10 rad). Sapien joints are
                # limited and cannot track such drive targets. Select the
                # 2*pi-equivalent value nearest the current configuration.
                q_goal[arm_idx] = start[arm_idx] + (
                    (q_goal[arm_idx] - start[arm_idx] + np.pi) % (2 * np.pi) - np.pi
                )
                planner.pinocchio_model.compute_forward_kinematics(q_goal)
                reached_wrapped = planner.pinocchio_model.get_link_pose(link_idx)
                wrapped_error = float(goal.distance(reached_wrapped))
                if wrapped_error > 0.003:
                    print(f"CLIK_WRAP_FAIL arm={arm} error={wrapped_error:.6f}")
                    return result
                if not getattr(this, f"_debug_printed_{arm}_joint_map", False):
                    print(
                        f"JOINT_MAP arm={arm} move_group_idx={idx.tolist()} "
                        f"arm_idx={arm_idx.tolist()} qdim={len(q_goal)} "
                        f"active_names={[j.get_name() for j in active]} "
                        f"arm_names={[j.get_name() for j in arm_joints]}"
                    )
                    setattr(this, f"_debug_printed_{arm}_joint_map", True)
                # set_arm_joints consumes values in self.{left,right}_arm_joints
                # order.  MPLib's move-group indices need not share that order.
                max_delta = float(np.max(np.abs(q_goal[arm_idx] - start[arm_idx])))
                n_steps = max(200, int(np.ceil(max_delta / 0.0025)))
                positions = np.linspace(start[arm_idx], q_goal[arm_idx], n_steps)
                velocities = np.zeros_like(positions)
                setattr(this, f"_debug_last_{arm}_goal_arm_q", positions[-1].tolist())
                print(f"CLIK_FALLBACK_SUCCESS arm={arm} error={error:.6f}")
                return {"status": "Success", "position": positions, "velocity": velocities}

            method = types.MethodType(plan_with_fallback, robot)
            def multi_with_fallback(this, target_lst, constraint_pose=None,
                                    use_point_cloud=False, use_attach=False,
                                    last_qpos=None):
                statuses, positions, velocities = [], [], []
                for target in target_lst:
                    res = plan_with_fallback(
                        this, target, constraint_pose=constraint_pose,
                        use_point_cloud=use_point_cloud, use_attach=use_attach,
                        last_qpos=last_qpos,
                    )
                    statuses.append(res.get("status", "Fail"))
                    positions.append(res.get("position", []))
                    velocities.append(res.get("velocity", []))
                return {
                    "status": np.asarray(statuses, dtype=object),
                    "position": positions,
                    "velocity": velocities,
                }

            multi_method = types.MethodType(multi_with_fallback, robot)
            if arm == "left":
                robot.left_plan_path = method
                robot.left_plan_multi_path = multi_method
            else:
                robot.right_plan_path = method
                robot.right_plan_multi_path = multi_method

        install("left")
        install("right")

    def play_once(self):
        arm_tag = ArmTag("right" if self.object.get_pose().p[0] > 0 else "left")
        self.arm_tag = arm_tag
        self.origin_z = self.object.get_pose().p[2]

        def run_stage(name, action):
            self.last_stage = name
            result = self.move(action)
            print(
                f"STAGE name={name} plan_success={self.plan_success} "
                f"task_success={self.check_success()} object_pose={self.object.get_pose().p.tolist()} "
                f"cabinet_qpos={self.cabinet.get_qpos().tolist()} "
                f"left_open={self.robot.is_left_gripper_open()} "
                f"right_open={self.robot.is_right_gripper_open()}"
            )
            if name in ("grasp_object", "grasp_cabinet"):
                def joint_debug(arm):
                    entity = self.robot.left_entity if arm == "left" else self.robot.right_entity
                    joints = self.robot.left_arm_joints if arm == "left" else self.robot.right_arm_joints
                    active = list(entity.get_active_joints())
                    idx = [active.index(j) for j in joints]
                    return {
                        "goal": getattr(self.robot, f"_debug_last_{arm}_goal_arm_q", None),
                        "actual": np.asarray(entity.get_qpos())[idx].tolist(),
                        "drive": [float(j.get_drive_target()[0]) for j in joints],
                    }
                print(
                    f"CONTACT_DEBUG stage={name} left_tcp={self.get_arm_pose('left')} "
                    f"right_tcp={self.get_arm_pose('right')} "
                    f"left_target={getattr(self.robot, '_debug_last_left_target', None)} "
                    f"right_target={getattr(self.robot, '_debug_last_right_target', None)}"
                )
                print(f"JOINT_DEBUG stage={name} left={joint_debug('left')} right={joint_debug('right')}")
            return result

        run_stage("grasp_object", self.grasp_actor(self.object, arm_tag=arm_tag, pre_grasp_dis=0.1))
        run_stage("grasp_cabinet", self.grasp_actor(self.cabinet, arm_tag=arm_tag.opposite, pre_grasp_dis=0.05))
        for i in range(4):
            run_stage(f"pull_drawer_{i}", self.move_by_displacement(arm_tag=arm_tag.opposite, y=-0.04))
        run_stage("lift_object", self.move_by_displacement(arm_tag=arm_tag, z=0.15))
        target_pose = self.cabinet.get_functional_point(0)
        # Preserve the demonstrated grasp orientation. The generic
        # place_actor may impose an additional object rotation which is not
        # required by this task and can be unreachable after side swapping.
        # Move above the drawer, descend, then release; all translations are
        # derived from the cabinet functional point and current object pose.
        obj_p = np.asarray(self.object.get_pose().p, dtype=np.float64)
        fp_p = np.asarray(target_pose[:3], dtype=np.float64)
        xy_delta = fp_p[:2] - obj_p[:2]
        xy_distance = float(np.linalg.norm(xy_delta))
        if xy_distance > 0.035:
            xy_delta *= (xy_distance - 0.035) / xy_distance
        xy_steps = max(1, int(np.ceil(np.linalg.norm(xy_delta) / 0.04)))
        for i in range(xy_steps):
            run_stage(
                f"place_object_xy_{i}",
                self.move_by_displacement(
                    arm_tag=arm_tag,
                    x=float(xy_delta[0] / xy_steps),
                    y=float(xy_delta[1] / xy_steps),
                ),
            )
        obj_p = np.asarray(self.object.get_pose().p, dtype=np.float64)
        release_z = float(fp_p[2] + 0.06)
        run_stage(
            "place_object_z",
            self.move_by_displacement(arm_tag=arm_tag, z=release_z - float(obj_p[2])),
        )
        run_stage("release_object", self.open_gripper(arm_tag=arm_tag))
        self.info["info"] = {
            "{A}": f"{self.selected_modelname}/base{self.selected_model_id}",
            "{B}": "036_cabinet/base0",
            "{a}": str(arm_tag),
            "{b}": str(arm_tag.opposite),
        }
        return self.info

    def load_actors(self):
        super().load_actors()
        self.original_object_pose = deepcopy(self.object.get_pose())
        self.original_cabinet_pose = deepcopy(self.cabinet.get_pose())
        self.object.actor.set_pose(
            map_pose_between_arms(self.original_object_pose, LEFT_NEUTRAL_TCP, RIGHT_NEUTRAL_TCP)
        )
        self.cabinet.actor.set_root_pose(
            map_pose_between_arms(self.original_cabinet_pose, RIGHT_NEUTRAL_TCP, LEFT_NEUTRAL_TCP)
        )

        # Base task uses these areas for collision-aware planning. Recompute them
        # after moving the actors so they describe the mirrored scene.
        self.prohibited_area = []
        self.add_prohibit_area(self.object, padding=0.01)
        self.add_prohibit_area(self.cabinet, padding=0.01)
        self.prohibited_area.append([-0.15, -0.3, 0.15, 0.3])


def get_embodiment_config(robot_file):
    with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def load_args(output_dir):
    with open(os.path.join(CONFIGS_PATH, "demo_clean.yml"), "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    embodiment = args["embodiment"]
    if len(embodiment) != 1:
        raise ValueError("This generator expects the dual-arm aloha-agilex embodiment")
    robot_file = embodiment_types[embodiment[0]]["file_path"]
    args["left_robot_file"] = robot_file
    args["right_robot_file"] = robot_file
    args["left_embodiment_config"] = get_embodiment_config(robot_file)
    args["right_embodiment_config"] = get_embodiment_config(robot_file)
    args["dual_arm_embodied"] = True
    args["embodiment_name"] = str(embodiment[0])
    args["task_name"] = "put_object_cabinet"
    args["task_config"] = "demo_clean_sym_aug"
    args["save_path"] = output_dir
    args["render_freq"] = 0
    args["save_data"] = False
    return args


def build_instructions(info, episode_idx):
    generated = generate_episode_descriptions(
        "put_object_cabinet", [info.get("info", {})], 100
    )
    result = generated[0] if generated else {"seen": [], "unseen": []}
    for split in ("seen", "unseen"):
        if not result.get(split):
            result[split] = ["put object cabinet"]
    result["episode_index"] = episode_idx
    return result


def save_instruction(save_path, episode_idx, instructions):
    directory = os.path.join(save_path, "instruction")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"episode_{episode_idx:07d}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"seen": instructions["seen"], "unseen": instructions["unseen"]},
            f,
            ensure_ascii=False,
            indent=2,
        )


def close_safely(env, clear_cache=False):
    if env is None:
        return
    try:
        env.close_env(clear_cache=clear_cache)
    except Exception:
        traceback.print_exc()


def plan_episode(base_args, seed, output_index):
    env = symmetric_put_object_cabinet()
    args = deepcopy(base_args)
    args.update(now_ep_num=output_index, seed=seed, need_plan=True, save_data=False)
    try:
        env.setup_demo(**args)
        before = env.original_object_pose
        after = env.object.get_pose()
        env.play_once()
        success = bool(env.plan_success and env.check_success())
        record = {
            "source_seed": int(seed),
            "source_object_x": float(before.p[0]),
            "mirrored_object_x": float(after.p[0]),
            "source_cabinet_x": float(env.original_cabinet_pose.p[0]),
            "mirrored_cabinet_x": float(env.cabinet.get_pose().p[0]),
            "model": f"{env.selected_modelname}/base{env.selected_model_id}",
            "plan_success": bool(env.plan_success),
            "task_success": bool(env.check_success()),
            "last_stage": getattr(env, "last_stage", None),
            "object_final_pose": env.object.get_pose().p.tolist(),
            "cabinet_qpos": env.cabinet.get_qpos().tolist(),
            "left_gripper_open": bool(env.robot.is_left_gripper_open()),
            "right_gripper_open": bool(env.robot.is_right_gripper_open()),
        }
        if success:
            env.save_traj_data(output_index)
        return success, record
    finally:
        close_safely(env)


def collect_episode(base_args, seed, output_index):
    env = symmetric_put_object_cabinet()
    args = deepcopy(base_args)
    args.update(now_ep_num=output_index, seed=seed, need_plan=False, save_data=True)
    traj_path = os.path.join(args["save_path"], "_traj_data", f"episode{output_index}.pkl")
    import pickle
    with open(traj_path, "rb") as f:
        traj = pickle.load(f)
    args["left_joint_path"] = traj["left_joint_path"]
    args["right_joint_path"] = traj["right_joint_path"]
    try:
        env.setup_demo(**args)
        info = env.play_once()
        success = bool(env.plan_success and env.check_success())
        if not success:
            return False, info
        instructions = build_instructions(info, output_index)
        save_instruction(args["save_path"], output_index, instructions)
        # Match the official collector: close simulation before encoding cached frames.
        close_safely(env)
        env = None
        # The official function writes both HDF5 and MP4.
        collector = symmetric_put_object_cabinet()
        collector.save_data = True
        collector.folder_path = {"cache": os.path.join(args["save_path"], ".cache", f"episode{output_index}") + "/"}
        collector.ep_num = output_index
        collector.save_dir = args["save_path"]
        collector.save_freq = args["save_freq"]
        collector.merge_pkl_to_hdf5_video(instructions=instructions["seen"])
        collector.remove_data_cache()
        return True, info
    finally:
        close_safely(env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-seeds", required=True)
    parser.add_argument("--limit", type=int, default=3)
    args_ns = parser.parse_args()

    os.chdir(ROB0TWIN_ROOT)
    os.makedirs(args_ns.output, exist_ok=True)
    os.makedirs(os.path.join(args_ns.output, "_traj_data"), exist_ok=True)
    with open(args_ns.source_seeds, "r", encoding="utf-8") as f:
        seeds = [int(x) for x in f.read().split()]
    if args_ns.limit > 0:
        seeds = seeds[: args_ns.limit]

    base_args = load_args(args_ns.output)
    scene_info = {}
    records = []
    output_index = 0
    for source_index, seed in enumerate(seeds):
        print(f"[PLAN] source_episode={source_index} seed={seed}", flush=True)
        try:
            planned, record = plan_episode(base_args, seed, output_index)
            record["source_episode"] = source_index
            record["output_episode"] = output_index if planned else None
            if not planned:
                record["status"] = "planning_or_task_failure"
                records.append(record)
                with open(os.path.join(args_ns.output, "generation_report.json"), "w", encoding="utf-8") as f:
                    json.dump(records, f, ensure_ascii=False, indent=2)
                continue
            print(f"[COLLECT] output_episode={output_index} seed={seed}", flush=True)
            collected, info = collect_episode(base_args, seed, output_index)
            if not collected:
                record["status"] = "collection_failure"
                records.append(record)
                continue
            record["status"] = "success"
            records.append(record)
            scene_info[f"episode_{output_index}"] = info
            output_index += 1
        except Exception as exc:
            records.append({
                "source_episode": source_index,
                "source_seed": seed,
                "output_episode": None,
                "status": "exception",
                "error": repr(exc),
            })
            traceback.print_exc()

        with open(os.path.join(args_ns.output, "generation_report.json"), "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        with open(os.path.join(args_ns.output, "scene_info.json"), "w", encoding="utf-8") as f:
            json.dump(scene_info, f, ensure_ascii=False, indent=2)

    print(f"DONE successful={output_index} attempted={len(seeds)} output={args_ns.output}")


if __name__ == "__main__":
    main()
