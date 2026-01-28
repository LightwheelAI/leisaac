import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import os
import time
import torch
from isaaclab.app import AppLauncher
import gymnasium as gym
import omni.kit.app

# ---- CLI ----
parser = argparse.ArgumentParser(description="leisaac data generation - Lula minimal")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--record", action="store_true")
parser.add_argument("--step_hz", type=int, default=60)
parser.add_argument("--dataset_file", type=str, default="./datasets/dataset.hdf5")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--num_demos", type=int, default=1)
parser.add_argument("--quality", action="store_true")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher_args = vars(args_cli)

app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

def load_extensions():
    app = omni.kit.app.get_app()
    ext_mgr = app.get_extension_manager()
    ext_mgr.set_extension_enabled_immediate("omni.isaac.motion_generation", True)
    ext_mgr.set_extension_enabled_immediate("omni.isaac.core", True)

load_extensions()

from leisaac.enhance.managers import StreamingRecorderManager, EnhanceDatasetExportMode
from leisaac.tasks.pick_orange.mdp import task_done
from isaaclab.managers import DatasetExportMode, TerminationTermCfg
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics

def manual_terminate(env: ManagerBasedRLEnv | DirectRLEnv, success: bool):
    """
    Programmatically mark the current episode as success or failure.

    This is the SAME implementation used in the teleoperation script.
    It does NOT require any human input.
    """
    if hasattr(env, "termination_manager"):
        if success:
            env.termination_manager.set_term_cfg(
                "success",
                TerminationTermCfg(
                    func=lambda env: torch.ones(
                        env.num_envs, dtype=torch.bool, device=env.device
                    )
                ),
            )
        else:
            env.termination_manager.set_term_cfg(
                "success",
                TerminationTermCfg(
                    func=lambda env: torch.zeros(
                        env.num_envs, dtype=torch.bool, device=env.device
                    )
                ),
            )
        env.termination_manager.compute()
    elif hasattr(env, "_get_dones"):
        # fallback for some Direct envs
        env.cfg.return_success_status = success
    return False
# ---- minimal helpers ----
def auto_fix_collision_issues():
    stage = omni.usd.get_context().get_stage()
    if not stage:
        return
    for prim in stage.Traverse():
        name = prim.GetName()
        if "handle" in name or "drawer" in name or "board" in name:
            if not prim.IsA(UsdGeom.Mesh):
                continue
            collision_api = UsdPhysics.MeshCollisionAPI(prim)
            if not collision_api:
                collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
            approx_attr = collision_api.GetApproximationAttr()
            current_val = approx_attr.Get()
            if current_val != "convexDecomposition":
                approx_attr.Set("convexDecomposition")

from isaaclab.utils.math import quat_inv, quat_apply

def is_grasp_phase_by_step(step_count):
    return None
def get_expert_action_pose_based(env, step_count, target, flag):
    device = env.device
    num_envs = env.num_envs
    orange_pos_w = env.scene[target].data.root_pos_w.clone()
    plate_pos_w = env.scene["Plate"].data.root_pos_w.clone()
    robot_base_pos_w = env.scene["robot"].data.root_pos_w.clone()
    robot_base_quat_w = env.scene["robot"].data.root_quat_w.clone()

    target_pos_w = orange_pos_w.clone()
    
    import math
    from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, quat_inv
    pitch = math.radians(0)

    target_quat_w = quat_from_euler_xyz(
        torch.tensor(pitch, device=device),
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
    ).repeat(num_envs, 1)

    target_quat = quat_mul(
        quat_inv(robot_base_quat_w),
        target_quat_w
    )
    def apply_triangle_offset(pos_tensor, flag, radius=0.1):
        """
        给位置张量的 x, y 轴增加等边三角形偏移量。
        Args:
            pos_tensor: (num_envs, 3) 的 target_pos_w 张量
            flag: 当前是第几个橙子 (1, 2, 3)
            radius: 距离中心的半径 (默认 0.2米)
        """

        idx = (flag - 1) % 3 
        angle = idx * (2 * math.pi / 3)
        
        offset_x = radius * math.cos(angle)
        offset_y = radius * math.sin(angle)
        
        pos_tensor[:, 0] += offset_x
        pos_tensor[:, 1] += offset_y
        
        return pos_tensor
    
    GRIPPER = 0.1
    target_pos_w[:,0] -= 0.03
    
    gripper_cmd = torch.full((num_envs,1), 1.0, device=device)
    if step_count < 120:
        gripper_cmd[:] = 1.0
        target_pos_w[:,2] += 0.1
        target_pos_w[:,2] += GRIPPER
    elif step_count < 150:
        gripper_cmd[:] = 1.0
        target_pos_w[:,2] += GRIPPER
    elif step_count < 180:
        gripper_cmd[:] = -1.0
        target_pos_w[:,2] += GRIPPER
    elif step_count < 220:
        gripper_cmd[:] = -1.0
        target_pos_w[:,2] += 0.25
    elif step_count < 320:
        gripper_cmd[:] = -1.0
        target_pos_w = plate_pos_w.clone()
        target_pos_w[:,2] += 0.25
    elif step_count < 350:
        target_pos_w = plate_pos_w.clone()
        target_pos_w[:,2] += GRIPPER + 0.1
        apply_triangle_offset(target_pos_w, flag)
        gripper_cmd[:] = -1.0
    elif step_count < 380:
        target_pos_w = plate_pos_w.clone()
        target_pos_w[:,2] += GRIPPER + 0.1
        apply_triangle_offset(target_pos_w, flag)
        gripper_cmd[:] = 1.0
    elif step_count < 420:
        target_pos_w = plate_pos_w.clone()
        target_pos_w[:,2] += 0.2
        apply_triangle_offset(target_pos_w, flag)
        gripper_cmd[:] = 1.0
    else:
        gripper_cmd[:] = 1.0
    
        
    diff_w = target_pos_w - robot_base_pos_w
    target_pos_local = quat_apply(quat_inv(robot_base_quat_w), diff_w)
    actions = torch.cat([target_pos_local, target_quat, gripper_cmd], dim=-1)
    return actions

        
def main():

    MAX_STEPS = 420
    # prepare
    task_name = getattr(args_cli, "task", None)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.use_teleop_device("so101_state_machine")
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args_cli.record:
        if args_cli.resume:
            env_cfg.recorders.dataset_export_mode = EnhanceDatasetExportMode.EXPORT_ALL_RESUME
            assert os.path.exists(args_cli.dataset_file)
        else:
            env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
            assert not os.path.exists(args_cli.dataset_file)
        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = output_file_name

        is_direct_env = "Direct" in task_name
        if is_direct_env:
            env_cfg.return_success_status = False
        else:
            if not hasattr(env_cfg.terminations, "success"):
                setattr(env_cfg.terminations, "success", None)
            env_cfg.terminations.success = TerminationTermCfg(
                func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            )
    else:
        env_cfg.recorders = None


    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    task_name = args_cli.task

    if args_cli.quality:
        env_cfg.sim.render.antialiasing_mode = "FXAA"
        env_cfg.sim.render.rendering_mode = "quality"

    # create env
    env: ManagerBasedRLEnv | DirectRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped

    # (optional) fix USD collisions
    auto_fix_collision_issues()
    resume_recorded_demo_count = 0
    
    if args_cli.record:
        try:
            del env.recorder_manager
        except Exception:
            pass
        print("Setting up recorder_manager for recording.")
        env.recorder_manager = StreamingRecorderManager(env_cfg.recorders, env)
        env.recorder_manager.flush_steps = 100
        env.recorder_manager.compression = "lzf"
        if args_cli.resume:
            resume_recorded_demo_count = env.recorder_manager._dataset_file_handler.get_num_episodes()
            print(f"Resume recording from existing dataset file with {resume_recorded_demo_count} demonstrations.")
        current_recorded_demo_count = resume_recorded_demo_count

    # init / reset
    if hasattr(env, "initialize"):
        env.initialize()
    env.reset()
    # Print debug values right after reset to verify randomization actually applied to physics
    try:
        target_name = f"Orange00{flag}"
        print("=== Post-reset diagnostics ===")
        print("physics_dt:", getattr(env, "physics_dt", None))
        if target_name in env.scene:
            phys_pos = env.scene[target_name].data.root_pos_w.clone()
            print(f"{target_name} physics root_pos_w:", phys_pos[0].cpu().numpy() if phys_pos.ndim>1 else phys_pos.cpu().numpy())
        else:
            print(f"{target_name} not found in scene after reset")
        # also print robot base to ensure base pose consistent
        if "robot" in env.scene:
            rb = env.scene["robot"].data.root_pos_w.clone()
            print("robot root_pos_w:", rb[0].cpu().numpy() if rb.ndim>1 else rb.cpu().numpy())
        print("===============================")
    except Exception as e:
        print("Post-reset diagnostic failed:", e)
    # try to write lower stiffness / higher damping to sim (best-effort)
    try:
        # get joint names / count for printing
        robot_prim = env.scene["robot"]
        joint_names = robot_prim.data.joint_names if hasattr(robot_prim, "data") and hasattr(robot_prim.data, "joint_names") else None
    except Exception:
        joint_names = None

    if joint_names is not None:
        print("Detected joints:", joint_names)
        # example starting values - tune these per-joint as needed
        n_joints = len(joint_names)
        # conservative starting point (you can further reduce stiffness)
        start_stiffness = [1000.0] * n_joints
        start_damping = [30.0] * n_joints
    else:
        start_stiffness = None
        start_damping = None

    # attempt to write gains into simulation (API differs across versions; best-effort)
    if start_stiffness is not None:
        try:
            # preferred helper if available
            robot_art.write_joint_stiffness_to_sim(start_stiffness)
            robot_art.write_joint_damping_to_sim(start_damping)
            print("Wrote joint stiffness/damping via robot_art helper.")
        except Exception:
            # fallback: print warning - user may need to set gains in asset or use other API
            print("Warning: robot_art helper methods for stiffness/damping not available. If you need to set them, check robot_art API or the asset YAML.")

    # rate limiter
    rate_limiter = None
    if args_cli.step_hz:
        rate_limiter = type("R", (), {"hz": args_cli.step_hz, "sleep": lambda self, e: time.sleep(1.0/self.hz)})()
    step_count = 0

    prev_gripper_target = torch.full((env.num_envs, 1), 1.0, device=env.device)

    

    max_gripper_delta_normal = 0.04
    max_gripper_delta_during_grasp = 0.01

    # simple step-based fallback

    # auto detector: distance + relative speed + gripper-target
    def is_grasp_phase_auto(env_local, orange_name: str, prev_gripper_t: torch.Tensor,
                            dist_thresh: float = 0.06, rel_vel_thresh: float = 0.08, gripper_close_threshold: float = 0.7):
        """
        Return (is_grasp: bool, info: dict).
        - device-safe: newly created tensors follow the device of scene tensors (cuda or cpu).
        - prev_gripper_t should be a tensor on the env device (if not, we'll move it).
        """
        info = {}
        try:
            orange = env_local.scene[orange_name]
            orange_pos = None
            device = None

            # obtain orange pos and device
            if hasattr(orange.data, "root_pos_w"):
                orange_pos = orange.data.root_pos_w  # (num_envs,3) tensor
                device = orange_pos.device

            # fallback device: try env_local.device or cpu
            if device is None:
                device = getattr(env_local, "device", torch.device("cpu"))

            # robot and ee
            robot_local = env_local.scene["robot"]
            try:
                ee_idx = robot_local.data.body_names.index("gripper")
            except Exception:
                ee_idx = 0

            # ee pos tensor (ensure on same device)
            ee_pos = robot_local.data.body_state_w[:, ee_idx, :3]  # (num_envs,3)
            if ee_pos is not None:
                device = ee_pos.device

            # if orange_pos missing -> cannot detect
            if orange_pos is None:
                return False, {"error": "orange pos not available"}

            # horizontal / full distance (tensors on same device)
            horiz_dist = torch.norm((orange_pos - ee_pos)[:, :2], dim=-1)  # (num_envs,)
            full_dist = torch.norm((orange_pos - ee_pos), dim=-1)
            info['horiz_dist'] = horiz_dist.detach().cpu().numpy().tolist()
            info['full_dist'] = full_dist.detach().cpu().numpy().tolist()

            # read linear velocities if available; default zeros on correct device
            obj_speed = torch.zeros((orange_pos.shape[0],), device=device)
            ee_speed = torch.zeros((orange_pos.shape[0],), device=device)
            if hasattr(orange.data, "root_linvel_w"):
                try:
                    obj_v = orange.data.root_linvel_w  # (num_envs,3)
                    obj_speed = torch.norm(obj_v, dim=-1).to(device)
                except Exception:
                    pass
            if hasattr(robot_local.data, "body_linvel_w"):
                try:
                    v_ee = robot_local.data.body_linvel_w[:, ee_idx, :]  # (num_envs,3)
                    ee_speed = torch.norm(v_ee, dim=-1).to(device)
                except Exception:
                    pass
            rel_speed = torch.minimum(obj_speed, ee_speed)
            info['obj_speed'] = obj_speed.detach().cpu().numpy().tolist() if obj_speed.numel() else None
            info['ee_speed'] = ee_speed.detach().cpu().numpy().tolist() if ee_speed.numel() else None
            info['rel_speed'] = rel_speed.detach().cpu().numpy().tolist()

            # gripper target: ensure prev_gripper_t is on same device
            try:
                if not isinstance(prev_gripper_t, torch.Tensor):
                    # try to coerce
                    prev_gripper_t = torch.as_tensor(prev_gripper_t, device=device)
                else:
                    prev_gripper_t = prev_gripper_t.to(device)
                gripper_target = prev_gripper_t.view(-1).detach().cpu().numpy().tolist()
            except Exception:
                gripper_target = None
            info['gripper_target'] = gripper_target

            # Decide masks (create boolean tensors on same device)
            near_mask = (horiz_dist <= torch.as_tensor(dist_thresh, device=device))
            slow_mask = (rel_speed <= torch.as_tensor(rel_vel_thresh, device=device))

            if gripper_target is not None:
                # gripper_target is a python list; build a boolean tensor on device
                try:
                    gripper_mask = torch.tensor([gt < gripper_close_threshold for gt in gripper_target], dtype=torch.bool, device=device)
                except Exception:
                    gripper_mask = torch.zeros_like(near_mask, dtype=torch.bool, device=device)
            else:
                gripper_mask = torch.zeros_like(near_mask, dtype=torch.bool, device=device)

            is_grasp_tensor = near_mask & slow_mask & gripper_mask
            # convert to python bool for single-env; if multi-env return list
            is_grasp_list = is_grasp_tensor.detach().cpu().numpy().tolist()
            is_grasp_any = any(is_grasp_list)
            return is_grasp_any, info

        except Exception as e:
            return False, {"error": str(e)}

    flag = 1  # which orange to pick
    start_record_state = False
    # main loop (NO pos/quaternion smoothing; gripper still clamped)
    while simulation_app.is_running():
        # choose Lula-based actions (supports num_envs>1)
        dt = env.physics_dt
        env.scene.update(dt)
        
        # 放在 while simulation_app.is_running(): 的顶部，紧跟 env.scene.update(dt) 之后或 actions 之前
        if args_cli.record and not start_record_state:
            print("Auto: enabling recorder / start recording state.")
            start_record_state = True
            # Best-effort: call recorder start/enable APIs if available
            try:
                rm = getattr(env, "recorder_manager", None)
                if rm is not None:
                    # try common start-like method names across versions
                    for meth in ("start", "start_recording", "begin_record", "enable", "start_capture"):
                        if hasattr(rm, meth) and callable(getattr(rm, meth)):
                            print(f"Calling recorder_manager.{meth}()")
                            getattr(rm, meth)()
                            break
                    # also try to ensure flush_steps/compression are set
                    try:
                        if hasattr(rm, "flush_steps"):
                            rm.flush_steps = getattr(rm, "flush_steps", 100)
                        if hasattr(rm, "compression"):
                            rm.compression = getattr(rm, "compression", "lzf")
                    except Exception:
                        pass
                else:
                    print("Warning: env.recorder_manager is None — recorder not created.")
            except Exception as e:
                print("Failed to start recorder_manager automatically:", e)
        actions = get_expert_action_pose_based(env, step_count, target=f"Orange00{flag}", flag=flag)
        # 强制把 prev_gripper_target 设为你刚才计算出的 desired（从 actions 取）
        # 这会让平滑阶段认为上次目标就是现在的目标，从而直接生效
        prev_gripper_target = actions[:, -1].clone().unsqueeze(-1)

        # ensure dtype/device
        actions = actions.to(device=env.device, dtype=torch.float32)

        # Decide grasp phase (auto or fallback)
        is_grasp_auto, grasp_info = is_grasp_phase_auto(env, f"Orange00{flag}", prev_gripper_target,
                                                       dist_thresh=0.06, rel_vel_thresh=0.08, gripper_close_threshold=0.7)
        if not is_grasp_auto:
            is_grasp = is_grasp_phase_by_step(step_count)
        else:
            is_grasp = is_grasp_auto

        # --- GRIPPER: clamp per-frame to avoid instant violent close ---
        desired_gripper = actions[:, -1].clone().unsqueeze(-1)  # (num_envs,1)
        cur_gripper = prev_gripper_target  # previous target
        # use different delta in grasp vs normal
        if isinstance(is_grasp, bool):
            if is_grasp:
                use_gripper_delta = torch.full((env.num_envs,), max_gripper_delta_during_grasp, device=env.device, dtype=torch.float32)
            else:
                use_gripper_delta = torch.full((env.num_envs,), max_gripper_delta_normal, device=env.device, dtype=torch.float32)
        else:
            # per-env list-like
            use_gripper_delta = torch.tensor([max_gripper_delta_during_grasp if g else max_gripper_delta_normal for g in (is_grasp if hasattr(is_grasp, "__len__") else [is_grasp])], device=env.device, dtype=torch.float32)
            if use_gripper_delta.numel() != env.num_envs:
                use_gripper_delta = torch.full((env.num_envs,), max_gripper_delta_normal, device=env.device, dtype=torch.float32)

        delta_gr = torch.clamp(desired_gripper - cur_gripper, -use_gripper_delta.unsqueeze(-1), use_gripper_delta.unsqueeze(-1))
        smooth_gripper = cur_gripper + delta_gr
        actions[:, -1] = smooth_gripper.squeeze(-1)
        prev_gripper_target = smooth_gripper.clone()

        # --- POSITION / QUATERNION: NO SMOOTHING, directly send ---
        # action layout expected: [pos_x,pos_y,pos_z, quat_x,quat_y,quat_z,quat_w, gripper]
        smoothed_actions = actions.to(device=env.device, dtype=torch.float32)

        step_count += 1
        
        from isaaclab.managers import SceneEntityCfg
        if step_count >= MAX_STEPS and flag >= 3:
            # --- 判断本次 episode 是否被认为成功（使用 env 的 task_done） ---
            try:
                print(f"完成一轮（{step_count} 步）, 检查任务成功状态...")
                success_tensor = task_done(
                    env,
                    oranges_cfg=[
                        SceneEntityCfg("Orange001"),
                        SceneEntityCfg("Orange002"),
                        SceneEntityCfg("Orange003"),
                    ],
                    plate_cfg=SceneEntityCfg("Plate"),
                )
                # 多 env 时，我们认为当所有 env 都为 True 才视为 success（根据你的需要可改为 any）
                success = bool(success_tensor.all().item())
                print("任务成功状态:", success)
            except Exception as e:
                print("task_done failed:", e)
                success = False

            # ✅ 只有成功时，才“告诉 recorder 这一轮要存”
            if args_cli.record and success:
                print("✅ 任务成功，标记本次演示为 SUCCESS")
                manual_terminate(env, True)
                print("SUCCESS!!!!!!!")
                current_recorded_demo_count += 1
            else:
                print("❌ 任务失败，标记本次演示为 FAILURE")
                manual_terminate(env, False)
                
            if (
                args_cli.record
                and env.recorder_manager.exported_successful_episode_count + resume_recorded_demo_count
                > current_recorded_demo_count
            ):
                current_recorded_demo_count = (
                    env.recorder_manager.exported_successful_episode_count + resume_recorded_demo_count
                )
                print(f"Recorded {current_recorded_demo_count} successful demonstrations.")
            if (
                args_cli.record
                and args_cli.num_demos > 0
                and env.recorder_manager.exported_successful_episode_count + resume_recorded_demo_count
                >= args_cli.num_demos
            ):
                print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                break
            
            # --- bookkeeping: 更新 recorder 导出的计数并判断是否达到 --num_demos ---
            env.reset()
            flag = 1
            step_count = 0    
            if args_cli.record and args_cli.num_demos > 0 and current_recorded_demo_count >= args_cli.num_demos:
                print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                break
        elif step_count >= MAX_STEPS and flag < 3:
            step_count = 0

            # --- 不论成功与否，都重置环境（你的要求） ---
            print(f"完成一轮（{step_count} 步）, 重置环境，准备下一个橙子 (flag {flag} -> {flag+1})")

            # --- 更新 step_count/flag 控制逻辑（和你原来逻辑一致） ---
            flag += 1
        else:
            env.step(smoothed_actions)
            pass
        
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

