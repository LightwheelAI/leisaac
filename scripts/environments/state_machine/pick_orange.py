"""Script to generate data using state machine with leisaac manipulation environments."""

"""Launch Isaac Sim Simulator first."""
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

# add argparse arguments
parser = argparse.ArgumentParser(description="leisaac data generation script for pick_orange task.")
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

from leisaac.enhance.managers import StreamingRecorderManager, EnhanceDatasetExportMode
from leisaac.tasks.pick_orange.mdp import task_done
from isaaclab.managers import DatasetExportMode, TerminationTermCfg
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg

class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def auto_terminate(env: ManagerBasedRLEnv | DirectRLEnv, success: bool):
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

from isaaclab.utils.math import quat_apply

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
    rate_limiter = RateLimiter(args_cli.step_hz)
    step_count = 0
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
                auto_terminate(env, True)
                print("SUCCESS!!!!!!!")
                current_recorded_demo_count += 1
            else:
                print("❌ 任务失败，标记本次演示为 FAILURE")
                auto_terminate(env, False)
                
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
            env.step(actions)
            pass
        if rate_limiter:
            rate_limiter.sleep(env)
        
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    # run the main function
    main()

