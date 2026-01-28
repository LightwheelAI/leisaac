"""Script to generate data using state machine with leisaac manipulation environments."""

"""Launch Isaac Sim Simulator first."""
import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
import argparse

import os
import time

from isaaclab.app import AppLauncher

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
parser.add_argument("--use_lerobot_recorder", action="store_true", help="whether to use lerobot recorder.")
parser.add_argument("--lerobot_dataset_repo_id", type=str, default=None, help="Lerobot Dataset repository ID.")
parser.add_argument("--lerobot_dataset_fps", type=int, default=30, help="Lerobot Dataset frames per second.")

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

from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_inv, quat_mul

import torch
import gymnasium as gym

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

def get_expert_action_pose_based(env, step_count, target, orange_now):
    device = env.device
    num_envs = env.num_envs
    orange_pos_w = env.scene[target].data.root_pos_w.clone()
    plate_pos_w = env.scene["Plate"].data.root_pos_w.clone()
    robot_base_pos_w = env.scene["robot"].data.root_pos_w.clone()
    robot_base_quat_w = env.scene["robot"].data.root_quat_w.clone()

    target_pos_w = orange_pos_w.clone()
    
    import math
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
    def apply_triangle_offset(pos_tensor, orange_now, radius=0.1):
        """
        给位置张量的 x, y 轴增加等边三角形偏移量。
        Args:
            pos_tensor: (num_envs, 3) 的 target_pos_w 张量
            orange_now: 当前是第几个橙子 (1, 2, 3)
            radius: 距离中心的半径 (默认 0.2米)
        """

        idx = (orange_now - 1) % 3 
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
        apply_triangle_offset(target_pos_w, orange_now)
        gripper_cmd[:] = -1.0
    elif step_count < 380:
        target_pos_w = plate_pos_w.clone()
        target_pos_w[:,2] += GRIPPER + 0.1
        apply_triangle_offset(target_pos_w, orange_now)
        gripper_cmd[:] = 1.0
    elif step_count < 420:
        target_pos_w = plate_pos_w.clone()
        target_pos_w[:,2] += 0.2
        apply_triangle_offset(target_pos_w, orange_now)
        gripper_cmd[:] = 1.0
    else:
        gripper_cmd[:] = 1.0
    
        
    diff_w = target_pos_w - robot_base_pos_w
    target_pos_local = quat_apply(quat_inv(robot_base_quat_w), diff_w)
    actions = torch.cat([target_pos_local, target_quat, gripper_cmd], dim=-1)
    return actions

MAX_STEPS = 420

def main() -> None:
    """
    Run a pick-orange state machine in an Isaac Lab manipulation environment.

    Creates the environment, initializes the pick-and-place state machine for
    picking an orange, and runs the main simulation loop until the application
    is closed.

    Returns:
        None
    """
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.use_teleop_device("so101_state_machine")
    task_name = args_cli.task   
    is_direct_env = "Direct" in task_name
    
    # timeout and terminate preprocess
    if is_direct_env:
        env_cfg.never_time_out = True
        env_cfg.manual_terminate = True
    else:
        # modify configuration
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        if hasattr(env_cfg.terminations, "success"):
            env_cfg.terminations.success = None
            
    # recorder preprocess & manual success terminate preprocess
    if args_cli.record:
        if args_cli.use_lerobot_recorder:
            if args_cli.resume:
                env_cfg.recorders.dataset_export_mode = EnhanceDatasetExportMode.EXPORT_SUCCEEDED_ONLY_RESUME
            else:
                env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
        else:
            if args_cli.resume:
                env_cfg.recorders.dataset_export_mode = EnhanceDatasetExportMode.EXPORT_ALL_RESUME
                assert os.path.exists(
                    args_cli.dataset_file
                ), "the dataset file does not exist, please don't use '--resume' if you want to record a new dataset"
            else:
                env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
                assert not os.path.exists(
                    args_cli.dataset_file
                ), "the dataset file already exists, please use '--resume' to resume recording"
        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = output_file_name
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
        
    # create env
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    env: ManagerBasedRLEnv | DirectRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped
    
    if args_cli.record:
        del env.recorder_manager
        if args_cli.use_lerobot_recorder:
            from leisaac.enhance.datasets.lerobot_dataset_handler import (
                LeRobotDatasetCfg,
            )
            from leisaac.enhance.managers.lerobot_recorder_manager import (
                LeRobotRecorderManager,
            )

            dataset_cfg = LeRobotDatasetCfg(
                repo_id=args_cli.lerobot_dataset_repo_id,
                fps=args_cli.lerobot_dataset_fps,
            )
            env.recorder_manager = LeRobotRecorderManager(env_cfg.recorders, dataset_cfg, env)
        else:
            env.recorder_manager = StreamingRecorderManager(env_cfg.recorders, env)
            env.recorder_manager.flush_steps = 100
            env.recorder_manager.compression = "lzf"
            
    # rate limiter        
    rate_limiter = RateLimiter(args_cli.step_hz)
    
    # init / reset
    if hasattr(env, "initialize"):
        env.initialize()
    env.reset()
    
    resume_recorded_demo_count = 0
    if args_cli.record and args_cli.resume:
        resume_recorded_demo_count = env.recorder_manager._dataset_file_handler.get_num_episodes()
        print(f"Resume recording from existing dataset file with {resume_recorded_demo_count} demonstrations.")
    current_recorded_demo_count = resume_recorded_demo_count
    
    step_count = 0
    orange_now = 1  # which orange to pick
    start_record_state = False
    
    while simulation_app.is_running() and not simulation_app.is_exiting():
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
        actions = get_expert_action_pose_based(env, step_count, target=f"Orange00{orange_now}", orange_now=orange_now)

        step_count += 1
        
        from isaaclab.managers import SceneEntityCfg
        if step_count >= MAX_STEPS and orange_now >= 3:
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
            orange_now = 1
            step_count = 0    
            if args_cli.record and args_cli.num_demos > 0 and current_recorded_demo_count >= args_cli.num_demos:
                print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                break
        elif step_count >= MAX_STEPS and orange_now < 3:
            step_count = 0

            # --- 不论成功与否，都重置环境（你的要求） ---
            print(f"完成一轮（{step_count} 步）, 重置环境，准备下一个橙子 (orange_now {orange_now} -> {orange_now+1})")

            # --- 更新 step_count/orange_now 控制逻辑（和你原来逻辑一致） ---
            orange_now += 1
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

