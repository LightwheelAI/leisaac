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

from isaaclab.managers import DatasetExportMode, TerminationTermCfg
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_inv, quat_mul
from isaaclab.managers import SceneEntityCfg
from leisaac.enhance.managers import StreamingRecorderManager, EnhanceDatasetExportMode
from leisaac.tasks.pick_orange.mdp import task_done
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim

import torch
import math
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

def apply_triangle_offset(pos_tensor, orange_now, radius=0.1):
    """
    Apply an equilateral triangle offset on the x-y plane.

    The offset places objects evenly around a circle, forming the vertices
    of an equilateral triangle. This is used to arrange multiple oranges
    on the plate without overlap.

    Args:
        pos_tensor (torch.Tensor):
            Position tensor of shape (num_envs, 3) in world coordinates.
        orange_now (int):
            Index of the current orange (1, 2, or 3).
        radius (float, optional):
            Distance from the center of the plate to each triangle vertex.
            Defaults to 0.1 meters.

    Returns:
        torch.Tensor:
            The input position tensor with the triangle offset applied.
    """

    idx = (orange_now - 1) % 3
    angle = idx * (2 * math.pi / 3)

    offset_x = radius * math.cos(angle)
    offset_y = radius * math.sin(angle)

    pos_tensor[:, 0] += offset_x
    pos_tensor[:, 1] += offset_y

    return pos_tensor

def state_machine(env, step_count, target, orange_now):
    """
    Finite state machine for a pick-and-place task with a robot gripper.

    This function generates low-level action commands based on the current
    simulation step. The robot approaches an orange, grasps it, lifts it,
    moves it above a plate, and places it at one vertex of an equilateral
    triangle arrangement.

    Args:
        env:
            The simulation environment instance. It is expected to provide
            access to scene objects, device information, and number of parallel
            environments.
        step_count (int):
            The current timestep of the episode, used to switch between
            different phases of the state machine.
        target (str):
            The name of the target object (e.g., an orange) in the scene.
        orange_now (int):
            Index of the current orange being placed (1, 2, or 3). This
            determines the triangle offset on the plate.

    Returns:
        torch.Tensor:
            Action tensor of shape (num_envs, action_dim), containing:
            - target position in the robot base frame (x, y, z)
            - target orientation as a quaternion (x, y, z, w)
            - gripper command (open / close)
    """

    device = env.device
    num_envs = env.num_envs

    orange_pos_w = env.scene[target].data.root_pos_w.clone()
    plate_pos_w = env.scene["Plate"].data.root_pos_w.clone()
    robot_base_pos_w = env.scene["robot"].data.root_pos_w.clone()
    robot_base_quat_w = env.scene["robot"].data.root_quat_w.clone()

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
    GRIPPER = 0.1
    
    gripper_cmd = torch.full((num_envs, 1), 1.0, device=device)

    #move above orange
    if step_count < 120:
        target_pos_w = orange_pos_w.clone()
        target_pos_w[:, 0] -= 0.03
        gripper_cmd[:] = 1.0
        target_pos_w[:, 2] += 0.1 + GRIPPER
    #lower to orange
    elif step_count < 150:
        target_pos_w = orange_pos_w.clone()
        target_pos_w[:, 0] -= 0.03
        gripper_cmd[:] = 1.0
        target_pos_w[:, 2] += GRIPPER
    #close gripper
    elif step_count < 180:
        target_pos_w = orange_pos_w.clone()
        target_pos_w[:, 0] -= 0.03
        gripper_cmd[:] = -1.0
        target_pos_w[:, 2] += GRIPPER
    #lift orange
    elif step_count < 220:
        target_pos_w = orange_pos_w.clone()
        target_pos_w[:, 0] -= 0.03
        gripper_cmd[:] = -1.0
        target_pos_w[:, 2] += 0.25
    #move above plate
    elif step_count < 320:
        gripper_cmd[:] = -1.0
        target_pos_w = plate_pos_w.clone()
        target_pos_w[:, 2] += 0.25
    #lower to plate
    elif step_count < 350:
        target_pos_w = plate_pos_w.clone()
        target_pos_w[:, 2] += GRIPPER + 0.1
        apply_triangle_offset(target_pos_w, orange_now)
        gripper_cmd[:] = -1.0
    # release orange
    elif step_count < 380:
        target_pos_w = plate_pos_w.clone()
        target_pos_w[:, 2] += GRIPPER + 0.1
        apply_triangle_offset(target_pos_w, orange_now)
        gripper_cmd[:] = 1.0
    # lift gripper
    elif step_count < 420:
        target_pos_w = plate_pos_w.clone()
        target_pos_w[:, 2] += 0.2
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

    # Temporarily use so101_state_machine as the teleoperation device
    device = "so101_state_machine"
    env_cfg.use_teleop_device(device)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    task_name = args_cli.task
    
    # Timeout and termination preprocessing
    is_direct_env = "Direct" in task_name
    if is_direct_env:
        env_cfg.never_time_out = True
        env_cfg.auto_terminate = True
    else:
        # Modify termination configuration
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        if hasattr(env_cfg.terminations, "success"):
            env_cfg.terminations.success = None

    # Recorder preprocessing & manual success-termination preprocessing
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
                ), "The dataset file does not exist. Do not use '--resume' when recording a new dataset."
            else:
                env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
                assert not os.path.exists(
                    args_cli.dataset_file
                ), "The dataset file already exists. Use '--resume' to resume recording."
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

    # Create environment
    env: ManagerBasedRLEnv | DirectRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped
        
    if args_cli.record:
        del env.recorder_manager
        if args_cli.use_lerobot_recorder:
            from leisaac.enhance.datasets.lerobot_dataset_handler import LeRobotDatasetCfg
            from leisaac.enhance.managers.lerobot_recorder_manager import LeRobotRecorderManager

            dataset_cfg = LeRobotDatasetCfg(
                repo_id=args_cli.lerobot_dataset_repo_id,
                fps=args_cli.lerobot_dataset_fps,
            )
            env.recorder_manager = LeRobotRecorderManager(env_cfg.recorders, dataset_cfg, env)
        else:
            env.recorder_manager = StreamingRecorderManager(env_cfg.recorders, env)
            env.recorder_manager.flush_steps = 100
            env.recorder_manager.compression = "lzf"

    # Rate limiter
    rate_limiter = RateLimiter(args_cli.step_hz)

    # Initialize / reset
    if hasattr(env, "initialize"):
        env.initialize()
    env.reset()

    resume_recorded_demo_count = 0
    if args_cli.record and args_cli.resume:
        resume_recorded_demo_count = env.recorder_manager._dataset_file_handler.get_num_episodes()
        print(f"Resuming recording from existing dataset with {resume_recorded_demo_count} demonstrations.")
    current_recorded_demo_count = resume_recorded_demo_count
    
    step_count = 0
    orange_now = 1  # Index of the current orange to pick
    start_record_state = False
    
    while simulation_app.is_running() and not simulation_app.is_exiting():
        # Place this at the top of the loop, right after env.scene.update(dt)
        # or before computing actions
        if env.cfg.dynamic_reset_gripper_effort_limit:
            dynamic_reset_gripper_effort_limit_sim(env, device)

        actions = state_machine(env, step_count, target=f"Orange00{orange_now}", orange_now=orange_now)
        if step_count >= MAX_STEPS and orange_now >= 3:
            # --- Check whether the current episode is considered successful ---
            try:
                print(f"Completed one cycle ({step_count} steps). Checking task success status...")
                success_tensor = task_done(
                    env,
                    oranges_cfg=[
                        SceneEntityCfg("Orange001"),
                        SceneEntityCfg("Orange002"),
                        SceneEntityCfg("Orange003"),
                    ],
                    plate_cfg=SceneEntityCfg("Plate"),
                )
                # For multiple environments, consider success only if all envs succeed
                success = bool(success_tensor.all().item())
                print("Task success status:", success)
            except Exception as e:
                print("Task failed due to:", e)
                success = False

            if start_record_state:
                if args_cli.record:
                    print("Stop recording.")
                start_record_state = False

            # Only mark the episode as successful if the task succeeds
            if args_cli.record and success:
                print("✅ Task succeeded. Marking this demonstration as SUCCESS.")
                auto_terminate(env, True)
                print("SUCCESS.")
                current_recorded_demo_count += 1
            else:
                print("❌ Task failed. Marking this demonstration as FAILURE.")
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
                print(f"All {args_cli.num_demos} demonstrations have been recorded. Exiting.")
                break

            # Reset environment and bookkeeping
            env.reset()
            orange_now = 1
            step_count = 0

            if args_cli.record and args_cli.num_demos > 0 and current_recorded_demo_count >= args_cli.num_demos:
                print(f"All {args_cli.num_demos} demonstrations have been recorded. Exiting.")
                break

        elif step_count >= MAX_STEPS and orange_now < 3:
            step_count = 0

            # Reset environment regardless of success or failure
            print(
                f"Completed one cycle. Resetting environment and moving to next orange "
                f"(orange_now {orange_now} -> {orange_now + 1})."
            )

            # Update control variables (same logic as original)
            orange_now += 1
        else:
            if not start_record_state:
                if args_cli.record:
                    print("Start recording.")
                start_record_state = True
            env.step(actions)
            step_count += 1
        if rate_limiter:
            rate_limiter.sleep(env)

    if args_cli.record and hasattr(env.recorder_manager, "finalize"):
        env.recorder_manager.finalize()

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    # run the main function
    main()

