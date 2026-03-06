"""Unified data generation script using state machines.

Selects the appropriate state machine based on --task and runs the recording loop.

Usage:
    python scripts/datagen/state_machine/generate.py \\
        --task LeIsaac-SO101-PickOrange-v0 \\
        --num_envs 1 --device cuda --enable_cameras \\
        --record --dataset_file ./datasets/pick_orange.hdf5 --num_demos 50
"""

"""Launch Isaac Sim Simulator first."""
import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import os
import time

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="State machine data generation for LeIsaac tasks.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed for the environment.")
parser.add_argument("--record", action="store_true", help="Whether to enable record function.")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos."
)
parser.add_argument("--resume", action="store_true", help="Whether to resume recording in the existing dataset file.")
parser.add_argument(
    "--num_demos", type=int, default=1, help="Number of demonstrations to record. Set to 0 for infinite."
)
parser.add_argument("--quality", action="store_true", help="Whether to enable quality render mode.")
parser.add_argument("--use_lerobot_recorder", action="store_true", help="Whether to use lerobot recorder.")
parser.add_argument("--lerobot_dataset_repo_id", type=str, default=None, help="Lerobot Dataset repository ID.")
parser.add_argument("--lerobot_dataset_fps", type=int, default=30, help="Lerobot Dataset frames per second.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import DatasetExportMode, TerminationTermCfg
from isaaclab_tasks.utils import parse_env_cfg

from leisaac.enhance.managers import EnhanceDatasetExportMode, StreamingRecorderManager
from leisaac.datagen.state_machine import PickOrangeStateMachine
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim

import leisaac.tasks  # noqa: F401

# ---------------------------------------------------------------------------
# Task registry: maps gym task id → (StateMachineClass, device_type)
# ---------------------------------------------------------------------------
TASK_REGISTRY = {
    "LeIsaac-SO101-PickOrange-v0": (PickOrangeStateMachine, "so101_state_machine"),
}


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
    if hasattr(env, "termination_manager"):
        if success:
            env.termination_manager.set_term_cfg(
                "success",
                TerminationTermCfg(func=lambda env: torch.ones(env.num_envs, dtype=torch.bool, device=env.device)),
            )
        else:
            env.termination_manager.set_term_cfg(
                "success",
                TerminationTermCfg(func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)),
            )
        env.termination_manager.compute()
    elif hasattr(env, "_get_dones"):
        env.cfg.return_success_status = success


def main():
    """Run a state machine in a LeIsaac manipulation environment."""
    task_name = args_cli.task
    if task_name not in TASK_REGISTRY:
        raise ValueError(
            f"Task '{task_name}' is not registered in TASK_REGISTRY.\n"
            f"Available tasks: {list(TASK_REGISTRY.keys())}"
        )
    SMClass, device = TASK_REGISTRY[task_name]

    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    # create directory if it does not exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    env_cfg = parse_env_cfg(task_name, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.use_teleop_device(device)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())

    # timeout and terminate preprocess
    is_direct_env = "Direct" in task_name
    if is_direct_env:
        env_cfg.never_time_out = True
        env_cfg.auto_terminate = True
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

    # create environment
    env: ManagerBasedRLEnv | DirectRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped

    # disable gravity for every robot link prim
    import omni.usd
    from pxr import PhysxSchema, UsdPhysics

    _stage = omni.usd.get_context().get_stage()
    for _prim in _stage.Traverse():
        if "Robot" in str(_prim.GetPath()) and _prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(_prim).CreateDisableGravityAttr(True)

    # replace the original recorder manager with the streaming recorder manager or lerobot recorder manager
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

    rate_limiter = RateLimiter(args_cli.step_hz)

    # reset environment
    if hasattr(env, "initialize"):
        env.initialize()
    env.reset()

    # one-time state machine setup (e.g. FK calibration)
    sm = SMClass()
    sm.setup(env)
    sm.reset()

    resume_recorded_demo_count = 0
    if args_cli.record and args_cli.resume:
        resume_recorded_demo_count = env.recorder_manager._dataset_file_handler.get_num_episodes()
        print(f"Resume recording from existing dataset file with {resume_recorded_demo_count} demonstrations.")
    current_recorded_demo_count = resume_recorded_demo_count

    start_record_state = False

    while simulation_app.is_running() and not simulation_app.is_exiting():
        # run everything in inference mode
        with torch.inference_mode():
            if env.cfg.dynamic_reset_gripper_effort_limit:
                dynamic_reset_gripper_effort_limit_sim(env, device)

            if sm.is_episode_done:
                try:
                    success = sm.check_success(env)
                except Exception as e:
                    print("Success check failed:", e)
                    success = False

                if start_record_state:
                    if args_cli.record:
                        print("Stop Recording!!!")
                    start_record_state = False

                if args_cli.record and success:
                    auto_terminate(env, True)
                    current_recorded_demo_count += 1
                else:
                    auto_terminate(env, False)

                # print out the current demo count if it has changed
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

                env.reset()
                sm.reset()

                if args_cli.record and args_cli.num_demos > 0 and current_recorded_demo_count >= args_cli.num_demos:
                    print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                    break
            else:
                if not start_record_state:
                    if args_cli.record:
                        print("Start Recording!!!")
                    start_record_state = True

                sm.pre_step(env)
                actions = sm.get_action(env)
                env.step(actions)
                sm.advance()

            if rate_limiter:
                rate_limiter.sleep(env)

    # finalize the recorder manager
    if args_cli.record and hasattr(env.recorder_manager, "finalize"):
        env.recorder_manager.finalize()
    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    # run the main function
    main()
