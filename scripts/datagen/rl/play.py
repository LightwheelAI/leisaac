"""Play or record a trained leisaac RL policy.

Use --record to save episodes to HDF5. Without --record, runs in visualization mode only.
"""

import os
import sys

# Make cli_args importable (local module in IsaacLab's rsl_rl script directory)
_ISAACLAB_RSL_RL_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "dependencies",
    "IsaacLab",
    "scripts",
    "reinforcement_learning",
    "rsl_rl",
)
sys.path.insert(0, os.path.abspath(_ISAACLAB_RSL_RL_DIR))

import argparse
import signal

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play or record a trained leisaac RL policy.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--num_episodes", type=int, default=0, help="Total episodes to run (0 = infinite).")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# HDF5 recording
parser.add_argument("--record", action="store_true", help="Save episodes to HDF5 with success/failure tags.")
parser.add_argument(
    "--resume_recording", action="store_true", help="Append to an existing dataset file instead of creating a new one."
)
parser.add_argument("--dataset_file", type=str, default="./datasets/rl_eval.hdf5", help="Output HDF5 file path.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import leisaac.tasks  # noqa: F401
import torch
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import DatasetExportMode
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from leisaac.enhance.managers import EnhanceDatasetExportMode, StreamingRecorderManager
from rsl_rl.runners import OnPolicyRunner


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play or record with RSL-RL agent."""
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # resolve checkpoint path
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # configure HDF5 recording
    if args_cli.record:
        output_dir = os.path.dirname(os.path.abspath(args_cli.dataset_file))
        output_filename = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
        if args_cli.resume_recording:
            env_cfg.recorders.dataset_export_mode = EnhanceDatasetExportMode.EXPORT_ALL_RESUME
            assert os.path.exists(
                args_cli.dataset_file
            ), "Dataset file does not exist. Remove --resume_recording to create a new one."
        else:
            env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
            assert not os.path.exists(
                args_cli.dataset_file
            ), "Dataset file already exists. Use --resume_recording to append, or choose a different path."
        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = output_filename
    else:
        env_cfg.recorders = None

    env = gym.make(args_cli.task, cfg=env_cfg)

    # replace recorder manager with streaming version for HDF5 recording
    if args_cli.record:
        unwrapped = env.unwrapped
        del unwrapped.recorder_manager
        unwrapped.recorder_manager = StreamingRecorderManager(env_cfg.recorders, unwrapped)
        unwrapped.recorder_manager.flush_steps = 100
        unwrapped.recorder_manager.compression = "lzf"

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.step_dt
    episode_count = 0
    success_count = 0
    interrupted = False

    def signal_handler(signum, frame):
        nonlocal interrupted
        interrupted = True
        print("\n[INFO] Interrupted. Cleaning up...")

    original_sigint_handler = signal.signal(signal.SIGINT, signal_handler)

    try:
        obs = env.get_observations()
        while (
            simulation_app.is_running()
            and not interrupted
            and (args_cli.num_episodes <= 0 or episode_count < args_cli.num_episodes)
        ):
            start_time = time.time()
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, extras = env.step(actions)

            time_outs = extras.get("time_outs", torch.zeros_like(dones))
            finished = dones.bool()
            if finished.any():
                successes = finished & ~time_outs.bool()
                for i in range(finished.shape[0]):
                    if finished[i]:
                        print("Episode success!" if successes[i] else "Episode failed!")
                episode_count += int(finished.sum().item())
                success_count += int(successes.sum().item())
                print(f"Success rate: {success_count / episode_count:.1%} ({success_count}/{episode_count})")

            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        import traceback

        print(f"\n[ERROR] {e}\n")
        traceback.print_exc()
    finally:
        signal.signal(signal.SIGINT, original_sigint_handler)
        if args_cli.record and hasattr(env.unwrapped.recorder_manager, "finalize"):
            env.unwrapped.recorder_manager.finalize()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
