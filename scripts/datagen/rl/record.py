"""Evaluate a trained RSL-RL policy on a leisaac task, with optional HDF5 episode recording."""

import argparse
import importlib
import os
import signal

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate an RSL-RL policy on a leisaac task.")
parser.add_argument("--task", type=str, required=True, help="Gym task ID (e.g. LeIsaac-SO101-LiftCube-RL-v0).")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--num_episodes", type=int, default=0, help="Total episodes to run (0 = infinite).")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--record", action="store_true", help="Save all episodes to HDF5 with success/failure tags.")
parser.add_argument(
    "--resume", action="store_true", help="Append to an existing dataset file instead of creating a new one."
)
parser.add_argument("--dataset_file", type=str, default="./datasets/rl_eval.hdf5", help="Output HDF5 file path.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import leisaac.tasks  # noqa: F401
import torch
from isaaclab.managers import DatasetExportMode
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from leisaac.enhance.managers import EnhanceDatasetExportMode, StreamingRecorderManager
from rsl_rl.runners import OnPolicyRunner


def _load_from_entry_point(entry_point: str):
    module_path, attr = entry_point.rsplit(":", 1)
    return getattr(importlib.import_module(module_path), attr)


def main():
    kwargs = gym.registry[args_cli.task].kwargs
    env_cfg = _load_from_entry_point(kwargs["env_cfg_entry_point"])()
    train_cfg = dict(_load_from_entry_point(kwargs["rsl_rl_cfg_entry_point"]))

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.record:
        output_dir = os.path.dirname(os.path.abspath(args_cli.dataset_file))
        output_filename = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
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
        env_cfg.recorders.dataset_filename = output_filename
    else:
        env_cfg.recorders = None

    env = gym.make(args_cli.task, cfg=env_cfg)

    if args_cli.record:
        unwrapped = env.unwrapped
        del unwrapped.recorder_manager
        unwrapped.recorder_manager = StreamingRecorderManager(env_cfg.recorders, unwrapped)
        unwrapped.recorder_manager.flush_steps = 100
        unwrapped.recorder_manager.compression = "lzf"

    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=env_cfg.sim.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env_cfg.sim.device)

    episode_count = 0
    success_count = 0
    interrupted = False

    def signal_handler(signum, frame):
        nonlocal interrupted
        interrupted = True
        print("\n[INFO] KeyboardInterrupt detected. Cleaning up resources...")

    original_sigint_handler = signal.signal(signal.SIGINT, signal_handler)

    try:
        step = 0
        obs = env.get_observations()
        while (
            simulation_app.is_running()
            and not interrupted
            and (args_cli.num_episodes <= 0 or episode_count < args_cli.num_episodes)
        ):
            with torch.no_grad():
                actions = policy(obs)

            step += 1
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
                print(f"Total success rate: {success_count / episode_count:.1%} ({success_count}/{episode_count})")
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
