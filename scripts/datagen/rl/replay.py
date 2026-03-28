"""Script to replay recorded RL episodes from HDF5 dataset."""

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay recorded RL episodes.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/rl_eval.hdf5", help="File path to load recorded episodes."
)
parser.add_argument(
    "--replay_mode",
    type=str,
    default="action",
    choices=["action", "state"],
    help="Replay mode: action replays actions, state replays joint states.",
)
parser.add_argument(
    "--select_episodes",
    type=int,
    nargs="+",
    default=[],
    help="List of episode indices to replay. Empty = replay all.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import contextlib
import os

import gymnasium as gym
import leisaac.tasks  # noqa: F401
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab_tasks.utils import parse_env_cfg


def get_next_action(episode_data: EpisodeData, return_state: bool = False) -> torch.Tensor | None:
    if return_state:
        next_state = episode_data.get_next_state()
        if next_state is None:
            return None
        return next_state["articulation"]["robot"]["joint_position"]
    else:
        return episode_data.get_next_action()


def main():
    if not os.path.exists(args_cli.dataset_file):
        raise FileNotFoundError(f"Dataset file not found: {args_cli.dataset_file}")

    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.dataset_file)
    episode_count = dataset_file_handler.get_num_episodes()

    if episode_count == 0:
        print("No episodes found in the dataset.")
        return

    episode_indices_to_replay = args_cli.select_episodes or list(range(episode_count))
    num_envs = args_cli.num_envs

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=num_envs)
    env_cfg.recorders = {}
    env_cfg.terminations = {}

    env: ManagerBasedRLEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    idle_action = torch.zeros(env.action_space.shape)

    if hasattr(env, "initialize"):
        env.initialize()
    env.reset()

    episode_names = list(dataset_file_handler.get_episode_names())
    replayed_episode_count = 0

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():
            env_episode_data_map = {i: EpisodeData() for i in range(num_envs)}
            has_next_action = True
            while has_next_action:
                actions = idle_action.clone()
                has_next_action = False
                for env_id in range(num_envs):
                    env_next_action = get_next_action(
                        env_episode_data_map[env_id],
                        return_state=args_cli.replay_mode == "state",
                    )
                    if env_next_action is None:
                        next_episode_index = None
                        while episode_indices_to_replay:
                            next_episode_index = episode_indices_to_replay.pop(0)
                            if next_episode_index < episode_count:
                                break
                            next_episode_index = None

                        if next_episode_index is not None:
                            replayed_episode_count += 1
                            print(f"{replayed_episode_count:4}: Loading #{next_episode_index} episode to env_{env_id}")
                            episode_data = dataset_file_handler.load_episode(
                                episode_names[next_episode_index], env.device
                            )
                            env_episode_data_map[env_id] = episode_data
                            initial_state = episode_data.get_initial_state()
                            env.reset_to(
                                initial_state,
                                torch.tensor([env_id], device=env.device),
                                seed=int(episode_data.seed) if episode_data.seed is not None else None,
                                is_relative=True,
                            )
                            env_next_action = get_next_action(
                                env_episode_data_map[env_id],
                                return_state=args_cli.replay_mode == "state",
                            )
                            has_next_action = True
                        else:
                            continue
                    else:
                        has_next_action = True
                    actions[env_id] = env_next_action

                env.step(actions)
            break

    print(f"Finished replaying {replayed_episode_count} episode{'s' if replayed_episode_count != 1 else ''}.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
