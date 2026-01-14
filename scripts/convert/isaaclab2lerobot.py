"""This script converts IsaacLab HDF5 datasets into LeRobot Dataset v2 format.

Since LeRobot is evolving rapidly, compatibility with the latest LeRobot versions is not guaranteed.
Please install the following specific versions of the dependencies:

pip install lerobot==0.3.3
pip install numpy==1.26.0

"""

import argparse
import os

from isaaclab.app import AppLauncher
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

# add argparse arguments
parser = argparse.ArgumentParser(description="Convert IsaacLab dataset to LeRobot Dataset v2.")
parser.add_argument("--task_name", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--task_type",
    type=str,
    default=None,
    help=(
        "Specify task type. If your dataset is recorded with keyboard/gamepad, you should set it to"
        " 'keyboard'/'gamepad', otherwise not to set it and keep default value None."
    ),
)
parser.add_argument(
    "--repo_id",
    type=str,
    default="EverNorif/so101_test_orange_pick",
    help="Repository ID",
)
parser.add_argument(
    "--fps",
    type=int,
    default=30,
    help="Frames per second",
)
parser.add_argument(
    "--hdf5_root",
    type=str,
    default="./datasets",
    help="HDF5 root directory",
)
parser.add_argument(
    "--hdf5_files",
    type=str,
    default=None,
    help="HDF5 files (comma-separated). If not provided, uses dataset.hdf5 in hdf5_root",
)
parser.add_argument(
    "--task_description",
    type=str,
    default=None,
    help="Task description. If not provided, will use the description defined in the task.",
)
parser.add_argument(
    "--push_to_hub",
    action="store_true",
    help="Push to hub",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# default arguments
default_args = {
    "headless": True,
    "enable_cameras": True,
}
app_launcher_args = vars(args_cli)
app_launcher_args.update(default_args)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app


import gymnasium as gym
import torch
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.enhance.datasets.lerobot_dataset_handler import LeRobotDatasetCfg
from leisaac.utils.env_utils import get_task_type
from leisaac.utils.robot_utils import build_feature_from_env


def split_episode(episode: EpisodeData, num_frames: int) -> list[EpisodeData]:
    def slice_at_index(data, idx: int):
        """Take the idx-th frame from the nested data structure."""
        if isinstance(data, dict):
            return {k: slice_at_index(v, idx) for k, v in data.items()}
        if isinstance(data, torch.Tensor):
            safe_idx = idx if idx < data.shape[0] else 0
            return [data[safe_idx]]
        return data

    full_data = episode.data
    sub_episodes: list[EpisodeData] = []
    for idx in range(num_frames):
        sub_episode = EpisodeData()
        sub_episode.data = slice_at_index(full_data, idx)
        sub_episodes.append(sub_episode)

    return sub_episodes


def add_episode(
    dataset: LeRobotDataset,
    episode: EpisodeData,
    env: ManagerBasedRLEnv | DirectRLEnv,
    dataset_cfg: LeRobotDatasetCfg,
    task: str,
):
    all_data = episode.data
    num_frames = all_data["actions"].shape[0]
    if num_frames < 10:
        print(f"Episode {episode.env_id} has less than 10 frames, skip it")
        return False

    episode_list = split_episode(episode, num_frames)
    # skip the first 5 frames
    for frame_index in tqdm(range(5, num_frames), desc="Processing each frame"):
        frame = env.cfg.build_lerobot_frame(episode_list[frame_index], dataset_cfg)
        predefined_task = frame.pop("task")
        dataset.add_frame(frame=frame, task=predefined_task if task is None else task)
    return True


def convert_isaaclab_to_lerobot():
    """automatically build features and dataset"""
    env_cfg = parse_env_cfg(args_cli.task_name, device=args_cli.device, num_envs=1)
    task_type = get_task_type(args_cli.task_name, args_cli.task_type)
    env_cfg.use_teleop_device(task_type)

    env: ManagerBasedRLEnv | DirectRLEnv = gym.make(args_cli.task_name, cfg=env_cfg).unwrapped

    dataset_cfg = LeRobotDatasetCfg(
        repo_id=args_cli.repo_id,
        fps=args_cli.fps,
        robot_type=env_cfg.robot_name,
    )
    dataset_cfg.features = build_feature_from_env(env, dataset_cfg)

    dataset = LeRobotDataset.create(
        repo_id=dataset_cfg.repo_id,
        fps=dataset_cfg.fps,
        robot_type=dataset_cfg.robot_type,
        features=dataset_cfg.features,
    )

    if args_cli.hdf5_files is None:
        hdf5_files_list = [os.path.join(args_cli.hdf5_root, "dataset.hdf5")]
    else:
        hdf5_files_list = [
            os.path.join(args_cli.hdf5_root, f.strip()) if not os.path.isabs(f.strip()) else f.strip()
            for f in args_cli.hdf5_files.split(",")
        ]

    now_episode_index = 0
    for hdf5_id, hdf5_file in enumerate(hdf5_files_list):
        print(f"[{hdf5_id+1}/{len(hdf5_files_list)}] Processing hdf5 file: {hdf5_file}")

        dataset_file_handler = HDF5DatasetFileHandler()
        dataset_file_handler.open(hdf5_file)

        episode_names = dataset_file_handler.get_episode_names()
        print(f"Found {len(episode_names)} episodes: {episode_names}")
        for episode_name in tqdm(episode_names, desc="Processing each episode"):
            episode = dataset_file_handler.load_episode(episode_name, device=args_cli.device)
            if not episode.success:
                print(f"Episode {episode_name} is not successful, skip it")
                continue
            valid = add_episode(dataset, episode, env, dataset_cfg, args_cli.task_description)
            if valid:
                now_episode_index += 1
                dataset.save_episode()
                print(f"Saving episode {now_episode_index} successfully")
            else:
                dataset.clear_episode_buffer()

        dataset_file_handler.close()

    if args_cli.push_to_hub:
        dataset.push_to_hub()

    print("Finished converting IsaacLab dataset to LeRobot dataset")
    env.close()


if __name__ == "__main__":
    convert_isaaclab_to_lerobot()
