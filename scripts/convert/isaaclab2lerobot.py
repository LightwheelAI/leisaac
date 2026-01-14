"""This script converts IsaacLab HDF5 datasets into LeRobot Dataset v2 format.

Since LeRobot is evolving rapidly, compatibility with the latest LeRobot versions is not guaranteed.
Please install the following specific versions of the dependencies:

pip install lerobot==0.3.3
pip install numpy==1.26.0

"""

import argparse
import os

import h5py
import numpy as np
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
        "Specify task type. If your dataset is recorded with keyboard, you should set it to 'keyboard', otherwise not"
        " to set it and keep default value None."
    ),
)
parser.add_argument(
    "--repo-id",
    "-r",
    type=str,
    default="EverNorif/so101_test_orange_pick",
    help="Repository ID",
)
parser.add_argument(
    "--fps",
    "-f",
    type=int,
    default=30,
    help="Frames per second",
)
parser.add_argument(
    "--hdf5-root",
    "-d",
    type=str,
    default="./datasets",
    help="HDF5 root directory",
)
parser.add_argument(
    "--hdf5-files",
    type=str,
    default=None,
    help="HDF5 files (comma-separated). If not provided, uses dataset.hdf5 in hdf5_root",
)
parser.add_argument(
    "--task_description",
    type=str,
    default="Grab orange and place into plate",
    help="Task description",
)
parser.add_argument(
    "--push-to-hub",
    action="store_true",
    help="Push to hub",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app


import gymnasium as gym

# import torch
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.enhance.datasets.lerobot_dataset_handler import LeRobotDatasetCfg
from leisaac.utils.env_utils import get_task_type
from leisaac.utils.robot_utils import build_feature_from_env


def process_single_arm_data(dataset: LeRobotDataset, task: str, demo_group: h5py.Group, demo_name: str) -> bool:
    try:
        actions = np.array(demo_group["actions"])
        joint_pos = np.array(demo_group["obs/joint_pos"])
        front_images = np.array(demo_group["obs/front"])
        wrist_images = np.array(demo_group["obs/wrist"])
    except KeyError:
        print(f"Demo {demo_name} is not valid, skip it")
        return False

    if actions.shape[0] < 10:
        print(f"Demo {demo_name} has less than 10 frames, skip it")
        return False

    # preprocess actions and joint pos
    # actions = preprocess_joint_pos(actions)
    # joint_pos = preprocess_joint_pos(joint_pos)

    assert actions.shape[0] == joint_pos.shape[0] == front_images.shape[0] == wrist_images.shape[0]
    total_state_frames = actions.shape[0]
    # skip the first 5 frames
    for frame_index in tqdm(range(5, total_state_frames), desc="Processing each frame"):
        frame = {
            "action": actions[frame_index],
            "observation.state": joint_pos[frame_index],
            "observation.images.front": front_images[frame_index],
            "observation.images.wrist": wrist_images[frame_index],
        }
        dataset.add_frame(frame=frame, task=task)

    return True


def process_bi_arm_data(dataset: LeRobotDataset, task: str, demo_group: h5py.Group, demo_name: str) -> bool:
    try:
        actions = np.array(demo_group["actions"])
        left_joint_pos = np.array(demo_group["obs/left_joint_pos"])
        right_joint_pos = np.array(demo_group["obs/right_joint_pos"])
        left_images = np.array(demo_group["obs/left_wrist"])
        right_images = np.array(demo_group["obs/right_wrist"])
        top_images = np.array(demo_group["obs/top"])
    except KeyError:
        print(f"Demo {demo_name} is not valid, skip it")
        return False

    if actions.shape[0] < 10:
        print(f"Demo {demo_name} has less than 10 frames, skip it")
        return False

    # preprocess actions and joint pos
    # actions = preprocess_joint_pos(actions)
    # left_joint_pos = preprocess_joint_pos(left_joint_pos)
    # right_joint_pos = preprocess_joint_pos(right_joint_pos)

    assert (
        actions.shape[0]
        == left_joint_pos.shape[0]
        == right_joint_pos.shape[0]
        == left_images.shape[0]
        == right_images.shape[0]
        == top_images.shape[0]
    )
    total_state_frames = actions.shape[0]
    # skip the first 5 frames
    for frame_index in tqdm(range(5, total_state_frames), desc="Processing each frame"):
        frame = {
            "action": actions[frame_index],
            "observation.state": np.concatenate([left_joint_pos[frame_index], right_joint_pos[frame_index]]),
            "observation.images.left_wrist": left_images[frame_index],
            "observation.images.top": top_images[frame_index],
            "observation.images.right_wrist": right_images[frame_index],
        }
        dataset.add_frame(frame=frame, task=task)

    return True


def save_episode(dataset: LeRobotDataset, episode: EpisodeData):
    # for frame_index in range(len(episode)):
    # frame = episode[frame_index]
    # dataset.add_frame(frame=frame, task=task)
    return True


def convert_isaaclab_to_lerobot():
    """automatically build features and dataset"""
    env_cfg = parse_env_cfg(args_cli.task_name, device=args_cli.device, num_envs=args_cli.num_envs)
    task_type = get_task_type(args_cli.task_name, args_cli.task_type)
    env_cfg.use_teleop_device(task_type)

    env: ManagerBasedRLEnv | DirectRLEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    dataset_cfg = LeRobotDatasetCfg(
        repo_id=args_cli.repo_id,
        fps=args_cli.fps,
        robot_type=env_cfg.robot_name,
        features=build_feature_from_env(env, env_cfg),
    )
    dataset_cfg.features = build_feature_from_env(env, env_cfg)

    dataset = LeRobotDataset.create(
        repo_id=dataset_cfg.repo_id,
        fps=dataset_cfg.fps,
        robot_type=dataset_cfg.robot_type,
        features=dataset_cfg.features,
    )

    """load datasets"""
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
            episode = dataset_file_handler.load_episode(episode_name)
            # if not success:
            # pass
            valid = save_episode(dataset, episode)
            if valid:
                now_episode_index += 1
                dataset.save_episode()
                print(f"Saving episode {now_episode_index} successfully")
            print(episode)
            save_episode(dataset, episode)

        dataset_file_handler.close()

    if args_cli.push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    convert_isaaclab_to_lerobot()
