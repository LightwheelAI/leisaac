"""Script to run eef action process for mimic recorded demos."""

"""Launch Isaac Sim Simulator first."""
import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="eef action process for mimic recorded demos.")
parser.add_argument(
    "--input_file",
    type=str,
    default="./datasets/mimic-lift-cube-example.hdf5",
    help="File path to load mimic recorded demos.",
)
parser.add_argument(
    "--output_file",
    type=str,
    default="./datasets/processed_mimic-lift-cube-example.hdf5",
    help="File path to save processed mimic recorded demos.",
)
parser.add_argument("--to_ik", action="store_true", help="Whether to convert the action to ik action.")
parser.add_argument("--to_joint", action="store_true", help="Whether to convert the action to joint action.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import os
from copy import deepcopy

import torch
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from tqdm import tqdm


def joint_action_to_ik(episode_data: EpisodeData) -> EpisodeData:
    """Convert the action to ik action."""
    eef_state = episode_data.data["obs"]["ee_frame_state"]

    action = episode_data.data["actions"]
    gripper_action = action[:, -1:]
    new_actions = torch.cat([eef_state, gripper_action], dim=1)
    episode_data.data["actions"] = new_actions

    return episode_data


def ik_action_to_joint(episode_data: EpisodeData) -> EpisodeData:
    """Convert the action to joint action."""
    joint_pos = episode_data.data["obs"]["joint_pos_target"]

    new_actions = joint_pos
    episode_data.data["actions"] = new_actions

    return episode_data


def main():
    """Process the eef action of the mimic annotated recorded demos."""
    # check arguments
    if args_cli.to_ik and args_cli.to_joint:
        raise ValueError("Cannot convert to both ik and joint action at the same time.")
    if not args_cli.to_ik and not args_cli.to_joint:
        raise ValueError("Must convert to either ik or joint action.")

    # Load dataset
    if not os.path.exists(args_cli.input_file):
        raise FileNotFoundError(f"The dataset file {args_cli.input_file} does not exist.")
    input_dataset_handler = HDF5DatasetFileHandler()
    input_dataset_handler.open(args_cli.input_file)

    output_dataset_handler = HDF5DatasetFileHandler()
    output_dataset_handler.create(args_cli.output_file)

    episode_names = list(input_dataset_handler.get_episode_names())
    for episode_name in tqdm(episode_names):
        episode_data = input_dataset_handler.load_episode(episode_name, device=args_cli.device)
        if episode_data.success is not None and not episode_data.success:
            continue
        process_episode_data = deepcopy(episode_data)
        if args_cli.to_ik:
            process_episode_data = joint_action_to_ik(process_episode_data)
        elif args_cli.to_joint:
            process_episode_data = ik_action_to_joint(process_episode_data)
        output_dataset_handler.write_episode(process_episode_data)

    input_dataset_handler.close()
    output_dataset_handler.flush()
    output_dataset_handler.close()


if __name__ == "__main__":
    # run the main function
    main()
