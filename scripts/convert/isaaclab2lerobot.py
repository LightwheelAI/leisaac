import os

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

"""
NOTE: Please use the environment of lerobot.

Because lerobot is rapidly developing, we don't guarantee the compatibility for the latest version of lerobot.
Currently, the commit we used is https://github.com/huggingface/lerobot/tree/v0.3.3
"""

# Feature definition for single-arm so101_follower
SINGLE_ARM_FEATURES = {
    "action": {
        "dtype": "float32",
        "shape": (6,),
        "names": [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (6,),
        "names": [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ],
    },
    "observation.images.front": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30.0,
            "video.channels": 3,
            "has_audio": False,
        },
    },
    "observation.images.wrist": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30.0,
            "video.channels": 3,
            "has_audio": False,
        },
    },
}

# Feature definition for bi-arm so101_follower
BI_ARM_FEATURES = {
    "action": {
        "dtype": "float32",
        "shape": (12,),
        "names": [
            "left_shoulder_pan.pos",
            "left_shoulder_lift.pos",
            "left_elbow_flex.pos",
            "left_wrist_flex.pos",
            "left_wrist_roll.pos",
            "left_gripper.pos",
            "right_shoulder_pan.pos",
            "right_shoulder_lift.pos",
            "right_elbow_flex.pos",
            "right_wrist_flex.pos",
            "right_wrist_roll.pos",
            "right_gripper.pos",
        ],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (12,),
        "names": [
            "left_shoulder_pan.pos",
            "left_shoulder_lift.pos",
            "left_elbow_flex.pos",
            "left_wrist_flex.pos",
            "left_wrist_roll.pos",
            "left_gripper.pos",
            "right_shoulder_pan.pos",
            "right_shoulder_lift.pos",
            "right_elbow_flex.pos",
            "right_wrist_flex.pos",
            "right_wrist_roll.pos",
            "right_gripper.pos",
        ],
    },
    "observation.images.left_wrist": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30.0,
            "video.channels": 3,
            "has_audio": False,
        },
    },
    "observation.images.top": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30.0,
            "video.channels": 3,
            "has_audio": False,
        },
    },
    "observation.images.right_wrist": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30.0,
            "video.channels": 3,
            "has_audio": False,
        },
    },
}

# preprocess actions and joint pos
ISAACLAB_JOINT_POS_LIMIT_RANGE = [
    (-110.0, 110.0),
    (-100.0, 100.0),
    (-100.0, 90.0),
    (-95.0, 95.0),
    (-160.0, 160.0),
    (-10, 100.0),
]
LEROBOT_JOINT_POS_LIMIT_RANGE = [
    (-100, 100),
    (-100, 100),
    (-100, 100),
    (-100, 100),
    (-100, 100),
    (0, 100),
]


def preprocess_joint_pos(joint_pos: np.ndarray) -> np.ndarray:
    joint_pos = joint_pos / np.pi * 180
    for i in range(6):
        isaaclab_min, isaaclab_max = ISAACLAB_JOINT_POS_LIMIT_RANGE[i]
        lerobot_min, lerobot_max = LEROBOT_JOINT_POS_LIMIT_RANGE[i]
        isaac_range = isaaclab_max - isaaclab_min
        lerobot_range = lerobot_max - lerobot_min
        joint_pos[:, i] = (joint_pos[:, i] - isaaclab_min) / isaac_range * lerobot_range + lerobot_min
    return joint_pos


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
    actions = preprocess_joint_pos(actions)
    joint_pos = preprocess_joint_pos(joint_pos)

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
    actions = preprocess_joint_pos(actions)
    left_joint_pos = preprocess_joint_pos(left_joint_pos)
    right_joint_pos = preprocess_joint_pos(right_joint_pos)

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


def convert_isaaclab_to_lerobot():
    """NOTE: Modify the following parameters to fit your own dataset"""
    repo_id = "EverNorif/so101_test_orange_pick"
    robot_type = "so101_follower"  # so101_follower, bi_so101_follower
    fps = 30
    hdf5_root = "./datasets"
    hdf5_files = [os.path.join(hdf5_root, "dataset.hdf5")]
    task = "Grab orange and place into plate"
    push_to_hub = False

    """parameters check"""
    assert robot_type in [
        "so101_follower",
        "bi_so101_follower",
    ], "robot_type must be so101_follower or bi_so101_follower"

    """convert to LeRobotDataset"""
    now_episode_index = 0
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=SINGLE_ARM_FEATURES if robot_type == "so101_follower" else BI_ARM_FEATURES,
    )

    for hdf5_id, hdf5_file in enumerate(hdf5_files):
        print(f"[{hdf5_id+1}/{len(hdf5_files)}] Processing hdf5 file: {hdf5_file}")
        with h5py.File(hdf5_file, "r") as f:
            demo_names = list(f["data"].keys())
            print(f"Found {len(demo_names)} demos: {demo_names}")

            for demo_name in tqdm(demo_names, desc="Processing each demo"):
                demo_group = f["data"][demo_name]
                if "success" in demo_group.attrs and not demo_group.attrs["success"]:
                    print(f"Demo {demo_name} is not successful, skip it")
                    continue

                if robot_type == "so101_follower":
                    valid = process_single_arm_data(dataset, task, demo_group, demo_name)
                elif robot_type == "bi_so101_follower":
                    valid = process_bi_arm_data(dataset, task, demo_group, demo_name)

                if valid:
                    now_episode_index += 1
                    dataset.save_episode()
                    print(f"Saving episode {now_episode_index} successfully")

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    convert_isaaclab_to_lerobot()
