import os
import h5py
import numpy as np

from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
"""NOTE: Please use the environment of lerobot."""

# Feature definition for so101_follower
FEATURES = {
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
        ]
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
        ]

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
    }
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
    joint_pos  = joint_pos / np.pi * 180
    for i in range(6):
        isaaclab_min, isaaclab_max = ISAACLAB_JOINT_POS_LIMIT_RANGE[i]
        lerobot_min, lerobot_max = LEROBOT_JOINT_POS_LIMIT_RANGE[i]
        joint_pos[:, i] = (joint_pos[:, i] - isaaclab_min) / (isaaclab_max - isaaclab_min) * (lerobot_max - lerobot_min) + lerobot_min
    return joint_pos

def convert_isaaclab_to_lerobot():
    """NOTE: Modify the following parameters to fit your own dataset"""
    repo_id = 'EverNorif/so101_test_orange_pick'
    robot_type = 'so101_follower'
    fps = 30
    hdf5_root = './datasets'
    hdf5_files = [os.path.join(hdf5_root, 'dataset.hdf5')]
    task = 'Grab orange and place into plate'
    push_to_hub = False

    now_episode_index = 0
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=FEATURES,
    )

    for hdf5_id, hdf5_file in enumerate(hdf5_files):
        print(f'[{hdf5_id+1}/{len(hdf5_files)}] Processing hdf5 file: {hdf5_file}')
        with h5py.File(hdf5_file, 'r') as f:
            demo_names = list(f['data'].keys())
            print(f'Found {len(demo_names)} demos: {demo_names}')
            
            for demo_name in tqdm(demo_names, desc='Processing each demo'):
                demo_group = f['data'][demo_name]
                if "success" in demo_group.attrs and not demo_group.attrs["success"]:
                    print(f'Demo {demo_name} is not successful, skip it')
                    continue
                
                try:
                    actions = np.array(demo_group['obs/actions'])
                    joint_pos = np.array(demo_group['obs/joint_pos'])
                    front_images = np.array(demo_group['obs/front'])
                    wrist_images = np.array(demo_group['obs/wrist'])
                except KeyError as error:
                    print(f'Demo {demo_name} is not valid, skip it')
                    continue

                # preprocess actions and joint pos
                actions = preprocess_joint_pos(actions)
                joint_pos = preprocess_joint_pos(joint_pos)

                assert actions.shape[0] == joint_pos.shape[0] == front_images.shape[0] == wrist_images.shape[0]
                total_state_frames = actions.shape[0]
                # skip the first 5 frames
                for frame_index in tqdm(range(5, total_state_frames), desc='Processing each frame'):
                    frame = {
                        "action": actions[frame_index],
                        "observation.state": joint_pos[frame_index],
                        "observation.images.front": front_images[frame_index],
                        "observation.images.wrist": wrist_images[frame_index],
                    }
                    dataset.add_frame(frame=frame, task=task)
                now_episode_index += 1
                dataset.save_episode()
                print(f'Saving episode {now_episode_index} successfully')

    if push_to_hub:
        dataset.push_to_hub()

if __name__ == '__main__':
    convert_isaaclab_to_lerobot()
    