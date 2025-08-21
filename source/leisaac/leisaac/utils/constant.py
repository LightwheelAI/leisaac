import os

try:
    from git import Repo

    repo = Repo(os.getcwd(), search_parent_directories=True)
    git_root = repo.git.rev_parse("--show-toplevel")
except Exception:
    from pathlib import Path

    git_root = Path(os.path.abspath(__file__)).parent.parent.parent.parent.parent

ASSETS_ROOT = os.path.join(git_root, 'assets')

SINGLE_ARM_JOINT_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
BI_ARM_JOINT_NAMES = ['left_shoulder_pan', 'left_shoulder_lift', 'left_elbow_flex', 'left_wrist_flex', 'left_wrist_roll', 'left_gripper',
                      'right_shoulder_pan', 'right_shoulder_lift', 'right_elbow_flex', 'right_wrist_flex', 'right_wrist_roll', 'right_gripper']
