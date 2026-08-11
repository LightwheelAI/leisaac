"""SO-101 <-> Pi0.5 observation and action adapters.

These adapters translate between the *canonical SO-101 simulation representation*
(IsaacLab articulation joint positions/velocities in radians, gripper in
radians, camera RGB uint8 HWC) and the *Pi0.5 (OpenPI) representation*
(image dict "images/<cam>" uint8 224x224, "state" float64[6], "prompt" str,
and an "actions" array of shape (action_chunk, 6)).

Grounded in (verified, not assumed):
  - leisaac.assets.robots.lerobot.SO101_FOLLOWER_CFG joint names/order:
        shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
  - leisaac.utils.robot_utils.convert_leisaac_action_to_lerobot /
    convert_lerobot_action_to_leisaac: radians<->degrees plus USD-joint-limit
    <-> motor-limit remapping per joint. Pi0.5 is trained in the LeRobot
    normalized joint space, so the gripper action is LeRobot-normalized [0,100].
  - leisaac.policy.service_policy_clients.OpenPIServicePolicyClient.get_action:
        sends images resized to 224x224 with pad, state float64[6], prompt str;
        receives {"actions": (chunk, 6)} and runs convert_lerobot_action_to_leisaac.

Design rule (from the project spec): do NOT leak Pi0.5-specific formats into the
simulation. The simulation only ever sees the canonical SO-101 action tensor.
The adapters are the single translation boundary.
"""

from __future__ import annotations

import numpy as np
import torch

# Canonical SO-101 joint order (radians for the 5 arm joints, radians for gripper).
SO101_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# Image size Pi0.5 (OpenPI) expects.
PI05_IMAGE_SIZE = 224


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert float [0,1] images to uint8, mirroring OpenPI image_tools."""
    if np.issubdtype(img.dtype, np.floating):
        img = (255.0 * img).astype(np.uint8)
    return img.astype(np.uint8)


def resize_with_pad(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize HWC image to (height, width) preserving aspect ratio with zero pad.

    Mirrors leisaac.policy.openpi.image_tools.resize_with_pad (PIL-based).
    Kept dependency-light here so adapters are unit-testable without PIL-heavy
    imports; the production path can delegate to image_tools directly.
    """
    from PIL import Image

    cur_h, cur_w = img.shape[:2]
    if (cur_h, cur_w) == (height, width):
        return img
    ratio = max(cur_w / width, cur_h / height)
    new_w, new_h = int(cur_w / ratio), int(cur_h / ratio)
    resized = np.asarray(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))
    out = np.zeros((height, width, img.shape[2]), dtype=img.dtype)
    pad_h, pad_w = (height - new_h) // 2, (width - new_w) // 2
    out[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized
    return out


def build_pi05_observation(
    camera_images: dict[str, np.ndarray],
    joint_state: np.ndarray,
    task_description: str,
) -> dict:
    """Canonical SO-101 observation -> Pi0.5 (OpenPI) observation dict.

    Args:
        camera_images: {camera_key: uint8 or float HWC image}. Keys are the
            SO-101 camera names (e.g. "front", "wrist"). Order in the dict is
            preserved into the OpenPI "images/<key>" mapping.
        joint_state: (6,) or (1,6) array in IsaacLab radians (canonical SO-101
            joint space). Converted to LeRobot-normalized [0,100] space.
        task_description: language instruction string.

    Returns:
        OpenPI observation dict:
            {"images/<cam>": uint8 (224,224,3), "state": float64 (6,),
             "prompt": str}
    """
    from leisaac.utils import robot_utils

    obs: dict[str, object] = {}
    for cam_key, img in camera_images.items():
        img_u8 = _to_uint8(img)
        obs[f"images/{cam_key}"] = resize_with_pad(img_u8, PI05_IMAGE_SIZE, PI05_IMAGE_SIZE)

    js = np.asarray(joint_state).reshape(-1, 6)
    # IsaacLab radians -> LeRobot normalized [0,100] (what Pi0.5 expects).
    lerobot_state = robot_utils.convert_leisaac_action_to_lerobot(js)
    obs["state"] = lerobot_state.astype(np.float64)
    obs["prompt"] = task_description
    return obs


def pi05_action_to_so101(
    pi05_actions: np.ndarray,
) -> torch.Tensor:
    """Pi0.5 action chunk -> canonical SO-101 action tensor.

    Args:
        pi05_actions: array of shape (action_chunk, 6) in LeRobot-normalized
            [0,100] space (5 arm + 1 gripper).

    Returns:
        torch.Tensor shape (action_chunk, 1, 6) in IsaacLab radians
        (canonical SO-101 action), matching the shape contract from
        OpenPIServicePolicyClient.get_action.
    """
    from leisaac.utils import robot_utils

    arr = np.asarray(pi05_actions).reshape(-1, 6)
    # LeRobot normalized [0,100] -> IsaacLab radians.
    so101 = robot_utils.convert_lerobot_action_to_leisaac(arr)
    return torch.from_numpy(so101[:, None, :])


class Pi05ObservationAdapter:
    """Stateful wrapper around build_pi05_observation.

    Keeps the canonical camera key order so repeated calls produce a stable
    layout for the policy server.
    """

    def __init__(self, camera_keys: tuple[str, ...] = ("front", "wrist")):
        self.camera_keys = tuple(camera_keys)

    def __call__(
        self,
        camera_images: dict[str, np.ndarray],
        joint_state: np.ndarray,
        task_description: str,
    ) -> dict:
        # Re-order images to the canonical camera order.
        ordered = {k: camera_images[k] for k in self.camera_keys if k in camera_images}
        return build_pi05_observation(ordered, joint_state, task_description)


class Pi05ActionAdapter:
    """Stateful wrapper around pi05_action_to_so101."""

    def __call__(self, pi05_actions: np.ndarray) -> torch.Tensor:
        return pi05_action_to_so101(pi05_actions)
