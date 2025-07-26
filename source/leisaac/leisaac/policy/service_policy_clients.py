import torch
import numpy as np

from .base import ServicePolicy

from leisaac.utils.robot_utils import convert_leisaac_action_to_lerobot, convert_lerobot_action_to_leisaac


class Gr00tServicePolicyClient(ServicePolicy):
    """
    Service policy client for GR00T N1.5: https://github.com/NVIDIA/Isaac-GR00T
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 5000,
        camera_keys: list[str] = ['front', 'wrist'],
        modality_keys: list[str] = ["single_arm", "gripper"],
    ):
        """
        Args:
            host: Host of the policy server.
            port: Port of the policy server.
            camera_keys: Keys of the cameras.
            timeout_ms: Timeout of the policy server.
            modality_keys: Keys of the modality.
        """
        super().__init__(host=host, port=port, timeout_ms=timeout_ms, ping_endpoint="ping", kill_endpoint="kill")
        self.camera_keys = camera_keys
        self.modality_keys = modality_keys

    def get_action(self, observation_dict: dict) -> torch.Tensor:
        obs_dict = {f"video.{key}": observation_dict[key].cpu().numpy().astype(np.uint8) for key in self.camera_keys}

        if "single_arm" in self.modality_keys:
            joint_pos = convert_leisaac_action_to_lerobot(observation_dict["joint_pos"])
            obs_dict["state.single_arm"] = joint_pos[:, 0:5]
            obs_dict["state.gripper"] = joint_pos[:, 5:6]
        # TODO: add bi-arm support

        obs_dict["annotation.human.task_description"] = observation_dict["task_description"]

        """
            Example of obs_dict for single arm task:
            obs_dict = {
                "video.front": np.zeros((1, 480, 640, 3), dtype=np.uint8),
                "video.wrist": np.zeros((1, 480, 640, 3), dtype=np.uint8),
                "state.single_arm": np.zeros((1, 5)),
                "state.gripper": np.zeros((1, 1)),
                "annotation.human.action.task_description": [observation_dict["task_description"]],
            }
        """

        # get the action chunk via the policy server
        action_chunk = self.call_endpoint("get_action", obs_dict)

        """
            Example of action_chunk for single arm task:
            action_chunk = {
                "action.single_arm": np.zeros((1, 5)),
                "action.gripper": np.zeros((1, 1)),
            }
        """
        concat_action = np.concatenate(
            [action_chunk["action.single_arm"], action_chunk["action.gripper"][:, None]],
            axis=1,
        )
        concat_action = convert_lerobot_action_to_leisaac(concat_action)

        return torch.from_numpy(concat_action[:, None, :])
