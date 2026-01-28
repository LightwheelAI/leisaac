import carb
import numpy as np

from ..device_base import Device


class SO101Statemachine(Device):
    """
    A statemachine controller to generate poses for so101 single arm.
    """

    def __init__(self, env, sensitivity: float = 1.0):
        super().__init__(env, "so101_state_machine")

        # store inputs
        self.pos_sensitivity = 0.01 * sensitivity
        self.joint_sensitivity = 0.15 * sensitivity
        self.rot_sensitivity = 0.15 * sensitivity

        # command buffers (dx, dy, dz, droll, dpitch, dyaw, d_shoulder_pan, d_gripper)
        self._delta_action = np.zeros(8)

        # initialize the target frame
        self.asset_name = "robot"
        self.robot_asset = self.env.scene[self.asset_name]

        self.target_frame = "gripper"
        body_idxs, _ = self.robot_asset.find_bodies(self.target_frame)
        self.target_frame_idx = body_idxs[0]


    def get_device_state(self):
        return self._convert_delta_from_frame(self._delta_action)

    def reset(self):
        self._delta_action[:] = 0.0