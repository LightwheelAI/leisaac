import numpy as np

from ..device_base import Device
from .gamepad_utils import GamepadController


class SO101Gamepad(Device):
    """Gamepad controller for sending SE(3) commands as delta poses for so101 single arm.

    Key bindings:
    ====================== ==========================================
    Description              Key
    ====================== ==========================================
    Forward / Backward       Left stick Y (forward / backward)
    Left / Right             Left stick X (left / right)
    Up / Down                Right stick Y (forward / backward)
    Rotate (Yaw) Left/Right  Right stick X (left / right)
    Rotate (Pitch) Up/Down   LB / LT
    Gripper Open / Close     RT / RB
    ====================== ==========================================
    """

    def __init__(self, env, sensitivity: float = 1.0):
        super().__init__(env, "gamepad")

        # store inputs
        self.pos_sensitivity = 0.01 * sensitivity
        self.joint_sensitivity = 0.15 * sensitivity
        self.rot_sensitivity = 0.15 * sensitivity

        # initialize gamepad controller
        self._gamepad = GamepadController()
        self._gamepad.start()
        if "xbox" not in self._gamepad.name:
            raise ValueError("Only Xbox gamepads are supported. Please connect an Xbox gamepad and try again.")
        self._create_key_mapping()

        # command buffers (dx, dy, dz, droll, dpitch, dyaw, d_shoulder_pan, d_gripper)
        self._delta_action = np.zeros(8)

        # initialize the target frame
        self.asset_name = "robot"
        self.robot_asset = self.env.scene[self.asset_name]

        self.target_frame = "gripper"
        body_idxs, _ = self.robot_asset.find_bodies(self.target_frame)
        self.target_frame_idx = body_idxs[0]

    def __del__(self):
        """Release the gamepad interface."""
        super().__del__()
        self._gamepad.stop()

    def __str__(self) -> str:
        """Returns: A string containing the information of gamepad controller."""
        msg = "Gamepad Controller for SO101 Single Arm (SE(3) Control).\n"
        msg += f"\tGamepad name: {self._gamepad.name}\n"
        msg += f"\ttarget frame: {self.target_frame}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tForward / Backward:       Left stick Y (forward / backward)\n"
        msg += "\tLeft / Right:             Left stick X (left / right)\n"
        msg += "\tUp / Down:                Right stick Y (forward / backward)\n"
        msg += "\tRotate (Yaw) Left/Right:  Right stick X (left / right)\n"
        msg += "\tRotate (Pitch) Up/Down:   LB / LT\n"
        msg += "\tGripper Open / Close:     RT / RB\n"
        msg += "\t----------------------------------------------\n"
        return msg

    def get_device_state(self):
        return self._convert_delta_from_frame(self._delta_action)

    def reset(self):
        self._delta_action[:] = 0.0

    def advance(self):
        self._delta_action[:] = 0.0
        self._update_action()
        return super().advance()

    def _create_key_mapping(self):
        self._ACTION_DELTA_MAPPING = {
            "forward": np.asarray([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.pos_sensitivity,
            "backward": np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.pos_sensitivity,
            "left": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0]) * self.joint_sensitivity,
            "right": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]) * self.joint_sensitivity,
            "up": np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.pos_sensitivity,
            "down": np.asarray([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.pos_sensitivity,
            "rotate_up": np.asarray([0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0]) * self.rot_sensitivity,
            "rotate_down": np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) * self.rot_sensitivity,
            "rotate_left": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]) * self.rot_sensitivity,
            "rotate_right": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0]) * self.rot_sensitivity,
            "gripper_open": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) * self.joint_sensitivity,
            "gripper_close": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]) * self.joint_sensitivity,
        }
        self._INPUT_KEY_MAPPING_LIST = [  # (action_name, controller_name, reverse[optional])
            ("forward", "L_Y", True),
            ("backward", "L_Y"),
            ("left", "L_X", True),
            ("right", "L_X"),
            ("up", "R_Y", True),
            ("down", "R_Y"),
            ("rotate_up", "LB"),
            ("rotate_down", "LT"),
            ("rotate_left", "R_X", True),
            ("rotate_right", "R_X"),
            ("gripper_open", "RT"),
            ("gripper_close", "RB"),
        ]

    def _update_action(self):
        """Update the delta action based on the gamepad state."""
        self._gamepad.update()
        for input_key_mapping in self._INPUT_KEY_MAPPING_LIST:
            action_name, controller_name = input_key_mapping[0], input_key_mapping[1]
            reverse = input_key_mapping[2] if len(input_key_mapping) > 2 else False
            controller_state = self._gamepad.get_state()
            is_activate, is_positive = self._gamepad.lookup_controller_state(controller_state, controller_name, reverse)
            if is_activate and is_positive:
                self._delta_action += self._ACTION_DELTA_MAPPING[action_name]
