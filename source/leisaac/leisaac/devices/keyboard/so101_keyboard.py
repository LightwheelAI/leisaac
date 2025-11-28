import carb
import numpy as np

from ..device_base import Device


class SO101Keyboard(Device):
    """A keyboard controller for sending SE(3) commands as delta poses for so101 single arm.

    Key bindings:
        ============================== ================= =================
        Description                    Key               Key
        ============================== ================= =================
        Forward / Backward              W                 S
        Left / Right                    A                 D
        Up / Down                       Q                 E
        Rotate (Yaw) Left / Right       J                 L
        Rotate (Pitch) Up / Down        K                 I
        Gripper Open / Close            U                 O
        ============================== ================= =================

    """

    def __init__(self, env, sensitivity: float = 1.0):
        super().__init__(env, "keyboard")

        # store inputs
        self.pos_sensitivity = 0.01 * sensitivity
        self.joint_sensitivity = 0.15 * sensitivity
        self.rot_sensitivity = 0.15 * sensitivity

        # bindings for keyboard to command
        self._create_key_bindings()

        # command buffers (dx, dy, dz, droll, dpitch, dyaw, d_shoulder_pan, d_gripper)
        self._delta_action = np.zeros(8)

        # initialize the target frame
        self.asset_name = "robot"
        self.robot_asset = self.env.scene[self.asset_name]

        self.target_frame = "gripper"
        body_idxs, _ = self.robot_asset.find_bodies(self.target_frame)
        self.target_frame_idx = body_idxs[0]

    def __str__(self) -> str:
        """Returns: A string containing the information of keyboard controller."""
        msg = "Keyboard Controller for SO101 Single Arm (SE(3) Control).\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += f"\ttarget frame: {self.target_frame}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tForward / Backward:                W / S\n"
        msg += "\tLeft / Right:                      A / D\n"
        msg += "\tUp / Down:                         Q / E\n"
        msg += "\tRotate (Yaw) Left / Right:         J / L\n"
        msg += "\tRotate (Pitch) Up / Down:          K / I\n"
        msg += "\tGripper Open / Close:              U / O\n"
        msg += "\t----------------------------------------------\n"
        return msg

    def get_device_state(self):
        return self._convert_delta_from_frame(self._delta_action)

    def reset(self):
        self._delta_action[:] = 0.0

    def _on_keyboard_event(self, event, *args, **kwargs):
        super()._on_keyboard_event(event, *args, **kwargs)
        # apply the command when pressed
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._INPUT_KEY_MAPPING.keys():
                self._delta_action += self._ACTION_DELTA_MAPPING[self._INPUT_KEY_MAPPING[event.input.name]]
        # remove the command when un-pressed
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._INPUT_KEY_MAPPING.keys():
                self._delta_action -= self._ACTION_DELTA_MAPPING[self._INPUT_KEY_MAPPING[event.input.name]]

    def _create_key_bindings(self):
        """Creates default key binding.
        Based on target frame to control the delta action.
        """
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
        self._INPUT_KEY_MAPPING = {
            "W": "forward",
            "S": "backward",
            "A": "left",
            "D": "right",
            "Q": "up",
            "E": "down",
            "K": "rotate_up",
            "I": "rotate_down",
            "J": "rotate_left",
            "L": "rotate_right",
            "U": "gripper_open",
            "O": "gripper_close",
        }
