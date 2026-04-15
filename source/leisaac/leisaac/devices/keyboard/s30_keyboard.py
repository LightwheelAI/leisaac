import carb
import numpy as np

from ..device_base import Device


class S30Keyboard(Device):
    """键盘控制器，用于 S30 + Robotiq 2F-140 机械臂的末端执行器控制。

    发送 SE(3) 增量位姿指令，通过差分逆运动学（DiffIK）驱动 S30 六轴臂，
    同时独立控制夹爪。

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

    Output (7D): [dx, dy, dz, droll, dpitch, dyaw, d_gripper]
    """

    def __init__(self, env, sensitivity: float = 1.0):
        super().__init__(env, "s30-keyboard")

        self.pos_sensitivity = 0.03 * sensitivity
        self.rot_sensitivity = 0.30 * sensitivity
        self.joint_sensitivity = 0.30 * sensitivity

        self._create_key_bindings()

        # 7D: (dx, dy, dz, droll, dpitch, dyaw, d_gripper)
        self._delta_action = np.zeros(7)

        self.asset_name = "robot"
        self.robot_asset = self.env.scene[self.asset_name]

        self.target_frame = "elfin_link6"
        body_idxs, _ = self.robot_asset.find_bodies(self.target_frame)
        self.target_frame_idx = body_idxs[0]

    def _add_device_control_description(self):
        self._display_controls_table.add_row(["W", "forward"])
        self._display_controls_table.add_row(["S", "backward"])
        self._display_controls_table.add_row(["A", "left"])
        self._display_controls_table.add_row(["D", "right"])
        self._display_controls_table.add_row(["Q", "up"])
        self._display_controls_table.add_row(["E", "down"])
        self._display_controls_table.add_row(["J", "rotate_left (yaw)"])
        self._display_controls_table.add_row(["L", "rotate_right (yaw)"])
        self._display_controls_table.add_row(["K", "rotate_down (pitch)"])
        self._display_controls_table.add_row(["I", "rotate_up (pitch)"])
        self._display_controls_table.add_row(["U", "gripper_open"])
        self._display_controls_table.add_row(["O", "gripper_close"])

    def get_device_state(self):
        return self._convert_delta_from_frame(self._delta_action)

    def reset(self):
        self._delta_action[:] = 0.0

    def _on_keyboard_event(self, event, *args, **kwargs):
        super()._on_keyboard_event(event, *args, **kwargs)
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._INPUT_KEY_MAPPING:
                self._delta_action += self._ACTION_DELTA_MAPPING[self._INPUT_KEY_MAPPING[event.input.name]]
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._INPUT_KEY_MAPPING:
                self._delta_action -= self._ACTION_DELTA_MAPPING[self._INPUT_KEY_MAPPING[event.input.name]]

    def _create_key_bindings(self):
        self._ACTION_DELTA_MAPPING = {
            "forward":       np.asarray([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0]) * self.pos_sensitivity,
            "backward":      np.asarray([0.0, 0.0,  1.0, 0.0, 0.0, 0.0, 0.0]) * self.pos_sensitivity,
            "left":          np.asarray([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.pos_sensitivity,
            "right":         np.asarray([ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.pos_sensitivity,
            "up":            np.asarray([0.0,  1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.pos_sensitivity,
            "down":          np.asarray([0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.pos_sensitivity,
            "rotate_left":   np.asarray([0.0, 0.0, 0.0, 0.0, 0.0,  1.0, 0.0]) * self.rot_sensitivity,
            "rotate_right":  np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0]) * self.rot_sensitivity,
            "rotate_up":     np.asarray([0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0]) * self.rot_sensitivity,
            "rotate_down":   np.asarray([0.0, 0.0, 0.0, 0.0,  1.0, 0.0, 0.0]) * self.rot_sensitivity,
            "gripper_open":  np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  1.0]) * self.joint_sensitivity,
            "gripper_close": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]) * self.joint_sensitivity,
        }
        self._INPUT_KEY_MAPPING = {
            "W": "forward",
            "S": "backward",
            "A": "left",
            "D": "right",
            "Q": "up",
            "E": "down",
            "J": "rotate_left",
            "L": "rotate_right",
            "K": "rotate_down",
            "I": "rotate_up",
            "U": "gripper_open",
            "O": "gripper_close",
        }
