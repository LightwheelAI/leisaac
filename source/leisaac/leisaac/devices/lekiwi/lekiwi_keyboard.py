import carb
import numpy as np
import torch
from leisaac.devices.keyboard import SO101Keyboard
from leisaac.utils.robot_utils import convert_lekiwi_wheel_action_robot2env


class LeKiwiKeyboard(SO101Keyboard):
    """A keyboard controller for sending SE(2) commands as velocity commands for lekiwi.

    Key bindings:
        ============================== ================= =================
        Description                    Key               Key
        ============================== ================= =================
        Forward / Backward              UP                DOWN
        Left / Right                    LEFT              RIGHT
        Rotate (Theta) Left / Right     Z                 X
        Speed Level                     1 / 2 / 3
        ============================== ================= =================
    """

    def __init__(self, env, sensitivity: float = 1.0):
        super().__init__(env, sensitivity=sensitivity)
        self.device_type = "lekiwi-keyboard"

        # speed_levels:
        self._speed_levels = [
            {"xy_vel": 0.1, "theta_vel": 30 / 180.0 * np.pi},  # slow
            {"xy_vel": 0.2, "theta_vel": 60 / 180.0 * np.pi},  # medium
            {"xy_vel": 0.3, "theta_vel": 90 / 180.0 * np.pi},  # fast
        ]
        self._speed_index = 0

        # bindings for keyboard to command
        self._create_key_bindings_for_wheel()

        # command buffers (x.vel, y.vel, theta.vel)
        self._vel_command = np.zeros(3)
        self._joint_names = self.env.scene["robot"].data.joint_names

    def __str__(self) -> str:
        """Returns: A string containing the information of keyboard controller."""
        msg = "Keyboard Controller for LeKiwi Control).\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tForward / Backward:                UP / DOWN\n"
        msg += "\tLeft / Right:                      LEFT / RIGHT\n"
        msg += "\tRotate (Theta) Left / Right:       Z / X\n"
        msg += "\tSpeed Level:                       1 / 2 / 3\n"
        msg += "\t----------------------------------------------\n"
        return msg

    def get_device_state(self):
        arm_action = super().get_device_state()

        wheel_action_user = torch.tensor(self._vel_command, device=self.env.device).repeat(self.env.num_envs, 1)

        robot_base_theta = self.env.scene["robot"].data.joint_pos[:, self._joint_names.index("base_theta")]
        wheel_action_world = convert_lekiwi_wheel_action_robot2env(wheel_action_user, robot_base_theta)[0]
        wheel_action_world = wheel_action_world.cpu().numpy()

        return np.concatenate([arm_action, wheel_action_world])

    def reset(self):
        super().reset()
        self._speed_index = 0
        self._vel_command[:] = 0.0

    def _on_keyboard_event(self, event, *args, **kwargs):
        super()._on_keyboard_event(event, *args, **kwargs)
        # apply the command when pressed
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._WHEEL_INPUT_KEY_MAPPING.keys():
                vel_key = self._WHEEL_INPUT_KEY_MAPPING[event.input.name]
                scale_key = "theta_vel" if vel_key in ["rotate_left", "rotate_right"] else "xy_vel"
                self._vel_command += (
                    self._VEL_COMMAND_MAPPING[vel_key] * self._speed_levels[self._speed_index][scale_key]
                )
            if event.input.name in ["KEY_1", "KEY_2", "KEY_3", "NUMPAD_1", "NUMPAD_2", "NUMPAD_3"]:
                self._speed_index = int(event.input.name.split("_")[-1]) - 1
                print(f"Speed level: {self._speed_index + 1}")
        # remove the command when un-pressed
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._WHEEL_INPUT_KEY_MAPPING.keys():
                self._vel_command[:] = 0.0

    def _create_key_bindings_for_wheel(self):
        """Creates default key binding.
        Based on arrow keys to control the velocity command.
        """
        self._VEL_COMMAND_MAPPING = {
            "forward": np.asarray([1.0, 0.0, 0.0]),
            "backward": np.asarray([-1.0, 0.0, 0.0]),
            "left": np.asarray([0.0, 1.0, 0.0]),
            "right": np.asarray([0.0, -1.0, 0.0]),
            "rotate_left": np.asarray([0.0, 0.0, 1.0]),
            "rotate_right": np.asarray([0.0, 0.0, -1.0]),
        }
        self._WHEEL_INPUT_KEY_MAPPING = {
            "UP": "forward",
            "DOWN": "backward",
            "LEFT": "left",
            "RIGHT": "right",
            "Z": "rotate_left",
            "X": "rotate_right",
        }
