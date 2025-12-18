import numpy as np
import torch
from leisaac.devices.gamepad import SO101Gamepad
from leisaac.utils.robot_utils import convert_lekiwi_wheel_action_robot2env


class LeKiwiGamepad(SO101Gamepad):
    """A gamepad controller for sending SE(2) commands as velocity commands for lekiwi.

    Key bindings:
        ============================== ================= =================
        Description                    Key               Key
        ============================== ================= =================
        Forward / Backward             hats-UP           hats-DOWN
        Left / Right                   hats-LEFT         hats-RIGHT
        Rotate (Theta) Left / Right    buttons-X         buttons-B
        Speed Level up / down          buttons-Y         buttons-A
        ============================== ================= =================
    """

    def __init__(self, env, sensitivity: float = 1.0):
        super().__init__(env, sensitivity=sensitivity)
        self.device_type = "lekiwi-gamepad"

        # speed_levels:
        self._speed_levels = [
            {"xy_vel": 0.1, "theta_vel": 30 / 180.0 * np.pi},  # slow
            {"xy_vel": 0.2, "theta_vel": 60 / 180.0 * np.pi},  # medium
            {"xy_vel": 0.3, "theta_vel": 90 / 180.0 * np.pi},  # fast
        ]
        self._speed_index = 0
        # for edge-triggered speed control (buttons Y/A)
        self._last_speed_button_state: dict[str, bool] = {"speed_up": False, "speed_down": False}

        self._create_key_bindings_for_wheel()
        self._action_update_list.append(self._update_wheel_action)

        # command buffers (x.vel, y.vel, theta.vel)
        self._vel_command = np.zeros(3)
        self._joint_names = self.env.scene["robot"].data.joint_names

    def __str__(self) -> str:
        """Returns: A string containing the information of gamepad controller."""
        msg = "Gamepad Controller for LeKiwi Control).\n"
        msg += f"\tGamepad name: {self._gamepad.name}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tForward / Backward:                hats-UP / hats-DOWN\n"
        msg += "\tLeft / Right:                      hats-LEFT / hats-RIGHT\n"
        msg += "\tRotate (Theta) Left / Right:       buttons-X / buttons-B\n"
        msg += "\tSpeed Level up / down:             buttons-Y / buttons-A\n"
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

    def advance(self):
        self._vel_command[:] = 0.0
        return super().advance()

    def _create_key_bindings_for_wheel(self):
        """Creates default key binding.
        Based on gamepad to control the velocity command.
        """
        self._VEL_COMMAND_MAPPING = {
            "forward": np.asarray([1.0, 0.0, 0.0]),
            "backward": np.asarray([-1.0, 0.0, 0.0]),
            "left": np.asarray([0.0, 1.0, 0.0]),
            "right": np.asarray([0.0, -1.0, 0.0]),
            "rotate_left": np.asarray([0.0, 0.0, 1.0]),
            "rotate_right": np.asarray([0.0, 0.0, -1.0]),
        }
        self._WHEEL_INPUT_KEY_MAPPING_LIST = [
            ("forward", "UP"),
            ("backward", "DOWN"),
            ("left", "LEFT"),
            ("right", "RIGHT"),
            ("rotate_left", "X"),
            ("rotate_right", "B"),
            ("speed_up", "Y"),
            ("speed_down", "A"),
        ]

    def _update_wheel_action(self):
        """Update the delta action based on the gamepad state."""
        for input_key_mapping in self._WHEEL_INPUT_KEY_MAPPING_LIST:
            action_name, controller_name = input_key_mapping[0], input_key_mapping[1]
            reverse = input_key_mapping[2] if len(input_key_mapping) > 2 else False
            controller_state = self._gamepad.get_state()
            is_activate, is_positive = self._gamepad.lookup_controller_state(controller_state, controller_name, reverse)
            pressed = is_activate and is_positive
            if action_name in ["speed_up", "speed_down"]:
                last_pressed = self._last_speed_button_state[action_name]
                if pressed and not last_pressed:
                    if action_name == "speed_up":
                        self._speed_index = min(self._speed_index + 1, len(self._speed_levels) - 1)
                        print(f"Speed level: {self._speed_index + 1}")
                    elif action_name == "speed_down":
                        self._speed_index = max(self._speed_index - 1, 0)
                        print(f"Speed level: {self._speed_index + 1}")
                self._last_speed_button_state[action_name] = pressed
            else:
                if pressed:
                    scale_key = "theta_vel" if action_name in ["rotate_left", "rotate_right"] else "xy_vel"
                    self._vel_command += (
                        self._VEL_COMMAND_MAPPING[action_name] * self._speed_levels[self._speed_index][scale_key]
                    )
