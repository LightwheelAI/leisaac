import carb
import numpy as np
import torch
from leisaac.utils.robot_utils import convert_lekiwi_wheel_action_robot2env

from ..device_base import Device
from ..lerobot import SO101Leader


class LeKiwiLeader(Device):
    def __init__(self, env, port: str = "/dev/ttyACM0", recalibrate: bool = False):
        super().__init__(env, "lekiwi-leader")

        # connect to so101 leader
        print("Connecting to so101 leader...")
        self.so101_leader = SO101Leader(env, port, recalibrate, "lekiwi_leader.json")
        self.so101_leader._stop_keyboard_listener()

        # speed_levels:
        self._speed_levels = [
            {"xy_vel": 0.1, "theta_vel": 30 / 180.0 * np.pi},  # slow
            {"xy_vel": 0.2, "theta_vel": 60 / 180.0 * np.pi},  # medium
            {"xy_vel": 0.3, "theta_vel": 90 / 180.0 * np.pi},  # fast
        ]
        self._speed_index = 0

        self._joint_names = self.env.scene["robot"].data.joint_names

        # bindings for keyboard to command
        self._create_key_bindings_for_wheel()

        # command buffers (x.vel, y.vel, theta.vel)
        self._vel_command = np.zeros(3)
        self._joint_names = self.env.scene["robot"].data.joint_names

    def __str__(self) -> str:
        """Returns: A string containing the information of lekiwi leader."""
        msg = "Keyboard Controller for LeKiwi Wheel Control).\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tForward / Backward:                UP / DOWN\n"
        msg += "\tLeft / Right:                      LEFT / RIGHT\n"
        msg += "\tRotate (Theta) Left / Right:       Z / X\n"
        msg += "\tSpeed Level:                       1 / 2 / 3\n"
        msg += "\t----------------------------------------------\n"
        return msg

    def get_device_state(self):
        arm_action = self.so101_leader.get_device_state()

        wheel_action_user = torch.tensor(self._vel_command, device=self.env.device).repeat(self.env.num_envs, 1)

        robot_base_theta = self.env.scene["robot"].data.joint_pos[:, self._joint_names.index("base_theta")]
        wheel_action_world = convert_lekiwi_wheel_action_robot2env(wheel_action_user, robot_base_theta)[0]

        return {
            "arm_action": arm_action,
            "wheel_action": wheel_action_world,
        }

    def input2action(self):
        ac_dict = super().input2action()
        ac_dict["motor_limits"] = self.so101_leader.motor_limits
        return ac_dict

    def reset(self):
        super().reset()
        self.so101_leader.reset()
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
