import torch
import numpy as np
import carb
import omni

from collections.abc import Callable

import isaaclab.utils.math as math_utils
from leisaac.utils.math_utils import rotvec_to_euler

from .gamepad_utils import GamepadController
from ..device_base import Device


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
        """Initialize the gamepad layer.

        Args:
            env: The environment which contains the robot(s) to control.
            sensitivity: Sensitivity multiplier for all inputs. Defaults to 1.0.
        """
        super().__init__(env)

        # initialize gamepad controller
        self._gamepad = GamepadController()
        self._gamepad.start()
        if "xbox" not in self._gamepad.name:
            raise ValueError("Only Xbox gamepads are supported. Please connect an Xbox gamepad and try again.")

        # store inputs
        self.pos_sensitivity = 0.01 * sensitivity
        self.joint_sensitivity = 0.20 * sensitivity
        self.rot_sensitivity = 0.15 * sensitivity

        # functional keyboard setup
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            self._on_keyboard_event,
        )

        # command buffers
        # (dx, dy, dz, droll, dpitch, dyaw, d_shoulder_pan, d_gripper)
        self._delta_action = np.zeros(8)
        self._create_action_mapping()

        # some flags and callbacks
        self._started = False
        self._reset_state = False
        self._additional_callbacks = {}

        # initialize the target frame
        self.asset_name = 'robot'
        self.robot_asset = self.env.scene[self.asset_name]

        self.target_frame = 'gripper'
        body_idxs, _ = self.robot_asset.find_bodies(self.target_frame)
        self.target_frame_idx = body_idxs[0]

    def __del__(self):
        """Release the gamepad interface."""
        self._gamepad.stop()
        if hasattr(self, '_input') and hasattr(self, '_keyboard') and hasattr(self, '_keyboard_sub'):
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of gamepad controller."""
        msg = "Gamepad Controller for SO101 Single Arm (SE(3) Control).\n"
        msg += f"\tGamepad name: {self._gamepad.name}\n"
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
        return self._delta_action

    def input2action(self):
        state = {}
        reset = state["reset"] = self._reset_state
        state['started'] = self._started
        if reset:
            self._reset_state = False
            return state
        state['joint_state'] = self._convert_delta_from_frame(self.get_device_state())

        ac_dict = {}
        ac_dict["reset"] = reset
        ac_dict['started'] = self._started
        ac_dict['gamepad'] = True
        if reset:
            return ac_dict
        ac_dict['joint_state'] = state['joint_state']
        return ac_dict

    def reset(self):
        self._delta_action[:] = 0.0

    def add_callback(self, key: str, func: Callable):
        self._additional_callbacks[key] = func

    def advance(self):
        self._delta_action[:] = 0.0
        self._update_action()
        return super().advance()

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Handle keyboard events using carb."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "B":
                self._started = True
                self._reset_state = False
            elif event.input.name == "R":
                self._started = False
                self._reset_state = True
                if "R" in self._additional_callbacks:
                    self._additional_callbacks["R"]()
            elif event.input.name == "N":
                self._started = False
                self._reset_state = True
                if "N" in self._additional_callbacks:
                    self._additional_callbacks["N"]()
        return True

    def _create_action_mapping(self):
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

    def _convert_delta_from_frame(self, delta_action: np.ndarray) -> np.ndarray:
        """
        Convert delta action from target frame to robot base frame.
        target_frame -> root_frame
        Args:
            delta_action: Delta action in target frame.
        Returns:
            Delta action in robot base frame.
        """
        if np.allclose(delta_action[:3], 0.0) and np.allclose(delta_action[3:6], 0.0):
            return delta_action
        is_delta_rot = not np.allclose(delta_action[3:6], 0.0)

        torch_delta_action = torch.tensor(delta_action, device=self.env.device, dtype=torch.float32)

        delta_pos_f = torch_delta_action[:3].repeat(self.env.num_envs, 1)
        delta_rot_f = torch_delta_action[3:6].repeat(self.env.num_envs, 1)
        delta_quat_f = math_utils.quat_from_euler_xyz(delta_rot_f[:, 0], delta_rot_f[:, 1], delta_rot_f[:, 2])
        delta_rotvec_f = math_utils.axis_angle_from_quat(delta_quat_f)

        frame_pos, frame_quat = self.robot_asset.data.root_pos_w, self.robot_asset.data.body_quat_w[:, self.target_frame_idx]  # don't consider frame pos here
        root_pos, root_quat = self.robot_asset.data.root_pos_w, self.robot_asset.data.root_quat_w
        _, frame2root = math_utils.subtract_frame_transforms(root_pos, root_quat, frame_pos, frame_quat)
        frame2root_quat = math_utils.quat_unique(frame2root)

        delta_pos_r = math_utils.quat_apply(frame2root_quat, delta_pos_f)
        delta_rotvec_r = math_utils.quat_apply(frame2root_quat, delta_rotvec_f)
        delta_rot_r = rotvec_to_euler(delta_rotvec_r) if is_delta_rot else torch.zeros(3, device=self.env.device)

        delta_action_r = torch.cat([delta_pos_r.squeeze(0), delta_rot_r, torch_delta_action[6:]], dim=0)

        return delta_action_r.cpu().numpy()
