import weakref
import torch
import numpy as np

from collections.abc import Callable

import carb
import omni

import isaaclab.utils.math as math_utils

from leisaac.utils.math_utils import rotvec_to_euler

from ..device_base import Device


class SO101Keyboard(Device):
    """A keyboard controller for sending SE(3) commands as delta poses for so101 single arm.

    Key bindings:
        ============================== ================= =================
        Description                    Key (+ve axis)    Key (-ve axis)
        ============================== ================= =================
        Forward/Backward                W                 S
        Left/Right                      A                 D
        Up/Down                         Q                 E
        Rotate (Yaw) Left/Right         J                 L
        Rotate (Pitch) Raise/Lower      I                 K
        Gripper Open/Close              U                 O
        ============================== ================= =================

    """

    def __init__(self, env, sensitivity: float = 1.0):
        super().__init__(env)
        """Initialize the keyboard layer.
        """
        # store inputs
        self.pos_sensitivity = 0.01 * sensitivity
        self.joint_sensitivity = 0.20 * sensitivity
        self.rot_sensitivity = 0.15 * sensitivity

        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called.
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )
        # bindings for keyboard to command
        self._create_key_bindings()

        # command buffers
        self._delta_action = np.zeros(8)  # (dx, dy, dz, droll, dpitch, dyaw, d_shoulder_pan, d_gripper)

        # some flags and callbacks
        self.started = False
        self._reset_state = 0
        self._additional_callbacks = {}

        # initialize the target frame
        self.asset_name = 'robot'
        self.robot_asset = self.env.scene[self.asset_name]

        self.target_frame = 'gripper'
        body_idxs, _ = self.robot_asset.find_bodies(self.target_frame)
        self.target_frame_idx = body_idxs[0]

    def __del__(self):
        """Release the keyboard interface."""
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of keyboard controller."""
        msg = "Keyboard Controller for SO101 Single Arm (SE(3) Control).\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += f"target frame: {self.target_frame}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tForward/Backward:                W / S\n"
        msg += "\tLeft/Right:                      A / D\n"
        msg += "\tUp/Down:                         Q / E\n"
        msg += "\tRotate (Yaw) Left/Right:         J / L\n"
        msg += "\tRotate (Pitch) Raise/Lower:      I / K\n"
        msg += "\tGripper Open/Close:              U / O\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tStart Control: B\n"
        msg += "\tTask Failed and Reset: R\n"
        msg += "\tTask Success and Reset: N\n"
        msg += "\tControl+C: quit"
        msg += "\t----------------------------------------------\n"
        return msg

    def get_device_state(self):
        return self._delta_action

    def input2action(self):
        state = {}
        reset = state["reset"] = self._reset_state
        state['started'] = self.started
        if reset:
            self._reset_state = False
            return state
        state['joint_state'] = self._convert_delta_from_frame(self.get_device_state())

        ac_dict = {}
        ac_dict["reset"] = reset
        ac_dict['started'] = self.started
        ac_dict['keyboard'] = True
        if reset:
            return ac_dict
        ac_dict['joint_state'] = state['joint_state']
        return ac_dict

    def reset(self):
        self._delta_action = np.zeros(8)

    def add_callback(self, key: str, func: Callable):
        self._additional_callbacks[key] = func

    def _on_keyboard_event(self, event, *args, **kwargs):
        # apply the command when pressed
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._INPUT_KEY_MAPPING.keys():
                self._delta_action += self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name == "B":
                self.started = True
                self._reset_state = False
            elif event.input.name == "R":
                self.started = False
                self._reset_state = True
                if "R" in self._additional_callbacks:
                    self._additional_callbacks["R"]()
            elif event.input.name == "N":
                self.started = False
                self._reset_state = True
                if "N" in self._additional_callbacks:
                    self._additional_callbacks["N"]()
        # remove the command when un-pressed
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._INPUT_KEY_MAPPING.keys():
                self._delta_action -= self._INPUT_KEY_MAPPING[event.input.name]
        return True

    def _create_key_bindings(self):
        """Creates default key binding.
        Based on target frame to control the delta action.
        """
        self._INPUT_KEY_MAPPING = {
            "W": np.asarray([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.pos_sensitivity,
            "S": np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.pos_sensitivity,
            "A": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0]) * self.joint_sensitivity,
            "D": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]) * self.joint_sensitivity,
            "Q": np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.pos_sensitivity,
            "E": np.asarray([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.pos_sensitivity,
            "I": np.asarray([0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0]) * self.rot_sensitivity,
            "K": np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) * self.rot_sensitivity,
            "J": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]) * self.rot_sensitivity,
            "L": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0]) * self.rot_sensitivity,
            "U": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) * self.joint_sensitivity,
            "O": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]) * self.joint_sensitivity,
        }

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

        delta_pos_i = math_utils.quat_apply(frame2root_quat, delta_pos_f)
        delta_rotvec_i = math_utils.quat_apply(frame2root_quat, delta_rotvec_f)
        delta_rot_i = rotvec_to_euler(delta_rotvec_i) if is_delta_rot else torch.zeros(3, device=self.env.device)

        delta_action_i = torch.cat([delta_pos_i.squeeze(0), delta_rot_i, torch_delta_action[6:]], dim=0)

        return delta_action_i.cpu().numpy()
