# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for teleoperation interface."""

import torch
import numpy as np

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import carb
import omni


class DeviceBase(ABC):
    """An interface class for teleoperation devices."""

    def __init__(self):
        """Initialize the teleoperation interface."""
        pass

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        return f"{self.__class__.__name__}"

    """
    Operations
    """

    @abstractmethod
    def reset(self):
        """Reset the internals."""
        raise NotImplementedError

    @abstractmethod
    def add_callback(self, key: Any, func: Callable):
        """Add additional functions to bind keyboard.

        Args:
            key: The button to check against.
            func: The function to call when key is pressed. The callback function should not
                take any arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def advance(self) -> Any:
        """Provides the joystick event state.

        Returns:
            The processed output form the joystick.
        """
        raise NotImplementedError


class Device(DeviceBase):
    def __init__(self, env, device_type: str):
        """
        Args:
            env (RobotEnv): The environment which contains the robot(s) to control
                            using this device.
        """
        self.env = env
        self.device_type = device_type

        # functional keyboard setup
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            self._on_keyboard_event,
        )

        # some flags and callbacks
        self._started = False
        self._reset_state = False
        self._additional_callbacks = {}

        # display controls
        self._display_controls()

    def __del__(self):
        """Release the keyboard interface."""
        self._stop_keyboard_listener()

    def get_device_state(self):
        raise NotImplementedError

    def input2action(self):
        state = {}
        reset = state["reset"] = self._reset_state
        state['started'] = self.started
        if reset:
            self._reset_state = False
            return state
        state['joint_state'] = self.get_device_state()

        ac_dict = {}
        ac_dict["reset"] = reset
        ac_dict['started'] = self.started
        ac_dict[self.device_type] = True
        if reset:
            return ac_dict
        ac_dict['joint_state'] = state['joint_state']
        return ac_dict

    def advance(self):
        """
        Returns:
            Can be:
                - torch.Tensor: The action to be applied to the robot.
                - dict: state of the scene and the task, and the task need to reset.
                - None: the scene is not started
        """
        action = self.input2action()
        if action is None:
            return self.env.action_manager.action
        if not action['started']:
            return None
        if action['reset']:
            return action
        for key, value in action.items():
            if isinstance(value, np.ndarray):
                action[key] = torch.tensor(value, device=self.env.device, dtype=torch.float32)
        return self.env.cfg.preprocess_device_action(action, self)

    def reset(self):
        pass

    def add_callback(self, key: str, func: Callable):
        self._additional_callbacks[key] = func

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

    def _stop_keyboard_listener(self):
        if hasattr(self, '_input') and hasattr(self, '_keyboard') and hasattr(self, '_keyboard_sub'):
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def _display_controls(self):
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (30 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("b", "start control")
        print_command("r", "reset simulation and set task success to False")
        print_command("n", "reset simulation and set task success to True")
        print_command("move leader", "control follower in the simulation")
        print_command("Control+C", "quit")
        print("")

    @property
    def started(self) -> bool:
        return self._started

    @property
    def reset_state(self) -> bool:
        return self._reset_state

    @reset_state.setter
    def reset_state(self, reset_state: bool):
        self._reset_state = reset_state
