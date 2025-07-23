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
    def __init__(self, env):
        """
        Args:
            env (RobotEnv): The environment which contains the robot(s) to control
                            using this device.
        """
        self.env = env

    def get_device_state(self):
        raise NotImplementedError

    def input2action(self):
        raise NotImplementedError

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
