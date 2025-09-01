import weakref
import numpy as np

from collections.abc import Callable

import carb
import omni

from ..device_base import Device


class Se3Keyboard(Device):
    """A keyboard controller for sending SE(3) commands as delta poses for lerobot.

    Key bindings:
        ============================== ================= =================
        Description                    Key (+ve axis)    Key (-ve axis)
        ============================== ================= =================
        Joint 1 (shoulder_pan)         Q                 U
        Joint 2 (shoulder_lift)        W                 I
        Joint 3 (elbow_flex)           E                 O
        Joint 4 (wrist_flex)           A                 J
        Joint 5 (wrist_roll)           S                 K
        Joint 6 (gripper)              D                 L
        ============================== ================= =================

    """

    def __init__(self, env, sensitivity: float = 0.05):
        super().__init__(env)
        """Initialize the keyboard layer.
        """
        # store inputs
        self.sensitivity = sensitivity

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
        self._delta_pos = np.zeros(6)

        # some flags and callbacks
        self.started = False
        self._reset_state = 0
        self._additional_callbacks = {}

    def __del__(self):
        """Release the keyboard interface."""
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = "Keyboard Controller for SE(3).\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tJoint 1 (shoulder_pan):  Q/U\n"
        msg += "\tJoint 2 (shoulder_lift): W/I\n"
        msg += "\tJoint 3 (elbow_flex):    E/O\n"
        msg += "\tJoint 4 (wrist_flex):    A/J\n"
        msg += "\tJoint 5 (wrist_roll):    S/K\n"
        msg += "\tJoint 6 (gripper):       D/L\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tStart Control: B\n"
        msg += "\tTask Failed and Reset: R\n"
        msg += "\tTask Success and Reset: N\n"
        msg += "\tControl+C: quit"
        return msg

    def get_device_state(self):
        return self._delta_pos

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
        ac_dict['keyboard'] = True
        if reset:
            return ac_dict
        ac_dict['joint_state'] = state['joint_state']
        return ac_dict

    def reset(self):
        self._delta_pos = np.zeros(6)

    def add_callback(self, key: str, func: Callable):
        self._additional_callbacks[key] = func

    def _on_keyboard_event(self, event, *args, **kwargs):
        # apply the command when pressed
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._INPUT_KEY_MAPPING.keys():
                self._delta_pos += self._INPUT_KEY_MAPPING[event.input.name]
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
                self._delta_pos -= self._INPUT_KEY_MAPPING[event.input.name]
        return True

    def _create_key_bindings(self):
        """Creates default key binding."""
        self._INPUT_KEY_MAPPING = {
            "Q": np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.sensitivity,
            "W": np.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]) * self.sensitivity,
            "E": np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) * self.sensitivity,
            "A": np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]) * self.sensitivity,
            "S": np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]) * self.sensitivity,
            "D": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) * self.sensitivity,
            "U": np.asarray([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.sensitivity,
            "I": np.asarray([0.0, -1.0, 0.0, 0.0, 0.0, 0.0]) * self.sensitivity,
            "O": np.asarray([0.0, 0.0, -1.0, 0.0, 0.0, 0.0]) * self.sensitivity,
            "J": np.asarray([0.0, 0.0, 0.0, -1.0, 0.0, 0.0]) * self.sensitivity,
            "K": np.asarray([0.0, 0.0, 0.0, 0.0, -1.0, 0.0]) * self.sensitivity,
            "L": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, -1.0]) * self.sensitivity,
        }
