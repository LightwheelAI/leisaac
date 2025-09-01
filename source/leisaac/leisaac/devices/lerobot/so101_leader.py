import os
import json
from collections.abc import Callable
from typing import Dict, Tuple

from .common.motors import FeetechMotorsBus, Motor, MotorNormMode, MotorCalibration, OperatingMode
from .common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from ..device_base import Device

from leisaac.assets.robots.lerobot import SO101_FOLLOWER_MOTOR_LIMITS

import carb
import omni


class SO101Leader(Device):
    """A SO101 Leader device for SE(3) control.
    """

    def __init__(self, env, port: str = '/dev/ttyACM0', recalibrate: bool = False, calibration_file_name: str = 'so101_leader.json'):
        super().__init__(env)
        self.port = port

        # calibration
        self.calibration_path = os.path.join(os.path.dirname(__file__), ".cache", calibration_file_name)
        if not os.path.exists(self.calibration_path) or recalibrate:
            self.calibrate()
        calibration = self._load_calibration()

        self._bus = FeetechMotorsBus(
            port=self.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
                "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
                "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
                "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
                "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=calibration,
        )
        self._motor_limits = SO101_FOLLOWER_MOTOR_LIMITS

        # connect
        self.connect()

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

        self._display_controls()

    def __del__(self):
        """Release the keyboard interface."""
        self.stop_keyboard_listener()

    def stop_keyboard_listener(self):
        if hasattr(self, '_input') and hasattr(self, '_keyboard') and hasattr(self, '_keyboard_sub'):
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of so101 leader."""
        msg = "SO101-Leader device for SE(3) control.\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tMove SO101-Leader to control SO101-Follower\n"
        msg += "\tIf SO101-Follower can't synchronize with SO101-Leader, please add --recalibrate and rerun to recalibrate SO101-Leader.\n"
        return msg

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

    def get_device_state(self):
        return self._bus.sync_read("Present_Position")

    def input2action(self):
        state = {}
        reset = state["reset"] = self._reset_state
        state['started'] = self._started
        if reset:
            self._reset_state = False
            return state
        state['joint_state'] = self.get_device_state()

        ac_dict = {}
        ac_dict["reset"] = reset
        ac_dict['started'] = self._started
        ac_dict['so101_leader'] = True
        if reset:
            return ac_dict
        ac_dict['joint_state'] = state['joint_state']
        ac_dict['motor_limits'] = self._motor_limits
        return ac_dict

    def reset(self):
        pass

    def add_callback(self, key: str, func: Callable):
        self._additional_callbacks[key] = func

    @property
    def started(self) -> bool:
        return self._started

    @property
    def reset_state(self) -> bool:
        return self._reset_state

    @reset_state.setter
    def reset_state(self, reset_state: bool):
        self._reset_state = reset_state

    @property
    def motor_limits(self) -> Dict[str, Tuple[float, float]]:
        return self._motor_limits

    @property
    def is_connected(self) -> bool:
        return self._bus.is_connected

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError("SO101-Leader is not connected.")
        self._bus.disconnect()
        print("SO101-Leader disconnected.")

    def connect(self):
        if self.is_connected:
            raise DeviceAlreadyConnectedError("SO101-Leader is already connected.")
        self._bus.connect()
        self.configure()
        print("SO101-Leader connected.")

    def configure(self) -> None:
        self._bus.disable_torque()
        self._bus.configure_motors()
        for motor in self._bus.motors:
            self._bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    def calibrate(self):
        self._bus = FeetechMotorsBus(
            port=self.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
                "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
                "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
                "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
                "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
        )
        self.connect()

        print("\n Running calibration of SO101-Leader")
        self._bus.disable_torque()
        for motor in self._bus.motors:
            self._bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input("Move SO101-Leader to the middle of its range of motion and press ENTER...")
        homing_offset = self._bus.set_half_turn_homings()
        print("Move all joints sequentially through their entire ranges of motion.")
        print("Recording positions. Press ENTER to stop...")
        range_mins, range_maxes = self._bus.record_ranges_of_motion()

        calibration = {}
        for motor, m in self._bus.motors.items():
            calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offset[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )
        self._bus.write_calibration(calibration)
        self._save_calibration(calibration)
        print(f"Calibration saved to {self.calibration_path}")

        self.disconnect()

    def _load_calibration(self) -> Dict[str, MotorCalibration]:
        with open(self.calibration_path, "r") as f:
            json_data = json.load(f)
        calibration = {}
        for motor_name, motor_data in json_data.items():
            calibration[motor_name] = MotorCalibration(
                id=int(motor_data["id"]),
                drive_mode=int(motor_data["drive_mode"]),
                homing_offset=int(motor_data["homing_offset"]),
                range_min=int(motor_data["range_min"]),
                range_max=int(motor_data["range_max"]),
            )
        return calibration

    def _save_calibration(self, calibration: Dict[str, MotorCalibration]):
        save_calibration = {k: {
            "id": v.id,
            "drive_mode": v.drive_mode,
            "homing_offset": v.homing_offset,
            "range_min": v.range_min,
            "range_max": v.range_max,
        } for k, v in calibration.items()}
        if not os.path.exists(os.path.dirname(self.calibration_path)):
            os.makedirs(os.path.dirname(self.calibration_path))
        with open(self.calibration_path, 'w') as f:
            json.dump(save_calibration, f, indent=4)
