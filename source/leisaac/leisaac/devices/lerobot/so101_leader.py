import json
import os

from leisaac.assets.robots.lerobot import SO101_FOLLOWER_MOTOR_LIMITS

from ..device_base import Device
from .common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from .common.motors import (
    FeetechMotorsBus,
    Motor,
    MotorCalibration,
    MotorNormMode,
    OperatingMode,
)


class SO101Leader(Device):
    """A SO101 Leader device for SE(3) control."""

    def __init__(
        self,
        env,
        port: str = "/dev/ttyACM0",
        recalibrate: bool = False,
        calibration_file_name: str = "so101_leader.json",
    ):
        super().__init__(env, "so101_leader")
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

    def __str__(self) -> str:
        """Returns: A string containing the information of so101 leader."""
        msg = "SO101-Leader device for SE(3) control.\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tMove SO101-Leader to control SO101-Follower\n"
        msg += (
            "\tIf SO101-Follower can't synchronize with SO101-Leader, please add --recalibrate and rerun to recalibrate"
            " SO101-Leader.\n"
        )
        msg += "\t----------------------------------------------\n"
        return msg

    def get_device_state(self):
        return self._bus.sync_read("Present_Position")

    def input2action(self):
        ac_dict = super().input2action()
        ac_dict["motor_limits"] = self._motor_limits
        return ac_dict

    @property
    def motor_limits(self) -> dict[str, tuple[float, float]]:
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

    def _load_calibration(self) -> dict[str, MotorCalibration]:
        with open(self.calibration_path) as f:
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

    def _save_calibration(self, calibration: dict[str, MotorCalibration]):
        save_calibration = {
            k: {
                "id": v.id,
                "drive_mode": v.drive_mode,
                "homing_offset": v.homing_offset,
                "range_min": v.range_min,
                "range_max": v.range_max,
            }
            for k, v in calibration.items()
        }
        if not os.path.exists(os.path.dirname(self.calibration_path)):
            os.makedirs(os.path.dirname(self.calibration_path))
        with open(self.calibration_path, "w") as f:
            json.dump(save_calibration, f, indent=4)
