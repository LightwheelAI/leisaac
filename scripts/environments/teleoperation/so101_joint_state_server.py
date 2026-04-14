# /// script
# requires-python = ">=3.10"
# dependencies = ["leisaac[remote]"]
# ///
"""Publish normalized SO101 leader arm joint states over ZMQ PUB.

Run this on the machine where the leader arm is physically connected.
A remote LeIsaac instance subscribes to receive joint states for teleoperation.

Usage:
    python so101_joint_state_server.py --port /dev/ttyACM0 --id leader_arm --rate 50
    python so101_joint_state_server.py --port /dev/ttyACM0 --id leader_arm --recalibrate
"""

import argparse
import json
import os
import struct
import time

import zmq

try:
    from leisaac.devices.lerobot.common.motors import (
        FeetechMotorsBus,
        Motor,
        MotorCalibration,
        MotorNormMode,
    )
except ImportError as e:
    raise ImportError("leisaac is required. Install with: pip install leisaac[remote]") from e

MOTORS = {
    "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
    "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}
MOTOR_NAMES = list(MOTORS.keys())
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")


def get_calibration_path(arm_id: str) -> str:
    return os.path.join(CACHE_DIR, f"{arm_id}.json")


def load_calibration(path: str) -> dict[str, MotorCalibration]:
    with open(path) as f:
        data = json.load(f)
    return {
        name: MotorCalibration(
            id=int(d["id"]),
            drive_mode=int(d["drive_mode"]),
            homing_offset=int(d["homing_offset"]),
            range_min=int(d["range_min"]),
            range_max=int(d["range_max"]),
        )
        for name, d in data.items()
    }


def save_calibration(path: str, calibration: dict[str, MotorCalibration]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        k: {
            "id": v.id,
            "drive_mode": v.drive_mode,
            "homing_offset": v.homing_offset,
            "range_min": v.range_min,
            "range_max": v.range_max,
        }
        for k, v in calibration.items()
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def calibrate(port: str, arm_id: str) -> dict[str, MotorCalibration]:
    """Run interactive calibration using leisaac's calibration logic."""
    print(f"\nRunning calibration for {arm_id} on {port}")

    bus = FeetechMotorsBus(port=port, motors=MOTORS)
    bus.connect()
    bus.disable_torque()

    input("Move the leader arm to the MIDDLE of its range of motion and press ENTER...")
    homing_offsets = bus.set_half_turn_homings()

    print("Move all joints through their FULL range of motion.")
    print("Press ENTER when done...")
    range_mins, range_maxes = bus.record_ranges_of_motion()

    calibration = {
        name: MotorCalibration(
            id=MOTORS[name].id,
            drive_mode=0,
            homing_offset=homing_offsets[name],
            range_min=range_mins[name],
            range_max=range_maxes[name],
        )
        for name in MOTOR_NAMES
    }

    path = get_calibration_path(arm_id)
    save_calibration(path, calibration)
    print(f"Calibration saved to {path}")

    bus.disconnect()
    return calibration


def main():
    parser = argparse.ArgumentParser(description="SO101 leader arm joint state publisher")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--id", default="leader_arm", help="Calibration ID")
    parser.add_argument("--bind", default="tcp://0.0.0.0:5556")
    parser.add_argument("--rate", type=int, default=50, help="Publish rate in Hz")
    parser.add_argument("--recalibrate", action="store_true", help="Force recalibration")
    args = parser.parse_args()

    calib_path = get_calibration_path(args.id)

    if args.recalibrate or not os.path.exists(calib_path):
        calibration = calibrate(args.port, args.id)
    else:
        calibration = load_calibration(calib_path)
        print(f"Loaded calibration from {calib_path}")

    bus = FeetechMotorsBus(port=args.port, motors=MOTORS, calibration=calibration)
    bus.connect()
    bus.disable_torque()

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.CONFLATE, 1)
    pub.bind(args.bind)
    print(f"Publishing on {args.bind} at {args.rate} Hz")
    time.sleep(0.5)

    interval = 1.0 / args.rate
    count = 0
    next_t = time.monotonic()

    try:
        while True:
            positions = bus.sync_read("Present_Position")
            values = [positions[name] for name in MOTOR_NAMES]
            pub.send(struct.pack("<6f", *values), zmq.NOBLOCK)

            count += 1
            if count % (args.rate * 10) == 0:
                print(f"{count} msgs sent")

            next_t += interval
            sleep_t = next_t - time.monotonic()
            if sleep_t > 0:
                time.sleep(sleep_t)
    except KeyboardInterrupt:
        print(f"\nDone: {count} msgs")
    finally:
        bus.disconnect()
        pub.close()
        ctx.term()


if __name__ == "__main__":
    main()
