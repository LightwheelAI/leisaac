import argparse

import numpy as np
from scipy.spatial.transform import Rotation as R


def compute_lookat(eye, euler_deg, distance=2.0):
    """Compute lookat point from eye position and euler angles.

    Args:
        eye: Camera eye position [x, y, z]
        euler_deg: Camera rotation in degrees [roll, pitch, yaw]
        distance: Distance from eye to lookat point (default: 2.0)
    """
    eye = np.array(eye)

    rot = R.from_euler("xyz", euler_deg, degrees=True)
    rot_matrix = rot.as_matrix()

    forward = rot_matrix @ np.array([0, 0, -1])

    lookat = eye + forward * distance

    print(f"  self.viewer.eye = ({eye[0]:.5f}, {eye[1]:.5f}, {eye[2]:.5f})")
    print(f"  self.viewer.lookat = ({lookat[0]:.5f}, {lookat[1]:.5f}, {lookat[2]:.5f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute camera lookat point from eye position and euler angles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--eye",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
        help="Camera eye position [x, y, z]",
    )
    parser.add_argument(
        "--euler",
        type=float,
        nargs=3,
        required=True,
        metavar=("ROLL", "PITCH", "YAW"),
        help="Camera rotation in degrees [roll, pitch, yaw]",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=2.0,
        help="Distance from eye to lookat point (default: 2.0)",
    )

    args = parser.parse_args()

    compute_lookat(args.eye, args.euler, distance=args.distance)
