import numpy as np


def body_vel_to_wheel_vel(body_vel: np.ndarray, wheel_radius: float = 0.05, base_radius: float = 0.125) -> np.ndarray:
    # Convert rotational velocity from deg/s to rad/s.
    theta_rad = body_vel[2] * (np.pi / 180.0)
    # Create the body velocity vector [x, y, theta_rad].
    velocity_vector = np.array([body_vel[0], body_vel[1], theta_rad])

    # Define the wheel mounting angles with a -90° offset.
    angles = np.radians(np.array([240, 0, 120]) - 90)
    # Build the kinematic matrix: each row maps body velocities to a wheel’s linear speed.
    # The third column (base_radius) accounts for the effect of rotation.
    m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

    # Compute each wheel’s linear speed (m/s) and then its angular speed (rad/s).
    wheel_linear_speeds = m.dot(velocity_vector)
    wheel_angular_speeds = wheel_linear_speeds / wheel_radius  # left, back, right

    # return the wheel angular speeds in the order of back, left, right
    return np.array([wheel_angular_speeds[1], wheel_angular_speeds[0], wheel_angular_speeds[2]])  # back, left, right
