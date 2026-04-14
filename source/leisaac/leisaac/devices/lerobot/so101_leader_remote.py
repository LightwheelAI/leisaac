"""SO101 Leader device that receives joint states from a remote ZMQ publisher.

Use with ``joint_state_server.py`` running on the machine where the leader arm
is physically connected.  This device is a drop-in replacement for
:class:`SO101Leader` — it uses the same device type (``so101_leader``) so no
changes to action processing or environment configuration are needed.
"""

import struct
import threading

from leisaac.assets.robots.lerobot import SO101_FOLLOWER_MOTOR_LIMITS

from ..device_base import Device

MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class SO101LeaderRemote(Device):
    """A SO101 Leader device that reads joint states over ZMQ instead of a local serial bus."""

    def __init__(self, env, endpoint: str = "tcp://localhost:5556"):
        self._endpoint = endpoint
        self._motor_limits = SO101_FOLLOWER_MOTOR_LIMITS
        self._connected = False
        self._ctx = None
        self._sub = None
        self._lock = threading.Lock()
        self._cached = dict.fromkeys(MOTOR_NAMES, 0.0)
        self._recv_thread = None
        super().__init__(env, "so101_leader")
        self.connect()

    def _recv_loop(self):
        while self._connected:
            try:
                msg = self._sub.recv(flags=0)  # blocks here, no lock held
                state = dict(zip(MOTOR_NAMES, struct.unpack("<6f", msg)))
                with self._lock:
                    self._cached = state
            except Exception:
                break

    def _add_device_control_description(self):
        self._display_controls_table.add_row(["so101-leader-remote", f"ZMQ endpoint: {self._endpoint}"])

    def get_device_state(self):
        with self._lock:
            return dict(self._cached)

    def input2action(self):
        ac_dict = super().input2action()
        ac_dict["motor_limits"] = self._motor_limits
        return ac_dict

    @property
    def motor_limits(self) -> dict[str, tuple[float, float]]:
        return self._motor_limits

    @property
    def is_connected(self) -> bool:
        return self._connected

    def disconnect(self):
        self._connected = False
        if self._recv_thread:
            self._recv_thread.join(timeout=2)
        if self._sub:
            self._sub.close()
        if self._ctx:
            self._ctx.term()
        print("SO101-Leader-Remote disconnected.")

    def connect(self):
        import zmq

        self._ctx = zmq.Context()
        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.setsockopt(zmq.CONFLATE, 1)
        self._sub.setsockopt(zmq.SUBSCRIBE, b"")
        self._sub.connect(self._endpoint)
        self._connected = True
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()
        print(f"SO101-Leader-Remote connected to {self._endpoint}")

    def configure(self) -> None:
        pass  # configuration happens on the publisher side

    def calibrate(self):
        raise RuntimeError("Calibrate on the machine where the leader arm is connected (lerobot-calibrate).")
