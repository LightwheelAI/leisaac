import torch
import zmq

from abc import ABC, abstractmethod
from io import BytesIO


class Policy(ABC):
    def __init__(self, type: str):
        self.type = type

    @abstractmethod
    def get_action(self, *args, **kwargs) -> torch.Tensor:
        """
        Get the action from the policy.
        Expect the action to be a torch.Tensor with shape (action_horizon, env_num, action_dim).
        """
        pass


class CheckpointPolicy(Policy):
    def __init__(self, checkpoint_path: str):
        super().__init__("checkpoint")
        self.checkpoint_path = checkpoint_path

    def get_action(self, *args, **kwargs) -> torch.Tensor:
        pass


class TorchSerializer:
    @staticmethod
    def to_bytes(data: dict) -> bytes:
        buffer = BytesIO()
        torch.save(data, buffer)
        return buffer.getvalue()

    @staticmethod
    def from_bytes(data: bytes) -> dict:
        buffer = BytesIO(data)
        obj = torch.load(buffer, weights_only=False)
        return obj


class ServicePolicy(Policy):
    def __init__(self, host: str, port: int, timeout_ms: int = 5000, ping_endpoint: str = "ping", kill_endpoint: str = "kill"):
        super().__init__("service")
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.context = zmq.Context()
        self._init_socket()
        self._ping_endpoint = ping_endpoint
        self._kill_endpoint = kill_endpoint
        self.check_service_status()

    def _init_socket(self):
        """Initialize or reinitialize the socket with current settings"""
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.host}:{self.port}")
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)

    def check_service_status(self):
        if not self._ping():
            raise RuntimeError("Service is not running, please start the service first.")
        else:
            print("Service is running.")

    def _ping(self) -> bool:
        if self._ping_endpoint is None:
            raise ValueError("ping_endpoint is not set")
        try:
            self.call_endpoint(self._ping_endpoint, requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()  # Recreate socket for next attempt
            return False

    def kill_server(self):
        if self._kill_endpoint is None:
            raise ValueError("kill_endpoint is not set")
        self.call_endpoint(self._kill_endpoint, requires_input=False)

    def call_endpoint(self, endpoint: str, data: dict | None = None, requires_input: bool = True) -> dict:
        """
        Call an endpoint on the server.

        Args:
            endpoint: The name of the endpoint.
            data: The input data for the endpoint.
            requires_input: Whether the endpoint requires input data.
        """
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data

        self.socket.send(TorchSerializer.to_bytes(request))
        message = self.socket.recv()
        if message == b"ERROR":
            raise RuntimeError("Server error")
        return TorchSerializer.from_bytes(message)

    def __del__(self):
        """Cleanup resources on destruction"""
        self.socket.close()
        self.context.term()
