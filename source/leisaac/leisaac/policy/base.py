import warnings
from abc import ABC, abstractmethod

import torch

try:
    import zmq
except ImportError:
    warnings.warn(
        "zmq is not installed, please install it with `pip install pyzmq` for full functionality of ZMQServicePolicy",
        ImportWarning,
    )
try:
    import websockets.sync.client
except ImportError:
    warnings.warn(
        "websockets is not installed, please install it with `pip install websockets` for full functionality of"
        " WebsocketServicePolicy",
        ImportWarning,
    )

from .gr00t import serialization
from .openpi import msgpack_numpy


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


class ZMQServicePolicy(Policy):
    def __init__(self, host: str, port: int, timeout_ms: int = 5000, ping_endpoint: str = "ping"):
        super().__init__("service")
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.context = zmq.Context()
        self._init_socket()
        self._ping_endpoint = ping_endpoint
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

        self.socket.send(serialization.MsgSerializer.to_bytes(request))
        message = self.socket.recv()
        if message == b"ERROR":
            raise RuntimeError("Server error")
        return serialization.MsgSerializer.from_bytes(message)

    def __del__(self):
        """Cleanup resources on destruction"""
        self.socket.close()
        self.context.term()


class WebsocketServicePolicy(Policy):
    def __init__(self, host: str, port: int | None = None, timeout_ms: int = 5000, api_key: str | None = None):
        super().__init__("service")
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._timeout_ms = timeout_ms
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def _wait_for_server(self) -> tuple[websockets.sync.client.ClientConnection, dict]:
        print(f"Waiting for server at {self._uri}...")
        try:
            headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
            conn = websockets.sync.client.connect(
                self._uri, compression=None, max_size=None, additional_headers=headers
            )
            metadata = msgpack_numpy.unpackb(conn.recv())
            return conn, metadata
        except ConnectionRefusedError:
            raise RuntimeError("Failed to connect to policy server.")

    def infer(self, obs: dict) -> dict:  # noqa: UP006
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)
