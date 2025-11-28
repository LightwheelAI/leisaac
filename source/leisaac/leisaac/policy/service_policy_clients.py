import pickle
import time

import grpc
import numpy as np
import torch
from leisaac.utils.constant import SINGLE_ARM_JOINT_NAMES
from leisaac.utils.robot_utils import (
    convert_leisaac_action_to_lerobot,
    convert_lerobot_action_to_leisaac,
)

from .base import Policy, WebsocketServicePolicy, ZMQServicePolicy
from .lerobot.helpers import RemotePolicyConfig, TimedObservation
from .lerobot.transport import services_pb2, services_pb2_grpc
from .lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from .openpi import image_tools


class Gr00tServicePolicyClient(ZMQServicePolicy):
    """
    Service policy client for GR00T N1.5: https://github.com/NVIDIA/Isaac-GR00T
    Target Commit: https://github.com/NVIDIA/Isaac-GR00T/commit/4af2b622892f7dcb5aae5a3fb70bcb02dc217b96
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 5000,
        camera_keys: list[str] = ["front", "wrist"],
        modality_keys: list[str] = ["single_arm", "gripper"],
    ):
        """
        Args:
            host: Host of the policy server.
            port: Port of the policy server.
            camera_keys: Keys of the cameras.
            timeout_ms: Timeout of the policy server.
            modality_keys: Keys of the modality.
        """
        super().__init__(host=host, port=port, timeout_ms=timeout_ms, ping_endpoint="ping")
        self.camera_keys = camera_keys
        self.modality_keys = modality_keys

    def get_action(self, observation_dict: dict) -> torch.Tensor:
        obs_dict = {f"video.{key}": observation_dict[key].cpu().numpy().astype(np.uint8) for key in self.camera_keys}

        if "single_arm" in self.modality_keys:
            joint_pos = convert_leisaac_action_to_lerobot(observation_dict["joint_pos"])
            obs_dict["state.single_arm"] = joint_pos[:, 0:5].astype(np.float64)
            obs_dict["state.gripper"] = joint_pos[:, 5:6].astype(np.float64)
        # TODO: add bi-arm support

        obs_dict["annotation.human.task_description"] = [observation_dict["task_description"]]

        """
            Example of obs_dict for single arm task:
            obs_dict = {
                "video.front": np.zeros((1, 480, 640, 3), dtype=np.uint8),
                "video.wrist": np.zeros((1, 480, 640, 3), dtype=np.uint8),
                "state.single_arm": np.zeros((1, 5)),
                "state.gripper": np.zeros((1, 1)),
                "annotation.human.action.task_description": [observation_dict["task_description"]],
            }
        """

        # get the action chunk via the policy server
        action_chunk = self.call_endpoint("get_action", obs_dict)

        """
            Example of action_chunk for single arm task:
            action_chunk = {
                "action.single_arm": np.zeros((1, 5)),
                "action.gripper": np.zeros((1, 1)),
            }
        """
        concat_action = np.concatenate(
            [action_chunk["action.single_arm"], action_chunk["action.gripper"]],
            axis=1,
        )
        concat_action = convert_lerobot_action_to_leisaac(concat_action)

        return torch.from_numpy(concat_action[:, None, :])


class LeRobotServicePolicyClient(Policy):
    """
    Service policy client for Lerobot: https://github.com/huggingface/lerobot
    Target Commit: https://github.com/huggingface/lerobot/tree/v0.3.3
    """

    def __init__(
        self,
        host: str,
        port: int,
        timeout_ms: int = 5000,
        camera_infos: dict[str, dict] = {},
        task_type: str = "so101leader",
        policy_type: str = "smolvla",
        pretrained_name_or_path: str = "checkpoints/last/pretrained_model",
        actions_per_chunk: int = 50,
        device: str = "cuda",
    ):
        """
        Args:
            host: Host of the policy server.
            port: Port of the policy server.
            timeout_ms: Timeout of the policy server.
            camera_infos: List of camera information. {camera_key: (height, width)}
            task_type: Type of task.
            policy_type: Type of policy.
            pretrained_name_or_path: Path to the pretrained model in the remote policy server.
            actions_per_chunk: Number of actions per chunk.
            device: Device to use.
        """
        super().__init__("service")
        service_address = f"{host}:{port}"
        self.timeout_ms = timeout_ms
        self.task_type = task_type
        self.actions_per_chunk = actions_per_chunk

        lerobot_features = {}
        self.last_action = None
        if task_type == "so101leader":
            lerobot_features["observation.state"] = {
                "dtype": "float32",
                "shape": (6,),
                "names": [f"{joint_name}.pos" for joint_name in SINGLE_ARM_JOINT_NAMES],
            }
            self.last_action = np.zeros((1, 6))
        # TODO: add bi-arm support

        for camera_key, camera_image_shape in camera_infos.items():
            lerobot_features[f"observation.images.{camera_key}"] = {
                "dtype": "image",
                "shape": (camera_image_shape[0], camera_image_shape[1], 3),
                "names": ["height", "width", "channels"],
            }
        self.camera_keys = list(camera_infos.keys())

        self.policy_config = RemotePolicyConfig(
            policy_type,
            pretrained_name_or_path,
            lerobot_features,
            actions_per_chunk,
            device,
        )
        self.channel = grpc.insecure_channel(service_address, grpc_channel_options())
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)

        self.latest_action_step = 0
        self.skip_send_observation = False

        self._init_service()

    def _init_service(self):
        try:
            self.stub.Ready(services_pb2.Empty())

            # send policy instructions
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            print("Sending policy instructions to policy server, it may take a while...")
            self.stub.SendPolicyInstructions(policy_setup)
            print("Policy server is ready.")

        except grpc.RpcError:
            raise RuntimeError("Failed to connect to policy server")

    def _send_observation(self, observation_dict: dict):
        raw_observation = {
            f"{key}": observation_dict[key].cpu().numpy().astype(np.uint8)[0] for key in self.camera_keys
        }
        raw_observation["task"] = observation_dict["task_description"]

        if self.task_type == "so101leader":
            joint_pos = convert_leisaac_action_to_lerobot(observation_dict["joint_pos"])
            for joint_name in SINGLE_ARM_JOINT_NAMES:
                raw_observation[f"{joint_name}.pos"] = joint_pos[0, SINGLE_ARM_JOINT_NAMES.index(joint_name)].item()
        # TODO: add bi-arm support

        """
            Example of raw_observation for single arm task:
            raw_observation = {
                "front": np.zeros((480, 640, 3), dtype=np.uint8),
                "wrist": np.zeros((480, 640, 3), dtype=np.uint8),
                "shoulder_pan.pos": 0.0,
                "shoulder_lift.pos": 0.0,
                "elbow_flex.pos": 0.0,
                "wrist_flex.pos": 0.0,
                "wrist_roll.pos": 0.0,
                "gripper.pos": 0.0,
                "task": "pick_and_place",
            }
        """
        self.latest_action_step += 1
        observation = TimedObservation(
            timestamp=time.time(),
            observation=raw_observation,
            timestep=self.latest_action_step,
        )

        # send observation to policy server
        observation_bytes = pickle.dumps(observation)
        observation_iterator = send_bytes_in_chunks(
            observation_bytes,
            services_pb2.Observation,
            log_prefix="[CLIENT] Observation",
            silent=True,
        )
        _ = self.stub.SendObservations(observation_iterator)

    def _receive_action(self) -> dict:
        actions_chunk = self.stub.GetActions(services_pb2.Empty())
        if len(actions_chunk.data) == 0:
            print("Received `Empty` from policy server, waiting for next call")
            return None
        return pickle.loads(actions_chunk.data)

    def get_action(self, observation_dict: dict) -> torch.Tensor:
        if not self.skip_send_observation:
            self._send_observation(observation_dict)
        action_chunk = self._receive_action()
        if action_chunk is None:
            self.skip_send_observation = True
            return torch.from_numpy(self.last_action).repeat(self.actions_per_chunk, 1)[:, None, :]

        action_list = [action.get_action()[None, :] for action in action_chunk]
        concat_action = torch.cat(action_list, dim=0)
        concat_action = convert_lerobot_action_to_leisaac(concat_action)

        self.last_action = concat_action[-1, :]
        self.skip_send_observation = False

        return torch.from_numpy(concat_action[:, None, :])


class OpenPIServicePolicyClient(WebsocketServicePolicy):
    """
    Service policy client for OpenPI: https://github.com/Physical-Intelligence/openpi
    Target Commit: https://github.com/Physical-Intelligence/openpi/commit/5bff19b0c0c447c7a7eaaaccf03f36d50998ec9d
    Reference: https://github.com/EverNorif/openpi/tree/lerobot-v0.3.3
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        camera_keys: list[str] = ["front", "wrist"],
        task_type: str = "so101leader",
        api_key: str = None,
    ):
        """
        Args:
            host: Host of the policy server.
            port: Port of the policy server.
            camera_keys: Keys of the cameras.
            task_type: Type of task.
            api_key: API key of the policy server.
        """
        super().__init__(host=host, port=port, api_key=api_key)
        self.camera_keys = camera_keys
        self.task_type = task_type

    def get_action(self, observation_dict: dict) -> torch.Tensor:
        obs_dict = {
            f"images/{key}": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(observation_dict[key].cpu().squeeze().numpy(), 224, 224)
            )
            for key in self.camera_keys
        }

        if self.task_type == "so101leader":
            joint_pos = convert_leisaac_action_to_lerobot(observation_dict["joint_pos"])
            obs_dict["state"] = joint_pos.squeeze().astype(np.float64)
        # TODO: add bi-arm support

        obs_dict["prompt"] = observation_dict["task_description"]

        """
            Example of obs_dict for single arm task:
            obs_dict = {
                "images/front": np.zeros((224, 224, 3), dtype=np.uint8),
                "images/wrist": np.zeros((224, 224, 3), dtype=np.uint8),
                "state": np.zeros(6),
                "prompt": observation_dict["task_description"],
            }
        """

        # get the action chunk via the policy server
        action_chunk = self.infer(obs_dict)["actions"]

        """
            Example of action_chunk for single arm task:
            action_chunk: np.zeros((10, 6))
        """
        processed_action = convert_lerobot_action_to_leisaac(action_chunk)

        return torch.from_numpy(processed_action[:, None, :])
