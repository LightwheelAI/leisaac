import copy

from isaaclab.utils import configclass
from isaaclab.utils.datasets.dataset_file_handler_base import DatasetFileHandlerBase
from isaaclab.utils.datasets.episode_data import EpisodeData
from lerobot.datasets.lerobot_dataset import LeRobotDataset


@configclass
class LeRobotDatasetCfg:
    """Configuration for the LeRobotDataset."""

    repo_id: str = None
    """Lerobot Dataset repository ID."""
    fps: int = 30
    """Lerobot Dataset frames per second."""
    robot_type: str = "so101_follower"
    """Robot type: so101_follower or bi_so101_follower, etc."""
    features: dict = None
    """Features for the LeRobotDataset."""
    action_align: bool = False
    """Whether the action shape equals to the joint number. If action align, we will convert action to lerobot limit range."""


class LeRobotDatasetHandler(DatasetFileHandlerBase):
    def __init__(self, cfg: LeRobotDatasetCfg):
        self._cfg = copy.deepcopy(cfg)
        self._lerobot_dataset = None
        self._demo_count = 0
        self._env_args = {}

    def create(self, file_path: str, env_name: str = None, resume: bool = False):
        if resume:
            self._lerobot_dataset = LeRobotDataset(
                repo_id=self._cfg.repo_id,
            )
        else:
            self._lerobot_dataset = LeRobotDataset.create(
                repo_id=self._cfg.repo_id,
                fps=self._cfg.fps,
                robot_type=self._cfg.robot_type,
                features=self._cfg.features,
            )
        self._env_args["env_name"] = env_name

    def open(self, file_path: str, mode: str = "r"):
        self._lerobot_dataset = LeRobotDataset(
            repo_id=self._cfg.repo_id,
        )

    def get_env_name(self) -> str | None:
        return self._env_args["env_name"]

    def add_frame(self, frame: dict):
        self._lerobot_dataset.add_frame(frame=frame)

    def flush(self):
        self._lerobot_dataset.save_episode(parallel_encoding=False)

    def clear(self):
        self._lerobot_dataset.clear_episode_buffer()

    def finalize(self):
        self._lerobot_dataset.finalize()

    def close(self):
        if self._lerobot_dataset is not None:
            self.finalize()
            self._lerobot_dataset = None

    # not used for now
    def write_episode(self, episode: EpisodeData):
        raise NotImplementedError("write_episode is not supported for LeRobotDatasetHandler")

    def load_episode(self, episode_name: str) -> EpisodeData | None:
        raise NotImplementedError("load_episode is not supported for LeRobotDatasetHandler")

    def get_num_episodes(self) -> int:
        raise NotImplementedError("get_num_episodes is not supported for LeRobotDatasetHandler")
