import os
import enum
import torch

from typing import Sequence
from isaaclab.managers import RecorderManager, DatasetExportMode
from isaaclab.envs import ManagerBasedEnv

from ..datasets import StreamingHDF5DatasetFileHandler, StreamWriteMode


class EnhanceDatasetExportMode(enum.IntEnum):
    """Enhanced dataset export modes that include additional options beyond the base DatasetExportMode."""
    EXPORT_ALL_RESUME = 0  # Export all episodes to a single dataset file and resume recording
    # NOTE: the exact value is same as DatasetExportMode.EXPORT_NONE, we don't support DatasetExportMode.EXPORT_NONE when recording


class StreamingRecorderManager(RecorderManager):
    def __init__(self, cfg: object, env: ManagerBasedEnv) -> None:
        # use streaming_hdf5_dataset_file_handler
        cfg.dataset_file_handler_class_type = StreamingHDF5DatasetFileHandler

        super().__init__(cfg, env)

        assert cfg.dataset_export_mode in [DatasetExportMode.EXPORT_ALL, EnhanceDatasetExportMode.EXPORT_ALL_RESUME], "only support EXPORT_ALL|EXPORT_ALL_RESUME"
        if cfg.dataset_export_mode == EnhanceDatasetExportMode.EXPORT_ALL_RESUME:
            # only process EXPORT_ALL_RESUME mode here, other modes are processed in the super class
            self._dataset_file_handler = cfg.dataset_file_handler_class_type()
            self._dataset_file_handler.create(
                os.path.join(cfg.dataset_export_dir_path, cfg.dataset_filename),
                resume=True
            )

        self._env_steps_record = torch.zeros(self._env.num_envs)
        self._flush_steps = 100
        self._compression = None
        if self._dataset_file_handler is not None:
            self._dataset_file_handler.chunks_length = self._flush_steps
            self._dataset_file_handler.compression = self._compression

    @property
    def flush_steps(self) -> int:
        return self._flush_steps

    @flush_steps.setter
    def flush_steps(self, flush_steps: int) -> None:
        self._flush_steps = flush_steps
        if self._dataset_file_handler is not None:
            self._dataset_file_handler.chunks_length = self._flush_steps

    @property
    def compression(self) -> str | None:
        return self._compression

    @compression.setter
    def compression(self, compression: str | None):
        self._compression = compression
        if self._dataset_file_handler is not None:
            self._dataset_file_handler.compression = self._compression

    def __str__(self) -> str:
        msg = "[Enhanced] StreamingRecorderManager. \n"
        msg += super().__str__()
        return msg

    def record_pre_step(self) -> None:
        self._env_steps_record += 1
        super().record_pre_step()
        self.export_episodes(from_step=True)

    def export_episodes(self, env_ids: Sequence[int] | None = None, from_step: bool = False) -> None:
        if len(self.active_terms) == 0:
            return

        if env_ids is None:
            env_ids = list(range(self._env.num_envs))
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()

        # Export episode data through dataset exporter
        for env_id in env_ids:
            if env_id in self._episodes and not self._episodes[env_id].is_empty() and (self._env_steps_record[env_id] >= self._flush_steps or not from_step):
                if self._env.cfg.seed is not None:
                    self._episodes[env_id].seed = self._env.cfg.seed
                episode_succeeded = self._episodes[env_id].success
                target_dataset_file_handler = None
                if self.cfg.dataset_export_mode == DatasetExportMode.EXPORT_ALL or self.cfg.dataset_export_mode == EnhanceDatasetExportMode.EXPORT_ALL_RESUME:
                    target_dataset_file_handler = self._dataset_file_handler
                if target_dataset_file_handler is not None:
                    write_mode = StreamWriteMode.APPEND if from_step else StreamWriteMode.LAST
                    target_dataset_file_handler.write_episode(self._episodes[env_id], write_mode)
                    self._clear_episode_cache([env_id])
                if episode_succeeded:
                    self._exported_successful_episode_count[env_id] = (
                        self._exported_successful_episode_count.get(env_id, 0) + 1
                    )
                else:
                    self._exported_failed_episode_count[env_id] = self._exported_failed_episode_count.get(env_id, 0) + 1

    def _clear_episode_cache(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = list(range(self._env.num_envs))
        for env_id in env_ids:
            del self._episodes[env_id]._data
            self._episodes[env_id].data = dict()
            self._env_steps_record[env_id] = 0
