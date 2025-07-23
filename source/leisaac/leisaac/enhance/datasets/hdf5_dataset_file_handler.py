import enum
import copy
import h5py

from concurrent.futures import ThreadPoolExecutor

from isaaclab.utils.datasets import HDF5DatasetFileHandler, EpisodeData


class StreamWriteMode(enum.Enum):
    APPEND = 0  # Append the record
    LAST = 1    # Write the last record


class StreamingHDF5DatasetFileHandler(HDF5DatasetFileHandler):
    def __init__(self):
        """
            compression options:
            - gzip: high compression ratio (50-80%), high latency due to CPU-intensive compression
            - lzf: moderate compression ratio (30-50%), low latency, fast compression algorithm
            - None: don't use compression, will cause minimum latency but largest file size
        """
        super().__init__()
        self._chunks_length = 100
        self._compression = None
        self._writer = self.SingleThreadHDF5DatasetWriter(self)

    class SingleThreadHDF5DatasetWriter:
        def __init__(self, file_handler):
            self.executor = ThreadPoolExecutor(max_workers=1)
            self.file_handler = file_handler

        def write_episode(self, h5_episode_group: h5py.Group, episode: EpisodeData, write_mode: StreamWriteMode):
            episode_copy = copy.deepcopy(episode)
            funture = self.executor.submit(self._do_write_episode, h5_episode_group, episode_copy)
            return funture.result() if write_mode == StreamWriteMode.LAST else funture

        def _do_write_episode(self, h5_episode_group: h5py.Group, episode: EpisodeData):
            def create_dataset_helper(group, key, value):
                """Helper method to create dataset that contains recursive dict objects."""
                if isinstance(value, dict):
                    if key not in group:
                        key_group = group.create_group(key)
                    else:
                        key_group = group[key]
                    for sub_key, sub_value in value.items():
                        create_dataset_helper(key_group, sub_key, sub_value)
                else:
                    data = value.cpu().numpy()
                    if key not in group:
                        dataset = group.create_dataset(
                            key,
                            shape=data.shape,
                            maxshape=(None, *data.shape[1:]),
                            chunks=(self.file_handler.chunks_length, *data.shape[1:]),
                            dtype=data.dtype,
                            compression=self.file_handler.compression,
                        )
                        dataset[0: data.shape[0]] = data
                    else:
                        dataset = group[key]
                        dataset.resize(dataset.shape[0] + data.shape[0], axis=0)
                        dataset[dataset.shape[0] - data.shape[0]:] = data

            for key, value in episode.data.items():
                create_dataset_helper(h5_episode_group, key, value)

            self.file_handler.flush()

        def shutdown(self):
            self.executor.shutdown(wait=True)

    @property
    def chunks_length(self) -> int:
        return self._chunks_length

    @chunks_length.setter
    def chunks_length(self, chunks_length: int):
        self._chunks_length = chunks_length

    @property
    def compression(self) -> str | None:
        return self._compression

    @compression.setter
    def compression(self, compression: str | None):
        self._compression = compression

    def write_episode(self, episode: EpisodeData, write_mode: StreamWriteMode):
        self._raise_if_not_initialized()
        if episode.is_empty():
            return

        group_name = f"demo_{self._demo_count}"
        if group_name not in self._hdf5_data_group:
            h5_episode_group = self._hdf5_data_group.create_group(group_name)
        else:
            h5_episode_group = self._hdf5_data_group[group_name]

        # store number of steps taken
        if "actions" in episode.data:
            if "num_samples" not in h5_episode_group.attrs:
                h5_episode_group.attrs["num_samples"] = 0
            h5_episode_group.attrs["num_samples"] += len(episode.data["actions"])
        else:
            h5_episode_group.attrs["num_samples"] = 0

        if episode.seed is not None:
            h5_episode_group.attrs["seed"] = episode.seed

        if episode.success is not None:
            h5_episode_group.attrs["success"] = episode.success

        self._writer.write_episode(h5_episode_group, episode, write_mode)

        if write_mode == StreamWriteMode.LAST:
            # increment total step counts
            self._hdf5_data_group.attrs["total"] += h5_episode_group.attrs["num_samples"]

            # increment total demo counts
            self._demo_count += 1

    def close(self):
        self._writer.shutdown()
        super().close()
