import enum
import copy
import h5py
import os

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

    def create(self, file_path: str, env_name: str = None, resume: bool = False):
        """Create a new dataset file."""
        if self._hdf5_file_stream is not None:
            raise RuntimeError("HDF5 dataset file stream is already in use")
        if not file_path.endswith(".hdf5"):
            file_path += ".hdf5"
        dir_path = os.path.dirname(file_path)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        if resume:
            self._hdf5_file_stream = h5py.File(file_path, "a")
            self._hdf5_data_group = self._hdf5_file_stream["data"]
            self._demo_count = len(self._hdf5_data_group)
        else:
            self._hdf5_file_stream = h5py.File(file_path, "w")
            # set up a data group in the file
            self._hdf5_data_group = self._hdf5_file_stream.create_group("data")
            self._hdf5_data_group.attrs["total"] = 0
            self._demo_count = 0

            env_name = env_name if env_name is not None else ""
            self.add_env_args({"env_name": env_name, "type": 2})

    class SingleThreadHDF5DatasetWriter:
        def __init__(self, file_handler):
            self.executor = ThreadPoolExecutor(max_workers=1)
            self.file_handler = file_handler

        def write_episode(self, h5_episode_group: h5py.Group, episode: EpisodeData, write_mode: StreamWriteMode):
            episode_copy = copy.deepcopy(episode)
            funture = self.executor.submit(self._do_write_episode, h5_episode_group, episode_copy)
            return funture.result() if write_mode == StreamWriteMode.LAST else funture

        def _do_write_episode(self, h5_episode_group: h5py.Group, episode: EpisodeData):
            """
            Helper that:
            - converts torch tensors (even on GPU) to numpy on CPU
            - converts lists/tuples to numpy arrays
            - handles scalars / 0-dim values
            - appends to an existing dataset or creates a resizable dataset
            """

            import numpy as np
            import torch

            def to_numpy(value):
                """Convert value to a NumPy array on CPU in a safe way."""
                # Torch tensor -> move to CPU then numpy
                if isinstance(value, torch.Tensor):
                    return value.detach().cpu().numpy()
                # Some tensor-like objects (have .cpu()) — try that path
                if hasattr(value, "cpu") and callable(getattr(value, "cpu")):
                    try:
                        return value.cpu().detach().numpy()
                    except Exception:
                        # fallback to numpy() if present
                        if hasattr(value, "numpy"):
                            return value.numpy()
                # If it's already a numpy array-like
                if isinstance(value, np.ndarray):
                    return value
                # If it's list/tuple -> convert
                if isinstance(value, (list, tuple)):
                    try:
                        return np.array(value)
                    except Exception:
                        # last resort: wrap into object array
                        return np.array(value, dtype=object)
                # If it exposes numpy()
                if hasattr(value, "numpy") and callable(getattr(value, "numpy")):
                    try:
                        return value.numpy()
                    except Exception:
                        pass
                # fallback scalar -> numpy
                return np.array(value)

            def ensure_time_axis(arr: np.ndarray):
                """
                Ensure arr has a time (first) axis so that append logic (along axis 0) works.
                If arr is scalar (0-dim) -> convert to shape (1,)
                If arr is 1-dim -> treat length as time dimension already
                """
                if arr.ndim == 0:
                    return arr.reshape(1)
                return arr

            def create_or_append_dataset(group: h5py.Group, key: str, data: np.ndarray):
                """Create a resizable dataset if missing; otherwise append along axis 0."""
                data = ensure_time_axis(data)
                time_len = data.shape[0]
                # dtype conversion - h5py expects numpy dtype
                dtype = data.dtype

                # Build chunk shape sensibly
                if data.ndim == 1:
                    chunk_shape = (min(self.file_handler.chunks_length, max(1, data.shape[0])),)
                else:
                    chunk_shape = (min(self.file_handler.chunks_length, max(1, data.shape[0])), *data.shape[1:])

                # If dataset does not exist -> create with maxshape (None, ...)
                if key not in group:
                    maxshape = (None, *data.shape[1:]) if data.ndim > 1 else (None,)
                    shape = data.shape
                    try:
                        ds = group.create_dataset(
                            key,
                            shape=shape,
                            maxshape=maxshape,
                            chunks=chunk_shape,
                            dtype=dtype,
                            compression=self.file_handler.compression,
                        )
                        # write initial data
                        ds[0: time_len] = data
                    except Exception as e:
                        # Some types (object dtype) cannot be stored — warn and skip
                        print(f"[WARN] Failed to create dataset '{key}' (dtype={dtype}, shape={shape}): {e}")
                        return
                else:
                    ds = group[key]
                    # Quick compatibility check: rank (ndim) must match (except first dim)
                    existing_shape = ds.shape
                    if len(existing_shape) != data.ndim:
                        print(f"[WARN] Incompatible dataset rank for key '{key}': existing {existing_shape}, new {data.shape}. Skipping append.")
                        return
                    if data.ndim > 1 and existing_shape[1:] != data.shape[1:]:
                        print(f"[WARN] Incompatible inner shape for key '{key}': existing {existing_shape[1:]}, new {data.shape[1:]}. Skipping append.")
                        return
                    # Append along axis 0
                    try:
                        old_len = ds.shape[0]
                        ds.resize(old_len + time_len, axis=0)
                        ds[old_len: old_len + time_len] = data
                    except Exception as e:
                        print(f"[WARN] Failed to append to dataset '{key}': {e}")
                        return

            def create_dataset_helper(group, key, value):
                """Recursive helper to create dataset(s) for dict-like episode data."""
                # If value is a dict -> recurse into group
                if isinstance(value, dict):
                    if key not in group:
                        key_group = group.create_group(key)
                    else:
                        key_group = group[key]
                    for sub_key, sub_value in value.items():
                        create_dataset_helper(key_group, sub_key, sub_value)
                else:
                    # First convert to numpy safely
                    try:
                        np_value = to_numpy(value)
                    except Exception as e:
                        print(f"[WARN] Could not convert key '{key}' to numpy: {e}")
                        return

                    # If numpy object dtype or zero-size, still attempt best-effort write
                    try:
                        create_or_append_dataset(group, key, np_value)
                    except Exception as e:
                        print(f"[WARN] Error writing key '{key}': {e}")

            # iterate over keys in episode
            for key, value in episode.data.items():
                create_dataset_helper(h5_episode_group, key, value)

            # flush to disk
            try:
                self.file_handler.flush()
            except Exception:
                # best-effort: ignore flush failures to avoid crashing recorder
                pass

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

        if write_mode == StreamWriteMode.LAST:
            # increment total step counts
            self._hdf5_data_group.attrs["total"] += h5_episode_group.attrs["num_samples"]

            # increment total demo counts
            self._demo_count += 1

        self._writer.write_episode(h5_episode_group, episode, write_mode)

    def close(self):
        self._writer.shutdown()
        super().close()
