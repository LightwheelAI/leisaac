"""
Merge a LeRobot-converted HDF5 with a source/template LeIsaac HDF5.

Inputs:
  - lerobot_converted_hdf5_path:
      HDF5 produced from local LeRobot dataset conversion (episode parquet columns stored as datasets)
  - source_template_hdf5_path:
      LeIsaac HDF5 providing initial_state + episode structure template + metadata
Output:
  - output_merged_hdf5_path:
      New replayable HDF5 that follows source structure, but actions/states come from the LeRobot-converted HDF5.

Default behavior:
  - Copy everything from source root EXCEPT '/data' into output root.
  - Choose a template episode from source '/data' (first one, or specified).
  - For each lerobot episode under /data/chunk-*/episode_*:
      - Copy template episode group into output '/data/<output_episode_name>'
      - Overwrite:
          /actions
          /obs/actions (if exists)
          /processed_actions
      - Optionally overwrite a user-specified joint/state dataset path with LeRobot observation.state
"""

import argparse
from contextlib import suppress
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def list_template_episode_names(source_hdf5_file: h5py.File) -> list[str]:
    """List episode group names under source '/data'."""
    if "data" not in source_hdf5_file:
        return []
    return sorted(list(source_hdf5_file["data"].keys()))


def ensure_parent_groups_exist(episode_root_group: h5py.Group, dataset_path: str) -> h5py.Group:
    """
    Ensure all parent groups exist for a dataset path like
    'states/articulation/robot/joint_position'. Returns the parent group.
    """
    path_parts = [part for part in dataset_path.split("/") if part]
    if len(path_parts) <= 1:
        return episode_root_group

    parent_parts = path_parts[:-1]
    current_group = episode_root_group
    for group_name in parent_parts:
        current_group = current_group.require_group(group_name)
    return current_group


def copy_hdf5_attributes(source_object, destination_object) -> None:
    """Copy all HDF5 attributes from source object to destination object."""
    for attribute_key, attribute_value in source_object.attrs.items():
        destination_object.attrs[attribute_key] = attribute_value


def copy_source_root_except_data_group(source_hdf5_file: h5py.File, output_hdf5_file: h5py.File) -> None:
    """
    Copy everything at the root of source HDF5 into output HDF5 except '/data'.
    This preserves metadata, configs, and other root-level structures.
    """
    copy_hdf5_attributes(source_hdf5_file, output_hdf5_file)
    for root_item_name in source_hdf5_file.keys():
        if root_item_name == "data":
            continue
        source_hdf5_file.copy(root_item_name, output_hdf5_file, name=root_item_name)


def copy_template_episode_group(
    source_hdf5_file: h5py.File,
    output_hdf5_file: h5py.File,
    template_episode_name: str,
    output_episode_name: str,
) -> None:
    """
    Copy '/data/<template_episode_name>' from source into output at '/data/<output_episode_name>'.
    """
    output_data_group = output_hdf5_file.require_group("data")

    if output_episode_name in output_data_group:
        del output_data_group[output_episode_name]

    source_hdf5_file["data"].copy(template_episode_name, output_data_group, name=output_episode_name)

    # Preserve attrs on /data itself if present
    copy_hdf5_attributes(source_hdf5_file["data"], output_data_group)


def write_or_overwrite_hdf5_dataset(
    parent_group: h5py.Group,
    dataset_name: str,
    dataset_data: np.ndarray,
    template_dataset: h5py.Dataset | None = None,
) -> None:
    """
    Overwrite dataset 'dataset_name' inside 'parent_group' with numpy 'dataset_data'.
    If 'template_dataset' is provided, mirror its dtype/chunking/compression settings when possible.
    """
    if dataset_name in parent_group:
        del parent_group[dataset_name]

    create_dataset_kwargs = {}

    if template_dataset is not None:
        # Mirror chunking/compression options
        if template_dataset.chunks is not None:
            create_dataset_kwargs["chunks"] = template_dataset.chunks
        if template_dataset.compression is not None:
            create_dataset_kwargs["compression"] = template_dataset.compression
            if template_dataset.compression_opts is not None:
                create_dataset_kwargs["compression_opts"] = template_dataset.compression_opts
        if template_dataset.shuffle:
            create_dataset_kwargs["shuffle"] = True

        # Try to match dtype (if possible)
        with suppress(Exception):
            dataset_data = dataset_data.astype(template_dataset.dtype, copy=False)

    parent_group.create_dataset(dataset_name, data=dataset_data, **create_dataset_kwargs)


def iterate_lerobot_episode_groups(
    lerobot_hdf5_file: h5py.File,
) -> list[tuple[str, h5py.Group]]:
    """
    Iterate episodes from a lerobot-converted HDF5:

    Supports layouts:
      1) /data/chunk-000/episode_000000 (nested)
      2) /data/episode_000000 (flat)

    Returns list of:
      (episode_identifier_string, episode_group)
    where episode_identifier_string is like 'chunk-000/episode_000000' or 'episode_000000'.
    """
    episodes: list[tuple[str, h5py.Group]] = []

    if "data" not in lerobot_hdf5_file:
        return episodes

    for key in sorted(lerobot_hdf5_file["data"].keys()):
        item = lerobot_hdf5_file["data"][key]

        # Case 1: Flat structure /data/episode_XXXXXX
        if key.startswith("episode_") and isinstance(item, h5py.Group):
            episodes.append((key, item))
            continue

        # Case 2: Nested structure /data/chunk-XXX/episode_XXXXXX
        # We assume any other group might contain episodes (e.g. chunk-000)
        if isinstance(item, h5py.Group):
            for sub_key in sorted(item.keys()):
                if sub_key.startswith("episode_"):
                    sub_item = item[sub_key]
                    if isinstance(sub_item, h5py.Group):
                        episodes.append((f"{key}/{sub_key}", sub_item))

    return episodes


def read_action_and_state_arrays_from_lerobot_group(
    lerobot_episode_group: h5py.Group,
    lerobot_action_dataset_key: str,
    lerobot_state_dataset_key: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Read action and state arrays from a lerobot episode group.
    Supports common naming variants for 'observation.state' due to prior sanitization.

    Returns:
      actions_array: np.ndarray
      state_array: Optional[np.ndarray]  (None if not found)
    """
    if lerobot_action_dataset_key not in lerobot_episode_group:
        raise KeyError(
            f"Missing action dataset '{lerobot_action_dataset_key}' "
            f"in lerobot episode group: {lerobot_episode_group.name}"
        )

    actions_array = np.array(lerobot_episode_group[lerobot_action_dataset_key])

    state_candidate_keys = [lerobot_state_dataset_key]

    # Accept common sanitized variants between 'observation.state' and 'observation_state'
    if lerobot_state_dataset_key == "observation.state":
        state_candidate_keys.append("observation_state")
    elif lerobot_state_dataset_key == "observation_state":
        state_candidate_keys.append("observation.state")

    state_array = None
    for state_key in state_candidate_keys:
        if state_key in lerobot_episode_group:
            state_array = np.array(lerobot_episode_group[state_key])
            break

    return actions_array, state_array


def merge_lerobot_converted_hdf5_with_source_template(
    lerobot_converted_hdf5_path: str,
    source_template_hdf5_path: str,
    output_merged_hdf5_path: str,
    source_template_episode_name: str | None,
    output_episode_name_prefix: str,
    lerobot_action_dataset_key: str,
    lerobot_state_dataset_key: str,
    lerobot_timestamp_dataset_key: str | None,
    write_state_to_output_hdf5: bool,
    output_state_target_dataset_path: str | None,
    skip_first_n_lerobot_episodes: int,
    max_number_of_episodes_to_merge: int | None,
) -> None:
    lerobot_converted_hdf5_path = str(Path(lerobot_converted_hdf5_path).expanduser().resolve())
    source_template_hdf5_path = str(Path(source_template_hdf5_path).expanduser().resolve())
    output_merged_hdf5_path = str(Path(output_merged_hdf5_path).expanduser().resolve())
    Path(output_merged_hdf5_path).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(source_template_hdf5_path, "r") as source_hdf5_file, h5py.File(
        lerobot_converted_hdf5_path, "r"
    ) as lerobot_hdf5_file, h5py.File(output_merged_hdf5_path, "w") as output_hdf5_file:

        # 1) Copy root-level template/metadata except '/data'
        copy_source_root_except_data_group(source_hdf5_file, output_hdf5_file)

        # 2) Choose a template episode in source '/data'
        available_template_episode_names = list_template_episode_names(source_hdf5_file)
        if not available_template_episode_names:
            raise RuntimeError("source_template_hdf5 has no '/data/<episode>' groups to use as template.")

        if source_template_episode_name is None:
            source_template_episode_name = available_template_episode_names[0]
        else:
            if source_template_episode_name not in source_hdf5_file["data"]:
                preview = available_template_episode_names[:10]
                raise RuntimeError(
                    f"template_episode '{source_template_episode_name}' not found in source '/data'. "
                    f"Available examples: {preview}{'...' if len(available_template_episode_names) > 10 else ''}"
                )

        # 3) Iterate lerobot episodes
        lerobot_episode_list = iterate_lerobot_episode_groups(lerobot_hdf5_file)
        if not lerobot_episode_list:
            raise RuntimeError(
                "lerobot_converted_hdf5 has no episodes under '/data/chunk-*/episode_*' or '/data/episode_*'."
            )

        # Apply slicing
        lerobot_episode_list = lerobot_episode_list[skip_first_n_lerobot_episodes:]
        if max_number_of_episodes_to_merge is not None:
            lerobot_episode_list = lerobot_episode_list[:max_number_of_episodes_to_merge]

        # Ensure /data exists in output
        output_hdf5_file.require_group("data")

        for output_episode_index, (lerobot_episode_identifier, lerobot_episode_group) in enumerate(
            tqdm(lerobot_episode_list, desc="Merging episodes")
        ):
            output_episode_name = f"{output_episode_name_prefix}{output_episode_index}"

            # 3.1 Copy template episode group to output
            copy_template_episode_group(
                source_hdf5_file=source_hdf5_file,
                output_hdf5_file=output_hdf5_file,
                template_episode_name=source_template_episode_name,
                output_episode_name=output_episode_name,
            )
            output_episode_group = output_hdf5_file["data"][output_episode_name]

            # 3.2 Read lerobot action/state arrays
            actions_array, state_array = read_action_and_state_arrays_from_lerobot_group(
                lerobot_episode_group=lerobot_episode_group,
                lerobot_action_dataset_key=lerobot_action_dataset_key,
                lerobot_state_dataset_key=lerobot_state_dataset_key,
            )

            # 3.3 Overwrite /actions
            template_actions_dataset = output_episode_group["actions"] if "actions" in output_episode_group else None
            if isinstance(template_actions_dataset, h5py.Group):
                template_actions_dataset = None

            write_or_overwrite_hdf5_dataset(
                parent_group=output_episode_group,
                dataset_name="actions",
                dataset_data=actions_array,
                template_dataset=(
                    template_actions_dataset if isinstance(template_actions_dataset, h5py.Dataset) else None
                ),
            )

            # 3.4 Overwrite /obs/actions if exists (or create)
            if "obs" in output_episode_group and isinstance(output_episode_group["obs"], h5py.Group):
                output_obs_group = output_episode_group["obs"]
            else:
                output_obs_group = output_episode_group.require_group("obs")

            template_obs_actions_dataset = output_obs_group["actions"] if "actions" in output_obs_group else None
            write_or_overwrite_hdf5_dataset(
                parent_group=output_obs_group,
                dataset_name="actions",
                dataset_data=actions_array,
                template_dataset=(
                    template_obs_actions_dataset if isinstance(template_obs_actions_dataset, h5py.Dataset) else None
                ),
            )

            # 3.5 Overwrite /processed_actions (create if missing)
            template_processed_actions_dataset = (
                output_episode_group["processed_actions"]
                if "processed_actions" in output_episode_group
                and isinstance(output_episode_group["processed_actions"], h5py.Dataset)
                else None
            )
            write_or_overwrite_hdf5_dataset(
                parent_group=output_episode_group,
                dataset_name="processed_actions",
                dataset_data=actions_array,
                template_dataset=template_processed_actions_dataset,
            )

            # 3.6 Optional: overwrite timestamps
            if lerobot_timestamp_dataset_key is not None and lerobot_timestamp_dataset_key in lerobot_episode_group:
                timestamps_array = np.array(lerobot_episode_group[lerobot_timestamp_dataset_key])
                template_timestamps_dataset = (
                    output_episode_group["timestamps"]
                    if "timestamps" in output_episode_group
                    and isinstance(output_episode_group["timestamps"], h5py.Dataset)
                    else None
                )
                write_or_overwrite_hdf5_dataset(
                    parent_group=output_episode_group,
                    dataset_name="timestamps",
                    dataset_data=timestamps_array,
                    template_dataset=template_timestamps_dataset,
                )

            # 3.7 Optional: write state/joint positions into a target dataset path
            if write_state_to_output_hdf5 and state_array is not None and output_state_target_dataset_path is not None:
                parent_group = ensure_parent_groups_exist(output_episode_group, output_state_target_dataset_path)
                target_dataset_name = output_state_target_dataset_path.split("/")[-1]

                # Mirror template dataset properties if it exists
                template_state_dataset = None
                if output_state_target_dataset_path in output_episode_group and isinstance(
                    output_episode_group[output_state_target_dataset_path], h5py.Dataset
                ):
                    template_state_dataset = output_episode_group[output_state_target_dataset_path]

                write_or_overwrite_hdf5_dataset(
                    parent_group=parent_group,
                    dataset_name=target_dataset_name,
                    dataset_data=state_array,
                    template_dataset=template_state_dataset,
                )

            # Provenance
            output_episode_group.attrs["lerobot_episode_identifier"] = lerobot_episode_identifier
            output_episode_group.attrs["source_template_episode_name"] = source_template_episode_name

        # Root provenance
        output_hdf5_file.attrs["merged_from_lerobot_converted_hdf5"] = lerobot_converted_hdf5_path
        output_hdf5_file.attrs["merged_from_source_template_hdf5"] = source_template_hdf5_path

    print(f"[OK] Wrote merged replayable HDF5: {output_merged_hdf5_path}")


def main():
    argument_parser = argparse.ArgumentParser(
        description="Merge lerobot-converted HDF5 with source IsaacLab/LeIsaac template HDF5."
    )

    argument_parser.add_argument(
        "--lerobot_hdf5",
        type=str,
        required=True,
        help="Path to HDF5 converted from local LeRobot dataset",
    )
    argument_parser.add_argument(
        "--source_hdf5",
        type=str,
        required=True,
        help="Path to source/template IsaacLab/LeIsaac HDF5 (provides initial_state + structure template)",
    )
    argument_parser.add_argument(
        "--output_hdf5",
        type=str,
        required=True,
        help="Path to output merged HDF5",
    )

    argument_parser.add_argument(
        "--template_episode",
        type=str,
        default=None,
        help="Episode name under source '/data' to use as template (default: first episode found)",
    )

    argument_parser.add_argument(
        "--output_episode_prefix",
        type=str,
        default="demo_",
        help="Output episode name prefix under '/data' (default: demo)",
    )

    # Keys in lerobot episode group
    argument_parser.add_argument(
        "--lerobot_action_key",
        type=str,
        default="action",
        help="Dataset key inside lerobot episode group for actions (default: action)",
    )
    argument_parser.add_argument(
        "--lerobot_state_key",
        type=str,
        default="observation.state",
        help=(
            "Dataset key inside lerobot episode group for state/joint pos (default: observation.state; also tries"
            " observation_state)"
        ),
    )
    argument_parser.add_argument(
        "--lerobot_timestamp_key",
        type=str,
        default=None,
        help="Optional dataset key inside lerobot episode group for timestamps (e.g. timestamp)",
    )

    # Where to write state in output episode
    argument_parser.add_argument(
        "--write_state",
        action="store_true",
        help="If set, write lerobot state/joint positions into output at --output_state_target_path",
    )
    argument_parser.add_argument(
        "--output_state_target_path",
        type=str,
        default=None,
        help=(
            "Target dataset path inside each output episode to store state "
            "(e.g. 'obs/joint_pos' or 'states/articulation/robot/joint_position')"
        ),
    )

    # Episode slicing
    argument_parser.add_argument(
        "--skip_first_n_episodes",
        type=int,
        default=0,
        help="Skip first N lerobot episodes (default: 0)",
    )
    argument_parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Merge at most N episodes (default: all)",
    )

    parsed_arguments = argument_parser.parse_args()

    if parsed_arguments.write_state and not parsed_arguments.output_state_target_path:
        raise SystemExit("--write_state requires --output_state_target_path to be set")

    merge_lerobot_converted_hdf5_with_source_template(
        lerobot_converted_hdf5_path=parsed_arguments.lerobot_hdf5,
        source_template_hdf5_path=parsed_arguments.source_hdf5,
        output_merged_hdf5_path=parsed_arguments.output_hdf5,
        source_template_episode_name=parsed_arguments.template_episode,
        output_episode_name_prefix=parsed_arguments.output_episode_prefix,
        lerobot_action_dataset_key=parsed_arguments.lerobot_action_key,
        lerobot_state_dataset_key=parsed_arguments.lerobot_state_key,
        lerobot_timestamp_dataset_key=parsed_arguments.lerobot_timestamp_key,
        write_state_to_output_hdf5=parsed_arguments.write_state,
        output_state_target_dataset_path=parsed_arguments.output_state_target_path,
        skip_first_n_lerobot_episodes=parsed_arguments.skip_first_n_episodes,
        max_number_of_episodes_to_merge=parsed_arguments.max_episodes,
    )


if __name__ == "__main__":
    main()
