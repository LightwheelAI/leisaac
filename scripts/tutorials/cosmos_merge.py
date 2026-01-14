"""
Merge LeRobot-converted actions into a source IsaacLab HDF5 template.

Inputs:
  - lerobot_hdf5: HDF5 containing actions (from lerobot2isaaclab.py)
  - source_hdf5: Template HDF5 (LeIsaac format)
Output:
  - output_hdf5: Merged HDF5 with source structure + LeRobot actions.
"""

import argparse
from pathlib import Path

import h5py
from tqdm import tqdm


def overwrite_dataset(group, name, data, compression="gzip"):
    if name in group:
        del group[name]
    group.create_dataset(name, data=data, compression=compression)


def main():
    parser = argparse.ArgumentParser(description="Merge LeRobot actions into IsaacLab HDF5 template")
    parser.add_argument("--lerobot_hdf5", required=True, help="Path to LeRobot action HDF5")
    parser.add_argument("--source_hdf5", required=True, help="Path to source template HDF5")
    parser.add_argument("--output_hdf5", required=True, help="Path to output merged HDF5")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index for output episodes")
    args = parser.parse_args()

    lerobot_path = Path(args.lerobot_hdf5).resolve()
    source_path = Path(args.source_hdf5).resolve()
    output_path = Path(args.output_hdf5).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Merging actions from {lerobot_path.name} into template {source_path.name}...")

    with h5py.File(source_path, "r") as src, h5py.File(lerobot_path, "r") as lerobot, h5py.File(
        output_path, "w"
    ) as out:

        # 1. Copy root metadata and structure (non-data)
        for key in src.attrs:
            out.attrs[key] = src.attrs[key]

        for key in src.keys():
            if key != "data":
                src.copy(key, out)

        # 2. Get template episode
        if "data" not in src or len(src["data"]) == 0:
            raise ValueError(f"No data/episodes found in source template: {source_path}")

        # Use simple alphabetical first episode as template
        template_name = sorted(src["data"].keys())[0]
        print(f"Using '{template_name}' as template episode.")

        # 3. Process episodes
        out_data = out.create_group("data")
        lerobot_data = lerobot.get("data")

        if not lerobot_data:
            print("No data group in LeRobot HDF5.")
            return

        # Sort input episodes to ensure order
        input_ep_names = sorted(lerobot_data.keys())

        for i, in_ep_name in enumerate(tqdm(input_ep_names, desc="Merging")):
            # Output naming convention: demo_{i}
            out_ep_name = f"demo_{args.start_idx + i}"

            # Copy template to output
            src["data"].copy(template_name, out_data, name=out_ep_name)
            out_ep = out_data[out_ep_name]

            # Get action data from LeRobot HDF5
            # Expecting: data/<episode>/action
            action = lerobot_data[in_ep_name]["action"][:]

            # Overwrite specific datasets
            overwrite_dataset(out_ep, "actions", action)
            overwrite_dataset(out_ep, "processed_actions", action)

            # Overwrite /obs/actions only if it exists in template
            if "obs" in out_ep and "actions" in out_ep["obs"]:
                overwrite_dataset(out_ep["obs"], "actions", action)

            # Update length metadata
            out_ep.attrs["num_samples"] = len(action)
            out_ep.attrs["num_frames"] = len(action)  # Update both if present (common convention)

    print(f"[OK] Wrote to {output_path}")


if __name__ == "__main__":
    main()
