# Dataset Replay

After teleoperation, you can replay the collected dataset in the simulation environment using the following script:

```shell
python scripts/environments/teleoperation/replay.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --num_envs=1 \
    --device=cuda \
    --enable_cameras \
    --replay_mode=action \
    --dataset_file=./datasets/dataset.hdf5 \
    --select_episodes 1 2
```

<details>
<summary><strong>Parameter descriptions for replay.py</strong></summary><p></p>

- `--task`: Specify the task environment name to run, e.g., `LeIsaac-SO101-PickOrange-v0`.

- `--num_envs`: Set the number of parallel simulation environments, usually `1` for replay.

- `--device`: Specify the computation device, such as `cpu` or `cuda` (GPU).

- `--enable_cameras`: Enable camera sensors to visualize when replay.

- `--replay_mode`: Replay mode, we support replay `action` or `state`.

- `--task_type`: Specify task type. If your dataset is recorded with keyboard, you should set it to `keyboard`, otherwise not to set it and keep default value None.

- `--dataset_file`: Path to the recorded dataset, e.g., `./datasets/record_data.hdf5`.

- `--select_episodes`: A list of episode indices to replayed, Keep empty to replay all episodes.

</details>

:::tip
NOTE: If you record datasets with devices other than the Leader (for example, the keyboard or gamepad), set `task_type` to the corresponding device, such as `--task_type=keyboard` or `--task_type=gamepad`.
:::
