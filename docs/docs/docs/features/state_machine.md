# State Machine Data Generation

The state machine module provides automated data collection for manipulation tasks without human teleoperation. It runs a scripted policy and records demonstrations to HDF5 datasets.

## Recording

```shell
python scripts/datagen/state_machine/generate.py \
    --task LeIsaac-SO101-PickOrange-v0 \
    --num_envs 1 \
    --device cuda \
    --enable_cameras \
    --record \
    --dataset_file ./datasets/pick_orange.hdf5 \
    --num_demos 50
```

<details>
<summary><strong>Parameter descriptions for generate.py</strong></summary>

- `--task`: Task environment name to run, e.g., `LeIsaac-SO101-PickOrange-v0`. See [here](/resources/available_env) for available tasks.

- `--num_envs`: Number of parallel simulation environments, usually `1`.

- `--device`: Computation device, such as `cpu` or `cuda` (GPU).

- `--enable_cameras`: Enable camera sensors to collect visual data.

- `--seed`: Seed for the environment. Defaults to current timestamp if not set.

- `--record`: Enable data recording; saves demonstrations to an HDF5 file.

- `--dataset_file`: Path to save the recorded dataset, e.g., `./datasets/pick_orange.hdf5`.

- `--resume`: Resume recording from an existing dataset file.

- `--num_demos`: Number of successful demonstrations to record. Set to `0` for infinite.

- `--step_hz`: Environment stepping rate in Hz (default: `60`).

- `--quality`: Enable quality render mode.

- `--use_lerobot_recorder`: Record directly in LeRobot Dataset format instead of HDF5.

- `--lerobot_dataset_repo_id`: HuggingFace dataset repository ID (format: `username/repository_name`). Required when `--use_lerobot_recorder` is set.

- `--lerobot_dataset_fps`: Dataset frame rate when using LeRobot recorder (default: `30`).

</details>

::::tip
Grasp success rate depends heavily on orange spawn positions. Adjusting the spawn positions in the task's environment config file (e.g. moving oranges closer to the robot base) can significantly improve success rate.
::::

## Replay

After recording, you can replay the collected demonstrations in simulation:

```shell
python scripts/datagen/state_machine/replay.py \
    --task LeIsaac-SO101-PickOrange-v0 \
    --dataset_file ./datasets/pick_orange.hdf5 \
    --task_type ik_so101leader \
    --select_episodes 0 \
    --device cuda \
    --enable_cameras \
    --replay_mode action
```

<details>
<summary><strong>Parameter descriptions for replay.py</strong></summary>

- `--task`: Task environment name to run, e.g., `LeIsaac-SO101-PickOrange-v0`.

- `--num_envs`: Number of parallel simulation environments, usually `1`.

- `--device`: Computation device, such as `cpu` or `cuda` (GPU).

- `--enable_cameras`: Enable camera sensors to visualize during replay.

- `--dataset_file`: Path to the recorded dataset, e.g., `./datasets/pick_orange.hdf5`.

- `--replay_mode`: Replay mode — `action` replays IK pose targets, `state` replays joint positions.

- `--task_type`: State machine device type used during recording, e.g., `ik_so101leader` or `bi_ik_so101leader`. Inferred from task name if not set.

- `--select_episodes`: List of episode indices to replay. Leave empty to replay all episodes.

- `--step_hz`: Environment stepping rate in Hz (default: `60`).

</details>

## Adding a New Task

1. Implement a `StateMachineBase` subclass in `source/leisaac/leisaac/datagen/state_machine/`.
2. Register it in `TASK_REGISTRY` inside `scripts/datagen/state_machine/generate.py`:

```python
TASK_REGISTRY = {
    "LeIsaac-SO101-PickOrange-v0": (PickOrangeStateMachine, "ik_so101leader"),
    "LeIsaac-MY-NewTask-v0":       (MyNewStateMachine,      "ik_so101leader"),
}
```
