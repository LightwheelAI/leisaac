# LeRobot Recorder
> LeRobot Dataset Recorder: Record LeRobot-format datasets directly during teleoperation

LeRobot Recorder is an enhanced feature provided by LeIsaac that allows users to record data directly in LeRobot format during teleoperation. This feature seamlessly integrates into the teleoperation workflow, generating LeRobot-standard datasets without requiring additional data conversion steps.

## How It Works

LeRobot Recorder replaces the default recorder manager with `LeRobotRecorderManager`. After each environment step:

1. **Data Collection**: Collects observations (including joint positions, camera images, etc.) and actions from the current step
2. **Format Conversion**: Converts data to LeRobot frame format via the `build_lerobot_frame` method
3. **Buffer Management**: Adds frame data to the LeRobot Dataset buffer
4. **Episode Processing**: When an episode ends, decides whether to save based on task success status:
   - **Success**: Calls `flush()` to save the entire episode to the dataset
   - **Failure**: Calls `clear()` to clear the buffer without saving data

LeIsaac automatically skips the first 5 frames of each episode to avoid instability from initial states affecting data quality.

## Usage

LeRobot Recorder requires the lerobot dependency. For installation instructions, refer to this [section](../getting_started/installation#optional-install-lerobot).

To record data in LeRobot format during teleoperation, use the following command:

```shell
python scripts/environments/teleoperation/teleop_se3_agent.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --teleop_device=so101leader \
    --port=/dev/ttyACM0 \
    --num_envs=1 \
    --device=cuda \
    --enable_cameras \
    --record \
    --use_lerobot_recorder \
    --lerobot_dataset_repo_id=EverNorif/test_lerobot_recorder \
    --lerobot_dataset_fps=30
```

Simply enable `--use_lerobot_recorder` and specify the `repo_id` and `dataset_fps` to record data directly in LeRobot Dataset format during teleoperation.

## Parameters

- `--use_lerobot_recorder`: Enables the LeRobot format recorder
- `--lerobot_dataset_repo_id`: HuggingFace dataset repository ID (format: `username/repository_name`)
- `--lerobot_dataset_fps`: Dataset frame rate, typically set to 30 FPS

::::tip
Compared to recording as hdf5, LeRobot Recorder integration may cause slight delays in teleoperation. If you encounter this issue, consider not using `--use_lerobot_recorder`.
::::
