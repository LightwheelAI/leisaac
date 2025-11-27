# Teleoperation

## Teleoperation Scripts

You can run teleoperation tasks with the script below. See [here](/resources/available_env) for more supported teleoperation tasks.

```shell
python scripts/environments/teleoperation/teleop_se3_agent.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --teleop_device=so101leader \
    --port=/dev/ttyACM0 \
    --num_envs=1 \
    --device=cuda \
    --enable_cameras \
    --record \
    --dataset_file=./datasets/dataset.hdf5
```

<details>
<summary><strong>Parameter descriptions for teleop_se3_agent.py</strong></summary>

- `--task`: Specify the task environment name to run, e.g., `LeIsaac-SO101-PickOrange-v0`.

- `--seed`: Specify the seed for environment, e.g., `42`.

- `--teleop_device`: Specify the teleoperation device type, e.g., `so101leader`, `bi-so101leader`, `keyboard`, `gamepad`.

-  `--port`: Specify the port of teleoperation device, e.g., `/dev/ttyACM0`. Only used when teleop_device is `so101leader`.

- `--left_arm_port`: Specify the port of left arm, e.g., `/dev/ttyACM0`. Only used when teleop_device is `bi-so101leader`.

- `--right_arm_port`: Specify the port of right arm, e.g., `/dev/ttyACM1`. Only used when teleop_device is `bi-so101leader`.

- `--num_envs`: Set the number of parallel simulation environments, usually `1` for teleoperation.

- `--device`: Specify the computation device, such as `cpu` or `cuda` (GPU).

- `--enable_cameras`: Enable camera sensors to collect visual data during teleoperation.

- `--record`: Enable data recording; saves teleoperation data to an HDF5 file.

- `--dataset_file`: Path to save the recorded dataset, e.g., `./datasets/record_data.hdf5`.

- `--resume`: Enable resume data recording from the existing dataset file.

- `--task_type`: Specify task type. If your dataset is recorded with keyboard, you should set it to `keyboard`, otherwise not to set it and keep default value None.

- `--quality`: Whether to enable quality render mode.

</details>

::::tip
We support multiple devices for teleoperation. See [here](/resources/available_devices) for more devices and usage instructions.
::::

## Operating Instructions

If the calibration file does not exist at the specified cache path, or if you launch with `--recalibrate`, you will be prompted to calibrate the SO101Leader.  Please refer to the [documentation](https://huggingface.co/docs/lerobot/so101#calibration-video) for calibration steps.

After entering the IsaacLab window, press the `b` key on your keyboard to start teleoperation. You can then use the specified teleop_device to control the robot in the simulation. If you need to reset the environment after completing your operation, simply press the `r` or `n` key. `r` means resetting the environment and marking the task as failed, while `n` means resetting the environment and marking the task as successful.

If you encounter permission errors such as `ConnectionError`, you can temporarily grant permission with the following command:
```bash
sudo chmod 666 /dev/ttyACM0
```

Alternatively, you can add the current user to the dialout group; you will need to restart your device for this to take effect:
```bash
sudo usermod -aG dialout $USER
```
