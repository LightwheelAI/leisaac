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

- `--teleop_device`: Specify the teleoperation device type, e.g., `so101leader`, `bi-so101leader`, `keyboard`, `gamepad`, `lekiwi-leader`, `lekiwi-keyboard`, `lekiwi-gamepad`.

-  `--port`: Specify the port of teleoperation device, e.g., `/dev/ttyACM0`. Only used when teleop_device is `so101leader` and `lekiwi-leader`.

- `--remote_endpoint`: ZMQ endpoint for remote so101leader (e.g., `tcp://192.168.1.10:5556`). When set, connects to a `so101_joint_state_server.py` running on the machine with the leader arm. See [Remote Teleoperation](#remote-teleoperation) below.

- `--left_arm_port`: Specify the port of left arm, e.g., `/dev/ttyACM0`. Only used when teleop_device is `bi-so101leader`.

- `--right_arm_port`: Specify the port of right arm, e.g., `/dev/ttyACM1`. Only used when teleop_device is `bi-so101leader`.

- `--num_envs`: Set the number of parallel simulation environments, usually `1` for teleoperation.

- `--device`: Specify the computation device, such as `cpu` or `cuda` (GPU).

- `--enable_cameras`: Enable camera sensors to collect visual data during teleoperation.

- `--record`: Enable data recording; saves teleoperation data to an HDF5 file.

- `--dataset_file`: Path to save the recorded dataset, e.g., `./datasets/record_data.hdf5`.

- `--resume`: Enable resume data recording from the existing dataset file.

- `--recalibrate`: Recalibrate SO101-Leader or Bi-SO101Leader.

- `--quality`: Whether to enable quality render mode.

- `--use_lerobot_recorder`: Whether to use lerobot recorder.

- `--lerobot_dataset_repo_id`: LeRobot dataset repository ID.

- `--lerobot_dataset_fps`: LeRobot dtaset frames per second.

</details>

::::tip
We support multiple devices for teleoperation. See [here](/resources/available_devices) for more devices and usage instructions.
::::

## Remote Teleoperation

When the leader arm is connected to a different machine than the one running Isaac Sim
(e.g., Isaac Sim on a cloud GPU instance, leader arm on your laptop), you can use
**remote teleoperation** via ZMQ.

### How it works

The leader arm machine runs a lightweight publisher that reads motor positions and
streams them over the network. The Isaac Sim machine subscribes and uses the joint
states for teleoperation — no USB forwarding needed.

```
Laptop (leader arm)                      Cloud GPU (Isaac Sim)
┌──────────────────────────┐  ZMQ PUB/SUB  ┌──────────────────────┐
│ so101_joint_state_server │──────────────►│ SO101LeaderRemote    │
│ reads motors             │   tcp:5556    │ teleop_se3_agent.py  │
│ at 50 Hz                 │               │ --remote_endpoint    │
└──────────────────────────┘               └──────────────────────┘
```

### Prerequisites

- Network connectivity between the two machines (direct or via SSH tunnel)
- `pyzmq` installed on the Isaac Sim machine: `pip install leisaac[remote]` or `pip install pyzmq`

### Local Machine Setup

On the machine where the leader arm is connected, install leisaac with remote support:

```bash
pip install leisaac[remote]
```

### Usage

**Terminal 1 — Local machine (leader arm):**

```bash
python scripts/environments/teleoperation/so101_joint_state_server.py \
    --port /dev/ttyACM0 --id leader_arm --rate 50
```

If no calibration file exists, the script will run an interactive calibration process automatically. To force recalibration, add `--recalibrate`.

**Terminal 2 — Remote machine (Isaac Sim):**

```bash
python scripts/environments/teleoperation/teleop_se3_agent.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --teleop_device=so101leader \
    --remote_endpoint=tcp://<local-machine-ip>:5556 \
    --num_envs=1 --device=cuda --enable_cameras
```

### SSH Reverse Port Forwarding

If the cloud instance cannot reach your laptop directly (e.g., behind NAT or firewall),
use SSH reverse port forwarding to expose the publisher's port on the remote machine:

```bash
# On your laptop — forward local port 5556 to the remote machine's localhost:5556
ssh -R 5556:localhost:5556 ubuntu@<cloud-instance-ip>
```

Then on the remote machine, connect to `localhost` instead of your laptop's IP:

```bash
python scripts/environments/teleoperation/teleop_se3_agent.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --teleop_device=so101leader \
    --remote_endpoint=tcp://localhost:5556 \
    --num_envs=1 --device=cuda --enable_cameras
```

### Parameters

- `--remote_endpoint`: ZMQ endpoint to subscribe to (e.g., `tcp://192.168.1.10:5556`
  or `tcp://localhost:5556` with SSH tunnel). When set, uses `SO101LeaderRemote` instead
  of the local `SO101Leader`.

- `--id` (publisher): Calibration ID (default: `leader_arm`). Calibration is stored in
  `scripts/environments/teleoperation/.cache/{id}.json`.

- `--rate` (publisher): Motor read rate in Hz (default: 50). 30–50 Hz is sufficient
  for LeIsaac teleoperation.

- `--recalibrate` (publisher): Force recalibration even if a calibration file exists.

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
