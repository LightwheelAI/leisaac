# Policy Training & Inference

## 1. Data Convention

Collected teleoperation data is stored in HDF5 format in the specified directory. We provide a script to convert HDF5 data to the LeRobot Dataset format. Only successful episode will be converted.

:::info
This script depends on the LeRobot runtime environment. We recommend using a separate Conda environment for LeRobot—see the official [LeRobot repo](https://github.com/huggingface/lerobot?tab=readme-ov-file#installation) for installation instructions.
:::

You can modify the parameters in the script and run the following command. (This script is a conversion example we provide; please modify the parameters according to your specific needs before using it.)

```bash
python scripts/convert/isaaclab2lerobot.py
```

## 2. Policy Training

Taking [GR00T N1.5](https://github.com/NVIDIA/Isaac-GR00T) as an example, which provides a fine-tuning workflow based on the LeRobot Dataset. You can refer to [nvidia/gr00t-n1.5-so101-tuning](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning) to fine-tune it with your collected lerobot data. We take pick-orange task as an example.

After completing policy training, you will obtain a checkpoint that can be used to launch the inference service using the `inference_service.py` provided by GR00T N1.5.

## 3. Policy Inference

We also provide interfaces for running policy inference in simulation. First, you need to install additional dependencies:

```bash
pip install -e "source/leisaac[gr00t]"
```

Then, you need to launch the GR00T N1.5 inference server. You can refer to the [GR00T evaluation documentation](https://github.com/NVIDIA/Isaac-GR00T?tab=readme-ov-file#4-evaluation) for detailed instructions.

After that, you can start inference with the following script:

```shell
python scripts/evaluation/policy_inference.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --eval_rounds=10 \
    --policy_type=gr00tn1.5 \
    --policy_host=localhost \
    --policy_port=5555 \
    --policy_timeout_ms=5000 \
    --policy_action_horizon=16 \
    --policy_language_instruction="Pick up the orange and place it on the plate" \
    --device=cuda \
    --enable_cameras
```

<details>
<summary><strong>Parameter descriptions for policy_inference.py</strong></summary><p></p>

- `--task`: Name of the task environment to run for inference (e.g., `LeIsaac-SO101-PickOrange-v0`).

- `--seed`: Seed of environment (default: current time).

- `--episode_length_s`: Episode length in seconds (default: `60`).

- `--eval_rounts`: Number of evaluation rounds. 0 means don't add time out termination, policy will run until success or manual reset (default: `0`)

- `--policy_type`: Type of policy to use (default: `gr00tn1.5`).
    - now we support `gr00tn1.5`, `lerobot-<model_type>`

- `--policy_host`: Host address of the policy server (default: `localhost`).

- `--policy_port`: Port of the policy server (default: `5555`).

- `--policy_timeout_ms`: Timeout for the policy server in milliseconds (default: `5000`).

- `--policy_action_horizon`: Number of actions to predict per inference (default: `16`).

- `--policy_language_instruction`: Language instruction for the policy (e.g., task description in natural language).

- `--policy_checkpoint_path`: Path to the policy checkpoint (if required).

- `--device`: Computation device, such as `cpu` or `cuda`.

You may also use additional arguments supported by IsaacLab's `AppLauncher` (see their documentation for details).

</details>

## 4. Examples

We provide simulation-collected data (Pick Orange) and the corresponding fine-tuned GR00T N1.5 policy, which can be downloaded from the following links:

- `dataset`: https://huggingface.co/datasets/LightwheelAI/leisaac-pick-orange
- `policy`: https://huggingface.co/LightwheelAI/leisaac-pick-orange-v0

The following videos demonstrate inference results in simulation, corresponding to two different tasks. Both tasks follow the complete workflow: data collection in simulation, fine-tuning GR00T N1.5, and inference in simulation.

| PickOrange | LiftCube |
| ---------- | -------- |
| <video src="https://github.com/user-attachments/assets/26c2b91d-3886-4fc5-839c-140d3839036b" autoPlay loop muted playsInline style={{maxHeight: '250px'}}></video> | <video src="https://github.com/user-attachments/assets/03f0649d-ddb6-419d-b4d9-e45cb91b2aa9" autoPlay loop muted playsInline style={{maxHeight: '250px'}}></video> |
