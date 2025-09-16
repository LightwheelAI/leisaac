# LeIsaac 🚀

https://github.com/user-attachments/assets/763acf27-d9a9-4163-8651-3ba0a6a185d7

This repository provides teleoperation functionality in [IsaacLab](https://isaac-sim.github.io/IsaacLab/main/index.html) using the SO101Leader ([LeRobot](https://github.com/huggingface/lerobot)), including data collection, data conversion, and subsequent policy training.

- 🤖 We use SO101Follower as the robot in IsaacLab and provide relevant teleoperation method.
- 🔄 We offer scripts to convert data from HDF5 format to the LeRobot Dataset.
- 🧠 We utilize simulation-collected data to fine-tune [GR00T N1.5](https://github.com/NVIDIA/Isaac-GR00T) and deploy it on real hardware.

> [!TIP]
> ***Welcome to the Lightwheel open-source community!***
>
> Join us, contribute, and help shape the future of AI and robotics. For questions or collaboration, contact [Zeyu](mailto:zeyu.hu@lightwheel.ai) or [Yinghao](mailto:yinghao.shuai@lightwheel.ai).

## Prerequisites & Installation 🛠️

### 1. Environment Setup

First, follow the [IsaacLab official installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) to install IsaacLab. We recommend using Conda for easier environment management. In summary, you only need to run the following command.

```bash
# Create and activate environment
conda create -n leisaac python=3.10
conda activate leisaac

# Install cuda-toolkit
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

# Install PyTorch
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118

# Install IsaacSim
pip install --upgrade pip
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

# Install IsaacLab
git clone git@github.com:isaac-sim/IsaacLab.git
sudo apt install cmake build-essential

cd IsaacLab
# fix isaaclab version for isaacsim4.5
git checkout v2.1.1
./isaaclab.sh --install
```

> [!TIP] 
> If you are using 50 series GPU, we recommend to use isaacsim5.0 and isaaclab with `v2.2.0`. We have tested it on isaacsim5.0 and it works properly. Below are some important dependency versions:
> | Dependency | Version |
> |------------|---------|
> | Python     | 3.11    |
> | CUDA       | 12.8    |
> | PyTorch    | 2.7.0   |
> | IsaacSim   | 5.0     |
> | IsaacLab   | 2.2.0   |


### 2. Clone This Repository and Install

Clone this repository and install it as dependency.

```bash
git clone https://github.com/LightwheelAI/leisaac.git
cd leisaac
pip install -e source/leisaac
```

## Asset Preparation 🏠

We provide an example USD asset—a kitchen scene. Please download related scene [here](https://github.com/LightwheelAI/leisaac/releases/tag/v0.1.0) and extract it into the `assets` directory. The directory structure should look like this:

```
<assets>
├── robots/
│   └── so101_follower.usd
└── scenes/
    └── kitchen_with_orange/
        ├── scene.usd
        ├── assets
        └── objects/
            ├── Orange001
            ├── Orange002
            ├── Orange003
            └── Plate
```

### Scene Assets Download Table

| Scene Name           | Description                        | Download Link                                                                            |
|----------------------|------------------------------------|------------------------------------------------------------------------------------------|
| Kitchen with Orange  | Example kitchen scene with oranges | [Download](https://github.com/LightwheelAI/leisaac/releases/tag/v0.1.0)                  |
| Lightwheel Toyroom   | Modern room with many toys         | [Download](https://github.com/LightwheelAI/leisaac/releases/tag/v0.1.1)                  |
| Table with Cube      | Simple table with one cube         | [Download](https://github.com/LightwheelAI/leisaac/releases/tag/v0.1.2)                  |
| Lightwheel Bedroom   | Realistic bedroom scene with cloth | [Download](https://github.com/LightwheelAI/leisaac/releases/tag/v0.2.0)                  |


> [!TIP] 
> For more high-quality scene assets, please visit our [official website](https://lightwheel.ai/) or the [Releases page](https://github.com/LightwheelAI/leisaac/releases).

## Device Setup 🎮

We use the SO101Leader as the teleoperation device. Please follow the [official documentation](https://huggingface.co/docs/lerobot/so101) for connection and configuration.

## Teleoperation Usage 🕹️

You can run teleoperation tasks with the following script:

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
<summary><strong>Parameter descriptions for teleop_se3_agent.py</strong></summary><p></p>

- `--task`: Specify the task environment name to run, e.g., `LeIsaac-SO101-PickOrange-v0`.

- `--seed`: Specify the seed for environment, e.g., `42`.

- `--teleop_device`: Specify the teleoperation device type, e.g., `so101leader`, `bi-so101leader`, `keyboard`.

-  `--port`: Specify the port of teleoperation device, e.g., `/dev/ttyACM0`. Only used when teleop_device is `so101leader`.

- `--left_arm_port`: Specify the port of left arm, e.g., `/dev/ttyACM0`. Only used when teleop_device is `bi-so101leader`.

- `--right_arm_port`: Specify the port of right arm, e.g., `/dev/ttyACM1`. Only used when teleop_device is `bi-so101leader`.

- `--num_envs`: Set the number of parallel simulation environments, usually `1` for teleoperation.

- `--device`: Specify the computation device, such as `cpu` or `cuda` (GPU).

- `--enable_cameras`: Enable camera sensors to collect visual data during teleoperation.

- `--record`: Enable data recording; saves teleoperation data to an HDF5 file.

- `--dataset_file`: Path to save the recorded dataset, e.g., `./datasets/record_data.hdf5`.

- `--task_type`: Specify task type. If your dataset is recorded with keyboard, you should set it to `keyboard`, otherwise not to set it and keep default value None.

- `--quality`: Whether to enable quality render mode.

</details>


If the calibration file does not exist at the specified cache path, or if you launch with `--recalibrate`, you will be prompted to calibrate the SO101Leader.  Please refer to the [documentation](https://huggingface.co/docs/lerobot/so101#calibration-video) for calibration steps.

After entering the IsaacLab window, press the `b` key on your keyboard to start teleoperation. You can then use the specified teleop_device to control the robot in the simulation. If you need to reset the environment after completing your operation, simply press the `r` or `n` key. `r` means resetting the environment and marking the task as failed, while `n` means resetting the environment and marking the task as successful.

**Troubleshooting:**

If you encounter permission errors like `ConnectionError`, you may need to run:

```bash
sudo chmod 666 /dev/ttyACM0

# or just add your user in related groups
sudo usermod -aG dialout $USER
```

## Dataset Replay 📺

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

## Data Convention & Conversion 📊

Collected teleoperation data is stored in HDF5 format in the specified directory. We provide a script to convert HDF5 data to the LeRobot Dataset format. Only successful episode will be converted.

>[!NOTE] 
> This script depends on the LeRobot runtime environment. We recommend using a separate Conda environment for LeRobot—see the official [LeRobot repo](https://github.com/huggingface/lerobot?tab=readme-ov-file#installation) for installation instructions.

You can modify the parameters in the script and run the following command:

```bash
python scripts/convert/isaaclab2lerobot.py
```

## Policy Training 🏋️‍♂️

[GR00T N1.5](https://github.com/NVIDIA/Isaac-GR00T) provides a fine-tuning workflow based on the LeRobot Dataset. You can refer to [nvidia/gr00t-n1.5-so101-tuning](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning) to fine-tune it with your collected lerobot data. We take pick-orange task as an example.

- First, collect a pick-orange dataset in simulation.
- Then, fine-tune GR00T N1.5 using this data.
- Finally, deploy the trained policy on real hardware.

## Policy Inference 🧩

We also provide interfaces for running policy inference in simulation. You can start inference with the following script (take gr00tn1.5 as example):

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

Depending on your use case, you may need to install additional dependencies to enable inference:

```shell
pip install -e "source/leisaac[gr00t]"
pip install -e "source/leisaac[lerobot-async]"
pip install -e "source/leisaac[openpi]"
```

> [!IMPORTANT]
> For service-based policies, you must start the corresponding service before running inference. For example, with GR00T, you need to launch the GR00T N1.5 inference server first. You can refer to the [GR00T evaluation documentation](https://github.com/NVIDIA/Isaac-GR00T?tab=readme-ov-file#4-evaluation) for detailed instructions.

Below are the currently supported policy inference methods:

<details>
<summary><strong>finetuned gr00t n1.5</strong></summary><p></p>

For usage instructions, please refer to the examples provided above.

</details>

<details>
<summary><strong>lerobot official policy</strong></summary><p></p>

We utilize lerobot's async inference capabilities for policy execution. For detailed information, please refer to the [official documentation](https://huggingface.co/docs/lerobot/async). Prior to execution, ensure that the policy server is running.

```shell
python scripts/evaluation/policy_inference.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --policy_type=lerobot-smolvla \
    --policy_host=localhost \
    --policy_port=8080 \
    --policy_timeout_ms=5000 \
    --policy_language_instruction='Pick the orange to the plate' \
    --policy_checkpoint_path=outputs/smolvla/leisaac-pick-orange/checkpoints/last/pretrained_model \
    --policy_action_horizon=50 \
    --device=cuda \
    --enable_cameras
```

</details>

<details>
<summary><strong>finetuned openpi</strong></summary><p></p>

We utilize openpi's remote inference capabilities for policy execution. For detailed information, please refer to the [official documentation](https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md). Prior to execution, ensure that the policy server is running.

```shell
python scripts/evaluation/policy_inference.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --policy_type=openpi \
    --policy_host=localhost \
    --policy_port=8000 \
    --policy_timeout_ms=5000 \
    --policy_language_instruction='Pick the orange to the plate' \
    --device=cuda \
    --enable_cameras
```

</details>


## Extra Feature ✨

We also provide some additional features. You can refer to the following instructions to try them out.

<details>
<summary><strong>DigitalTwin Env: Make Sim2Real Simple</strong></summary><p></p>

Inspired by the greenscreening functionality from [SIMPLER](https://simpler-env.github.io/) and [ManiSkill](https://github.com/haosulab/ManiSkill), we have implemented DigitalTwin Env. This feature allows you to replace the background in simulation environments with real background images while preserving the foreground elements such as robotic arms and interactive objects. This approach significantly reduces the gap between simulation and reality, enabling better sim2real transfer.

To use this feature, simply create a task configuration class that inherits from `ManagerBasedRLDigitalTwinEnvCfg` and launch it through the corresponding environment. In the configuration class, you can specify relevant parameters including overlay_mode, background images path, and foreground environment components to preserve. 

For usage examples, please refer to the sample task: [LiftCubeDigitalTwinEnvCfg](https://github.com/LightwheelAI/leisaac/blob/main/source/leisaac/leisaac/tasks/lift_cube/lift_cube_env_cfg.py).

</details>

<details>
<summary><strong>MimicGen Env: Generate Data From Demonstrations</strong></summary><p></p>

We have integrated [IsaacLab MimicGen](https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html), a powerful feature that automatically generates additional demonstrations from expert demonstrations.

To use this functionality, you first need to record some demonstrations. Recording scripts can be referenced from the instructions above. (Below we use the MimicGen for the `LeIsaac-SO101-LiftCube-v0` task as an example).

> **[NOTE]** 
> Pay attention to the `input_file` and `output_file` parameters in the following scripts. Typically, the `output_file` from the previous script becomes the `input_file` for the next script.

Since MimicGen requires trajectory generalization based on end-effector pose and object pose, we first convert joint-position-based action data to IK-based action data. The conversion process is as follows, where `input_file` specifies the collected demonstration data:

```shell
python scripts/mimic/eef_action_process.py \
    --input_file ./datasets/mimic-lift-cube-example.hdf5 \
    --output_file ./datasets/processed_mimic-lift-cube-example.hdf5 \
    --to_ik --headless
```

Next, we perform sub-task annotation based on the converted action data. Annotation can be done in two ways: automatic and manual. If you want to use automatic annotation, please add the `--auto` startup option.

```shell
python scripts/mimic/annotate_demos.py \
    --device cuda \
    --task LeIsaac-SO101-LiftCube-Mimic-v0 \
    --input_file ./datasets/processed_mimic-lift-cube-example.hdf5 \
    --output_file ./datasets/annotated_mimic-lift-cube-example.hdf5 \
    --enable_cameras
```

After annotation is complete, we can proceed with data generation. The generation process is as follows:

```shell
python scripts/mimic/generate_dataset.py \
    --device cuda \
    --num_envs 1 \
    --generation_num_trials 10 \
    --input_file ./datasets/annotated_mimic-lift-cube-example.hdf5 \
    --output_file ./datasets/generated_mimic-lift-cube-example.hdf5 \
    --enable_cameras
```

After obtaining the generated data, we also provide a conversion script to transform it from IK-based action data back to joint-position-based action data, as follows:

```shell
python scripts/mimic/eef_action_process.py \
    --input_file ./datasets/generated_mimic-lift-cube-example.hdf5 \
    --output_file ./datasets/final_generated_mimic-lift-cube-example.hdf5 \
    --to_joint --headless
```

Finally, you can use replay to view the effects of the generated data. It's worth noting that due to the inherent randomness in IsaacLab simulation, the replay performance may vary.

> **[NOTE]** 
> Depending on the device used to collect the data, you need to specify the corresponding task type with `--task_type`. For example, if your demonstrations were collected using the keyboard, add `--task_type=keyboard` when running `annotate_demos` and `generate_dataset`.
>
> `task_type` dose not need to be provided during replay the results `final_generated_dataset.hdf5`.

</details>

## Acknowledgements 🙏

We gratefully acknowledge [IsaacLab](https://github.com/isaac-sim/IsaacLab) and [LeRobot](https://github.com/huggingface/lerobot) for their excellent work, from which we have borrowed some code.

## Join Our Team! 💼

We're always looking for talented individuals passionate about AI and robotics! If you're interested in:

- 🤖 **Robotics Engineering**: Working with cutting-edge robotic systems and teleoperation
- 🧠 **AI/ML Research**: Developing next-generation AI models for robotics
- 💻 **Software Engineering**: Building robust, scalable robotics software
- 🔬 **Research & Development**: Pushing the boundaries of what's possible in robotics

**Join us at Lightwheel AI!** We offer:
- Competitive compensation and benefits
- Work with state-of-the-art robotics technology
- Collaborative, innovative environment
- Opportunity to shape the future of AI-powered robotics

**[Apply Now →](https://lightwheel.ai/career)** | **[Contact Now →](mailto:zeyu.hu@lightwheel.ai)** | **[Learn More About Us →](https://lightwheel.ai)**

---

**Let's build the future of robotics together! 🤝**

---
