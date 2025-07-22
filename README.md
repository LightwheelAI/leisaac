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
./isaaclab.sh --install
```

### 2. Clone This Repository

Clone this repository and install it as dependency.

```bash
git clone https://github.com/LightwheelAI/leisaac.git
cd leisaac
pip install -e source/leisaac
```

### 3. Install Extra Dependencies

```bash
pip install pynput pyserial deepdiff feetech-servo-sdk
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

| Scene Name           | Description                        | Download Link                                                                             |
|----------------------|------------------------------------|------------------------------------------------------------------------------------------|
| Kitchen with Orange  | Example kitchen scene with oranges | [Download](https://github.com/LightwheelAI/leisaac/releases/tag/v0.1.0)                  |
| Lightwheel Toyroom   | Modern room with many toys         | [Download](https://github.com/LightwheelAI/leisaac/releases/tag/v0.1.1)                  |


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
    --device=cpu \
    --enable_cameras \
    --record \
    --dataset_file=./datasets/dataset.hdf5
```

**Parameter Descriptions:**

- `--task`: Specify the task environment name to run, e.g., `LeIsaac-SO101-PickOrange-v0`.

- `--teleop_device`: Specify the teleoperation device type, e.g., `so101leader`, `bi-so101-leader`, `keyboard`.

-  `--port`: Specify the port of teleoperation device, e.g., `/dev/ttyACM0`. only used when teleop_device is `so101leader`.

- `--num_envs`: Set the number of parallel simulation environments, usually `1` for teleoperation.

- `--device`: Specify the computation device, such as `cpu` or `cuda` (GPU).

- `--enable_cameras`: Enable camera sensors to collect visual data during teleoperation.

- `--record`: Enable data recording; saves teleoperation data to an HDF5 file.

- `--dataset_file`: Path to save the recorded dataset, e.g., `./datasets/record_data.hdf5`.
  
If the calibration file does not exist at the specified cache path, or if you launch with `--recalibrate`, you will be prompted to calibrate the SO101Leader.  Please refer to the [documentation](https://huggingface.co/docs/lerobot/so101#calibration-video) for calibration steps.

After entering the IsaacLab window, press the `b` key on your keyboard to start teleoperation. You can then use the specified teleop_device to control the robot in the simulation. If you need to reset the environment after completing your operation, simply press the `r` or `n` key. `r` means resetting the environment and marking the task as failed, while `n` means resetting the environment and marking the task as successful.

**Troubleshooting:**

If you encounter permission errors like `ConnectionError`, you may need to run:

```bash
sudo chmod 666 /dev/ttyACM0

# or just add your user in related groups
sudo usermod -aG dialout $USER
```

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
