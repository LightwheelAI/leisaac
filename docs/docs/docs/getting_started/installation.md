# Installation

## 1. Environment Setup

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

::::tip
The steps above are essentially the same as the official IsaacLab documentation; please adjust according to the versions you use. Below is the compatibility between LeIsaac and IsaacLab and the related version dependencies. 

If you are using a 50-series GPU, we recommend using IsaacSim 5.0+ and IsaacLab `v2.2.1+`. We have tested on IsaacSim 5.0 and it works properly.

| Dependency | IsaacSim4.5 | IsaaSim5.0 | IsaacSim5.1 |
| ---------- | ----------- | ---------- | ----------- |
| Python     | 3.10        | 3.11       | 3.11        |
| IsaacLab   | v2.1.1      | v2.2.1     | v2.3.0      |
| CUDA       | 11.8        | 12.8       | 12.8        |
| PyTorch    | 2.5.1       | 2.7.0      | 2.7.0       |
::::

## 2. Clone LeIsaac Repository and Install

Clone our repository and install it as dependency.

```bash
cd ..
git clone https://github.com/LightwheelAI/leisaac.git
cd leisaac
pip install -e source/leisaac
```

## 3. Asset Preparation

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

::::info
Below are the download links for the scenes we provide. For more high-quality scene assets, please visit our [official website](https://lightwheel.ai/) or the [Releases page](https://github.com/LightwheelAI/leisaac/releases).

| Scene Name           | Description                        | Download Link                                                                            |
|----------------------|------------------------------------|------------------------------------------------------------------------------------------|
| Kitchen with Orange  | Example kitchen scene with oranges | [Download](https://github.com/LightwheelAI/leisaac/releases/tag/v0.1.0)                  |
| Lightwheel Toyroom   | Modern room with many toys         | [Download](https://github.com/LightwheelAI/leisaac/releases/tag/v0.1.1)                  |
| Table with Cube      | Simple table with one cube         | [Download](https://github.com/LightwheelAI/leisaac/releases/tag/v0.1.2)                  |
| Lightwheel Bedroom   | Realistic bedroom scene with cloth | [Download](https://github.com/LightwheelAI/leisaac/releases/tag/v0.2.0)                  |
::::

## 4. Device Setup 

We use the SO101Leader as the teleoperation device. Please follow the [official documentation](https://huggingface.co/docs/lerobot/so101) for connection and configuration.

::::tip
Note that you do not need to use the LeRobot repository for calibration; our codebase provides guided steps for the calibration process.
::::
