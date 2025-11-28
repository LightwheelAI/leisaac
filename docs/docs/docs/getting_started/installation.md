# Installation

## 1. Environment Setup

### Install as a Package

You can install LeIsaac as a dependency. The script below provisions IsaacLab, IsaacSim, and all required components.

```bash
conda create -n leisaac python=3.11
conda activate leisaac

# Install cuda-toolkit
conda install -c "nvidia/label/cuda-12.8.1" cuda-toolkit

# Install PyTorch (CUDA 12.8 wheels)
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# Install LeIsaac and IsaacLab/IsaacSim extras
pip install 'leisaac[isaaclab] @ git+https://github.com/LightwheelAI/leisaac.git#subdirectory=source/leisaac' --extra-index-url https://pypi.nvidia.com
```

::::tip
Install as a Package may expose edge cases. If you encounter issues, please open an issue on GitHub and consider switching to the “install from source” workflow described below.
::::

### Install from Source

You can also install directly from the source for local development. First, clone our repository and related submodules.

```bash
git clone https://github.com/LightwheelAI/leisaac.git --recursive
```

Then follow the [IsaacLab official installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) to install IsaacLab. We recommend using Conda for easier environment management. In summary, you only need to run the following command.

```bash
# Create and activate environment
conda create -n leisaac python=3.11
conda activate leisaac

# Install cuda-toolkit
conda install -c "nvidia/label/cuda-12.8.1" cuda-toolkit

# Install PyTorch
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# Install IsaacSim
pip install --upgrade pip
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# Install IsaacLab
sudo apt install cmake build-essential

cd leisaac/dependencies/IsaacLab
./isaaclab.sh --install
```

Finally, install leisaac as dependency.
```bash
cd ../..
pip install -e source/leisaac
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

## 2. Asset Preparation

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

You can also download scenes from [huggingface](https://huggingface.co/LightwheelAI/leisaac_env/tree/main), which be stored in the `assets` directory.
::::

## 3. Device Setup

We use the SO101Leader as the teleoperation device. Please follow the [official documentation](https://huggingface.co/docs/lerobot/so101) for connection and configuration.

::::tip
Note that you do not need to use the LeRobot repository for calibration; our codebase provides guided steps for the calibration process.
::::
