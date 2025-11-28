---
sidebar_position: 1
slug: /
---

# Introduction

<video
  controls
  src="https://github.com/user-attachments/assets/763acf27-d9a9-4163-8651-3ba0a6a185d7"
  style={{ width: '100%', maxWidth: '960px', borderRadius: '8px' }}
/>

LeIsaac provides teleoperation functionality in [IsaacLab](https://isaac-sim.github.io/IsaacLab/main/index.html) using the SO101Leader ([LeRobot](https://github.com/huggingface/lerobot)), including data collection, data conversion, and subsequent policy training.

- 🤖 We use the SO101 Follower robot (and other related robot) in IsaacLab and provide practical teleoperation methods.
- 🔄 Ready-to-use scripts convert HDF5 data into the LeRobot dataset format.
- 🧠 Simulation data is used to fine-tune [GR00T N1.5](https://github.com/NVIDIA/Isaac-GR00T) and deploy the policy on real hardware. And more policies will be supported.

## News
- [25/11/27] We now support more teleoperation devices, including the enhanced keyboard and gamepad. Refer to [the device guide](/resources/available_devices) for usage details.
- [25/11/26] LeIsaac is now the official imitation-learning (IL) simulation playground integrated into LeRobot’s EnvHub. It provides fast, scalable Isaac-based environments designed for imitation learning, control, and policy evaluation. It is featured in LeRobot: [LeIsaac × LeRobot EnvHub](https://huggingface.co/docs/lerobot/en/envhub_leisaac)
