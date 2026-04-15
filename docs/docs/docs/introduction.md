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
- 🦾 State machine scripted policies enable fully automated data collection without human teleoperation.
- 🔄 Ready-to-use scripts convert HDF5 data into the LeRobot dataset format.
- 🧠 Simulation data is used to fine-tune [GR00T N1.5](https://github.com/NVIDIA/Isaac-GR00T) and deploy the policy on real hardware. And more policies will be supported.

## News
- [26/04/14] Remote teleoperation is now available in LeIsaac! Try it out [here](/docs/getting_started/teleoperation#remote-teleoperation).
- [26/03/10] With the new `datagen` module, LeIsaac can generate motion trajectories programmatically. See [State Machine Data Generation](/docs/features/state_machine).
- [26/01/16] Added inference support for GR00T N1.6; details are in [Available Policy Inference](/resources/available_policy#finetuned-gr00t-n16).
- [26/01/13] Try our tutorial [LeIsaac x Cosmos](/docs/tutorials/cosmos_tutorial) to get a video2action data generation pipeline.
- [26/01/12] Extra feature of [lerobot recorder integration](/docs/features/lerobot_recorder) released! You can now record data directly in LeRobot Dataset format during teleoperation.
- [25/12/19] Try our tutorial [LeIsaac x Marble](/docs/tutorials/marble_tutorial) to build and evaluate diverse embodied tasks across large-scale generalized environments.
- [25/12/19] We now support lekiwi-based teleoperation and provide a Loft scene for the community. See the example task [here](/resources/available_env).
- [25/11/27] We now support more teleoperation devices, including the enhanced keyboard and gamepad. Refer to [the device guide](/resources/available_devices) for usage details.
- [25/11/26] LeIsaac is now the official imitation-learning (IL) simulation playground integrated into LeRobot’s EnvHub. It provides fast, scalable Isaac-based environments designed for imitation learning, control, and policy evaluation. It is featured in LeRobot: [LeIsaac × LeRobot EnvHub](https://huggingface.co/docs/lerobot/en/envhub_leisaac)
