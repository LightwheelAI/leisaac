# LeIsaac 🚀

https://github.com/user-attachments/assets/763acf27-d9a9-4163-8651-3ba0a6a185d7

This repository provides teleoperation functionality in [IsaacLab](https://isaac-sim.github.io/IsaacLab/main/index.html) using the SO101Leader ([LeRobot](https://github.com/huggingface/lerobot)), including data collection, data conversion, and subsequent policy training.

- 🤖 We use SO101Follower as the robot (and other related robot) in IsaacLab and provide relevant teleoperation method.
- 🔄 We offer scripts to convert data from HDF5 format to the LeRobot Dataset.
- 🧠 We utilize simulation-collected data to fine-tune [GR00T N1.5](https://github.com/NVIDIA/Isaac-GR00T) and deploy it on real hardware. And more policies will be supported.

> [!TIP]
> ***Welcome to the Lightwheel open-source community!***
>
> Join us, contribute, and help shape the future of AI and robotics. For questions or collaboration, contact [Zeyu](mailto:zeyu.hu@lightwheel.ai) or [Yinghao](mailto:yinghao.shuai@lightwheel.ai).

## News 🗞️
- [26/04/14] Remote teleoperation is now available in LeIsaac! Try it out [here](https://lightwheelai.github.io/leisaac/docs/getting_started/teleoperation#remote-teleoperation).
- [26/03/10] With the new `datagen` module, LeIsaac can generate motion trajectories programmatically. See [State Machine Data Generation](https://lightwheelai.github.io/leisaac/docs/features/state_machine).
- [26/01/16] Added inference support for GR00T N1.6; details are in [Available Policy Inference](https://lightwheelai.github.io/leisaac/resources/available_policy#finetuned-gr00t-n16).
- [26/01/13] Try our tutorial [LeIsaac x Cosmos](https://lightwheelai.github.io/leisaac/docs/tutorials/cosmos_tutorial) to get a video2action data generation pipeline.
- [26/01/12] Extra feature of [lerobot recorder integration](https://lightwheelai.github.io/leisaac/docs/features/lerobot_recorder) released! You can now record data directly in LeRobot Dataset format during teleoperation.
- [25/12/19] Try our tutorial [LeIsaac x Marble](https://lightwheelai.github.io/leisaac/docs/tutorials/marble_tutorial) to build and evaluate diverse embodied tasks across large-scale generalized environments.
- [25/12/19] We now support lekiwi-based teleoperation and provide a Loft scene for the community. See the example task [here](https://lightwheelai.github.io/leisaac/resources/available_env).
- [25/11/27] We now support more teleoperation devices, including the enhanced keyboard and gamepad. Refer to [the device guide](https://lightwheelai.github.io/leisaac/resources/available_devices) for usage details.
- [25/11/26] LeIsaac is now the official imitation-learning (IL) simulation playground integrated into LeRobot’s EnvHub. It provides fast, scalable Isaac-based environments designed for imitation learning, control, and policy evaluation. It is featured in LeRobot: [LeIsaac × LeRobot EnvHub](https://huggingface.co/docs/lerobot/en/envhub_leisaac)


## Getting Started 📚

Please refer to our [documentation](https://lightwheelai.github.io/leisaac/) to learn how to use this repository. Follow these links to learn more about:

- [Installation and Setup](https://lightwheelai.github.io/leisaac/docs/getting_started/installation)
- [Extra Features](https://lightwheelai.github.io/leisaac/docs/features)
- [Policy Inference](https://lightwheelai.github.io/leisaac/docs/getting_started/policy_support)
- [Available Robots](https://lightwheelai.github.io/leisaac/resources/available_robots), [Environments](https://lightwheelai.github.io/leisaac/resources/available_env) and [Policy](https://lightwheelai.github.io/leisaac/resources/available_policy)

## Contributing 🤝

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for how to report issues and submit pull requests.

## Citation 📝

If you use leisaac, please cite it as follows.

```txt
@software{Lightwheel_and_LeIsaac_Project_Developers_LeIsaac_2025,
author = {{Lightwheel} and {LeIsaac Project Developers}},
license = {Apache-2.0},
month = dec,
title = {{LeIsaac}},
url = {https://github.com/LightwheelAI/leisaac},
version = {0.4.0},
year = {2026}
}
```

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
