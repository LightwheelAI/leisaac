# DigitalTwin Env
> DigitalTwin Env: Make Sim2Real Simple

Inspired by the greenscreening functionality from [SIMPLER](https://simpler-env.github.io/) and [ManiSkill](https://github.com/haosulab/ManiSkill), we have implemented DigitalTwin Env. This feature allows you to replace the background in simulation environments with real background images while preserving the foreground elements such as robotic arms and interactive objects. This approach significantly reduces the gap between simulation and reality, enabling better sim2real transfer.

To use this feature, simply create a task configuration class that inherits from `ManagerBasedRLDigitalTwinEnvCfg` and launch it through the corresponding environment. In the configuration class, you can specify relevant parameters including overlay_mode, background images path, and foreground environment components to preserve.

:::info
For usage examples, please refer to the sample task: [LiftCubeDigitalTwinEnvCfg](https://github.com/LightwheelAI/leisaac/blob/main/source/leisaac/leisaac/tasks/lift_cube/lift_cube_env_cfg.py).
:::
