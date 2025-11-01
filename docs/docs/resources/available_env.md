import useBaseUrl from '@docusaurus/useBaseUrl';

# Available Environments

The following table lists all available tasks and environments in LeIsaac. You can also get the latest list of environments by running the following command:

```bash
python scripts/environments/list_envs.py
```

| Task | Environment ID | Task Description | Related Robot |
| :--- | :-------------- | :--------------- | :------------ |
| <video src={useBaseUrl('/video/envs/pick_orange.mp4')} autoPlay loop muted playsInline style={{maxWidth: '300px'}}></video> | [LeIsaac-SO101-PickOrange-v0](https://github.com/LightwheelAI/leisaac/blob/main/source/leisaac/leisaac/tasks/pick_orange/pick_orange_env_cfg.py)<br /><br />[LeIsaac-SO101-PickOrange-Direct-v0](https://github.com/LightwheelAI/leisaac/blob/main/source/leisaac/leisaac/tasks/pick_orange/direct/pick_orange_env.py) | Pick three oranges and put them into the plate, then reset the arm to rest state. | Single-Arm SO101 Follower |
| <video src={useBaseUrl('/video/envs/lift_cube.mp4')} autoPlay loop muted playsInline style={{maxWidth: '300px'}}></video> | [LeIsaac-SO101-LiftCube-v0](https://github.com/LightwheelAI/leisaac/blob/main/source/leisaac/leisaac/tasks/lift_cube/lift_cube_env_cfg.py)<br /><br />[LeIsaac-SO101-LiftCube-Direct-v0](https://github.com/LightwheelAI/leisaac/blob/main/source/leisaac/leisaac/tasks/lift_cube/direct/lift_cube_env.py) | Lift the red cube up. | Single-Arm SO101 Follower |
| <video src={useBaseUrl('/video/envs/clean_toyroom.mp4')} autoPlay loop muted playsInline style={{maxWidth: '300px'}}></video> | [LeIsaac-SO101-CleanToyTable-v0](https://github.com/LightwheelAI/leisaac/blob/main/source/leisaac/leisaac/tasks/clean_toy_table/clean_toy_table_env_cfg.py)<br /><br />[LeIsaac-SO101-CleanToyTable-BiArm-v0](https://github.com/LightwheelAI/leisaac/blob/main/source/leisaac/leisaac/tasks/clean_toy_table/clean_toy_table_bi_arm_env_cfg.py)<br /><br />[LeIsaac-SO101-CleanToyTable-BiArm-Direct-v0](https://github.com/LightwheelAI/leisaac/blob/main/source/leisaac/leisaac/tasks/clean_toy_table/direct/clean_toy_table_bi_arm_env.py) | Pick two letter e objects into the box, and reset the arm to rest state. | Single-Arm SO101 Follower<br /><br />Bi-Arm SO101 Follower |
| <video src={useBaseUrl('/video/envs/fold_cloth.mp4')} autoPlay loop muted playsInline style={{maxWidth: '300px'}}></video> | [LeIsaac-SO101-FoldCloth-BiArm-v0](https://github.com/LightwheelAI/leisaac/blob/main/source/leisaac/leisaac/tasks/fold_cloth/fold_cloth_bi_arm_env_cfg.py)<br /><br />[LeIsaac-SO101-FoldCloth-BiArm-Direct-v0](https://github.com/LightwheelAI/leisaac/blob/main/source/leisaac/leisaac/tasks/fold_cloth/direct/fold_cloth_bi_arm_env.py) | Fold the cloth, and reset the arm to rest state.<br /><br />*Note: Only the DirectEnv support check_success in this task.* | Bi-Arm SO101 Follower |