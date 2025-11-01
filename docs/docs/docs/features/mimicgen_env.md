# MimicGen Env
> MimicGen Env: Generate Data From Demonstrations

We have integrated [IsaacLab MimicGen](https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html), a powerful feature that automatically generates additional demonstrations from expert demonstrations.

To use this functionality, you first need to record some demonstrations. Recording scripts can be referenced from the instructions above. (Below we use the MimicGen for the `LeIsaac-SO101-LiftCube-v0` task as an example).

:::info
Pay attention to the `input_file` and `output_file` parameters in the following scripts. Typically, the `output_file` from the previous script becomes the `input_file` for the next script.
:::

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

:::info
Depending on the device used to collect the data, you need to specify the corresponding task type with `--task_type`. For example, if your demonstrations were collected using the keyboard, add `--task_type=keyboard` when running `annotate_demos` and `generate_dataset`.

`task_type` does not need to be provided when replaying the results from `final_generated_dataset.hdf5`.
:::

For reference, we also provide example data, including both the [original collected data](https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2FLightwheelAI%2Fleisaac-pick-orange%2Fepisode_0) and the [MimicGen-generated data](https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2FLightwheelAI%2Fleisaac-pick-orange-mimic-v0%2Fepisode_0).
