from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from .pick_orange_env_cfg import PickOrangeEnvCfg


@configclass
class PickOrangeMimicEnvCfg(PickOrangeEnvCfg, MimicEnvCfg):
    """
    Configuration for the pick orange task with mimic environment.
    """

    def __post_init__(self):
        super().__post_init__()

        self.datagen_config.name = "pick_orange_leisaac_task_v0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 42

        subtask_configs = []
        """
        subtask: pick_orange001 -> put_orange001_to_plate -> pick_orange002 -> put_orange002_to_plate -> pick_orange003 -> put_orange003_to_plate -> rest robot
        """
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref="Orange001",
                # This key corresponds to the binary indicator in "datagen_info" that signals
                # when this subtask is finished (e.g., on a 0 to 1 edge).
                subtask_term_signal="pick_orange001",
                # Specifies time offsets for data generation when splitting a trajectory into
                # subtask segments. Random offsets are added to the termination boundary.
                subtask_term_offset_range=(10, 20),
                # Selection strategy for the source subtask segment during data generation
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.003,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=5,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
                description="Pick Orange001",
                next_subtask_description="Put Orange001 to plate",
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="Plate",
                subtask_term_signal="put_orange001_to_plate",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                next_subtask_description="Pick Orange002",
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="Orange002",
                subtask_term_signal="pick_orange002",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                next_subtask_description="Put Orange002 to plate",
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="Plate",
                subtask_term_signal="put_orange002_to_plate",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                next_subtask_description="Pick Orange003",
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="Orange003",
                subtask_term_signal="pick_orange003",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                next_subtask_description="Put Orange003 to plate",
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="Plate",
                subtask_term_signal="put_orange003_to_plate",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                next_subtask_description="Rest robot",
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref=None,
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="random",
                selection_strategy_kwargs={},
                action_noise=0.003,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        self.subtask_configs["so101_follower"] = subtask_configs
