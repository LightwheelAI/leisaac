"""State machine for the lift-cube task."""

import torch
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_inv, quat_mul

from .base import StateMachineBase

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSE = -1.0
_GRIPPER_OFFSET = 0.08  # vertical clearance for gripper tip above cube center
_APPROACH_STEPS: int = 120  # steps to smoothly interpolate from init EE pos to hover
_EE_X_OFFSET = -0.03  # tune to align fingertip midpoint with cube center in x
_EE_Y_OFFSET = -0.01  # tune to align fingertip midpoint with cube center in y


_REST_POSE_DEG = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -100.0,
    "elbow_flex": 90.0,
    "wrist_flex": 50.0,
    "wrist_roll": 0.0,
    "gripper": -10.0,
}

_SUCCESS_HEIGHT = 0.20  # cube must be >= this height (m) above robot base


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class LiftCubeStateMachine(StateMachineBase):
    """State machine for the lift-cube manipulation task.

    The robot lifts a single cube to at least 20 cm above the robot base.
    The episode ends while the cube is still held aloft so that
    :meth:`check_success` can measure the achieved height reliably.

    Phase sequence:

    ====  =====  ===================================================
    From  To     Description
    ====  =====  ===================================================
       0    119  Smooth approach from initial EE position to hover.
     120    219  Hold hover above cube, gripper open.
     220    319  Lower EE to grasp height.
     320    380  Close gripper on cube.
     381    500  Lift cube upward.
     501    550  Hold cube at lifted height — episode ends here.
    ====  =====  ===================================================
    """

    MAX_STEPS: int = 551

    def __init__(self) -> None:
        self._step_count: int = 0
        self._episode_done: bool = False
        self._initial_ee_pos: torch.Tensor | None = None
        self._rest_ee_pos_world: torch.Tensor | None = None
        self._rest_joint_pos: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # StateMachineBase interface
    # ------------------------------------------------------------------

    def setup(self, env) -> None:
        """FK calibration: drive arm to rest pose and record EE world position."""
        # Override gripper close position for this task.
        gripper_term = env.action_manager._terms["gripper_action"]
        gripper_term._close_command[:] = -1.0

        robot = env.scene["robot"]
        joint_names = list(robot.data.joint_names)

        self._rest_joint_pos = torch.zeros(env.num_envs, len(joint_names), device=env.device)
        for idx, name in enumerate(joint_names):
            if name in _REST_POSE_DEG:
                self._rest_joint_pos[:, idx] = _REST_POSE_DEG[name] * torch.pi / 180.0

        robot.write_joint_state_to_sim(
            position=self._rest_joint_pos,
            velocity=torch.zeros_like(self._rest_joint_pos),
        )
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
        self._rest_ee_pos_world = robot.data.body_pos_w[:, -1, :].clone()

    def check_success(self, env) -> bool:
        """Return True if the cube is at least 20 cm above the robot base."""
        robot = env.scene["robot"]
        cube = env.scene["cube"]
        base_index = robot.data.body_names.index("base")
        robot_base_height = robot.data.body_pos_w[:, base_index, 2]
        cube_height = cube.data.root_pos_w[:, 2]
        height_above_base = cube_height - robot_base_height
        return bool((height_above_base >= _SUCCESS_HEIGHT).all().item())

    def get_action(self, env) -> torch.Tensor:
        """Compute the action tensor for the current step (8D IK pose target)."""
        robot = env.scene["robot"]
        robot.write_joint_damping_to_sim(damping=5.0)

        device = env.device
        num_envs = env.num_envs
        step = self._step_count

        cube_pos_w = env.scene["cube"].data.root_pos_w.clone()
        robot_base_pos_w = robot.data.root_pos_w.clone()
        robot_base_quat_w = robot.data.root_quat_w.clone()

        target_quat_w = quat_from_euler_xyz(
            torch.tensor(0.0, device=device),
            torch.tensor(0.0, device=device),
            torch.tensor(0.0, device=device),
        ).repeat(num_envs, 1)
        target_quat = quat_mul(quat_inv(robot_base_quat_w), target_quat_w)

        if step == 0:
            self._initial_ee_pos = robot.data.body_pos_w[:, -1, :].clone()

        if step < _APPROACH_STEPS:
            target_pos_w, gripper_cmd = self._phase_approach_hover(cube_pos_w, num_envs, device)
        elif step < 220:
            target_pos_w, gripper_cmd = self._phase_hover_above_cube(cube_pos_w, num_envs, device)
        elif step < 320:
            target_pos_w, gripper_cmd = self._phase_lower_to_cube(cube_pos_w, num_envs, device)
        elif step < 381:
            target_pos_w, gripper_cmd = self._phase_grasp(cube_pos_w, num_envs, device)
        elif step < 501:
            target_pos_w, gripper_cmd = self._phase_lift_cube(cube_pos_w, num_envs, device)
        else:
            target_pos_w, gripper_cmd = self._phase_hold_cube(cube_pos_w, num_envs, device)

        diff_w = target_pos_w - robot_base_pos_w
        target_pos_local = quat_apply(quat_inv(robot_base_quat_w), diff_w)
        return torch.cat([target_pos_local, target_quat, gripper_cmd], dim=-1)

    def advance(self) -> None:
        """Advance step counter and mark episode done when finished."""
        self._step_count += 1
        if self._step_count >= self.MAX_STEPS:
            self._episode_done = True

    def reset(self) -> None:
        """Reset the state machine to its initial state for a new episode."""
        self._step_count = 0
        self._episode_done = False
        self._initial_ee_pos = None

    # ------------------------------------------------------------------
    # Phase methods
    # ------------------------------------------------------------------

    def _phase_approach_hover(self, cube_pos_w, num_envs, device):
        hover_target = cube_pos_w.clone()
        hover_target[:, 0] += _EE_X_OFFSET
        hover_target[:, 1] += _EE_Y_OFFSET
        hover_target[:, 2] += 0.08 + _GRIPPER_OFFSET
        alpha = self._step_count / _APPROACH_STEPS
        if self._initial_ee_pos is not None:
            target_pos_w = (1.0 - alpha) * self._initial_ee_pos + alpha * hover_target
        else:
            target_pos_w = hover_target
        return target_pos_w, torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)

    def _phase_hover_above_cube(self, cube_pos_w, num_envs, device):
        target_pos_w = cube_pos_w.clone()
        target_pos_w[:, 0] += _EE_X_OFFSET
        target_pos_w[:, 1] += _EE_Y_OFFSET
        target_pos_w[:, 2] += 0.08 + _GRIPPER_OFFSET
        return target_pos_w, torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)

    def _phase_lower_to_cube(self, cube_pos_w, num_envs, device):
        target_pos_w = cube_pos_w.clone()
        target_pos_w[:, 0] += _EE_X_OFFSET
        target_pos_w[:, 1] += _EE_Y_OFFSET
        target_pos_w[:, 2] += _GRIPPER_OFFSET
        return target_pos_w, torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)

    def _phase_grasp(self, cube_pos_w, num_envs, device):
        target_pos_w = cube_pos_w.clone()
        target_pos_w[:, 0] += _EE_X_OFFSET
        target_pos_w[:, 1] += _EE_Y_OFFSET
        target_pos_w[:, 2] += _GRIPPER_OFFSET
        return target_pos_w, torch.full((num_envs, 1), _GRIPPER_CLOSE, device=device)

    def _phase_lift_cube(self, cube_pos_w, num_envs, device):
        target_pos_w = cube_pos_w.clone()
        target_pos_w[:, 0] += _EE_X_OFFSET
        target_pos_w[:, 1] += _EE_Y_OFFSET
        target_pos_w[:, 2] += 0.30
        return target_pos_w, torch.full((num_envs, 1), _GRIPPER_CLOSE, device=device)

    def _phase_hold_cube(self, cube_pos_w, num_envs, device):
        target_pos_w = cube_pos_w.clone()
        target_pos_w[:, 0] += _EE_X_OFFSET
        target_pos_w[:, 1] += _EE_Y_OFFSET
        target_pos_w[:, 2] += 0.30
        return target_pos_w, torch.full((num_envs, 1), _GRIPPER_CLOSE, device=device)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_episode_done(self) -> bool:
        return self._episode_done

    @property
    def step_count(self) -> int:
        return self._step_count
