"""State machine for the pick-orange task."""

import math

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_inv, quat_mul

from .base import StateMachineBase
from leisaac.tasks.pick_orange.mdp import task_done

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSE = -1.0
_GRIPPER_OFFSET = 0.1  # vertical clearance for the gripper tip
_APPROACH_STEPS: int = 120  # steps to smoothly interpolate from init EE pos to hover (first orange only)

_REST_POSE_DEG = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -100.0,
    "elbow_flex": 90.0,
    "wrist_flex": 50.0,
    "wrist_roll": 0.0,
    "gripper": -10.0,
}


def _apply_triangle_offset(pos_tensor: torch.Tensor, orange_now: int, radius: float = 0.1) -> torch.Tensor:
    """Apply an equilateral-triangle offset on the x-y plane."""
    idx = (orange_now - 1) % 3
    angle = idx * (2 * math.pi / 3)
    pos_tensor[:, 0] += radius * math.cos(angle)
    pos_tensor[:, 1] += radius * math.sin(angle)
    return pos_tensor


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class PickOrangeStateMachine(StateMachineBase):
    """State machine for the pick-orange manipulation task.

    The robot cycles through *num_oranges* oranges.  For each orange it
    executes a fixed sequence of steps that moves the gripper above the
    orange, grasps it, lifts it, transports it to the plate and places it.

    Args:
        num_oranges: Total number of oranges to pick and place. Defaults to 3.
    """

    MAX_STEPS_PER_ORANGE: int = 980

    def __init__(self, num_oranges: int = 3) -> None:
        self._num_oranges = num_oranges
        self._step_count: int = 0
        self._orange_now: int = 1
        self._episode_done: bool = False
        self._initial_ee_pos: torch.Tensor | None = None
        self._rest_ee_pos_world: torch.Tensor | None = None
        self._rest_joint_pos: torch.Tensor | None = None
        self._home_start_pos: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # StateMachineBase interface
    # ------------------------------------------------------------------

    def setup(self, env) -> None:
        """FK calibration: drive arm to rest pose and record the EE world position.

        Teleports joints to the SO-101 rest pose, steps the simulation once
        to propagate kinematics, then reads ``body_pos_w`` to get the EE world
        position that corresponds to the rest pose.  This position is used as
        the home target during the return-home phase so that ``task_done()``
        can verify the arm is within the expected joint-angle tolerances.
        """
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
        print(f"[Calibration] Rest pose EE (world): {self._rest_ee_pos_world[0].cpu().numpy()}")

        # Return scene to default initial state before the recording loop.
        env.reset()

    def check_success(self, env) -> bool:
        """Return True if all oranges are on the plate and the arm is at rest."""
        robot = env.scene["robot"]
        if self._rest_joint_pos is not None:
            robot.write_joint_state_to_sim(
                position=self._rest_joint_pos,
                velocity=torch.zeros_like(self._rest_joint_pos),
            )
            env.scene.update(dt=env.physics_dt)

        success_tensor = task_done(
            env,
            oranges_cfg=[
                SceneEntityCfg("Orange001"),
                SceneEntityCfg("Orange002"),
                SceneEntityCfg("Orange003"),
            ],
            plate_cfg=SceneEntityCfg("Plate"),
        )
        return bool(success_tensor.all().item())

    def pre_step(self, env) -> None:
        """Blend joint state toward rest pose during the final orange's home phase.

        Only active when ``orange_now == num_oranges`` and ``step_count >= 680``.
        Writes blended joint positions directly to the sim so the arm smoothly
        returns home while IK still receives the rest-pose EE target.
        """
        if (
            self._orange_now == self._num_oranges
            and self._step_count >= 680
            and self._rest_joint_pos is not None
        ):
            robot = env.scene["robot"]
            if self._step_count == 680:
                self._home_start_pos = robot.data.joint_pos.clone()
            if self._home_start_pos is not None:
                alpha = min((self._step_count - 680) / 299.0, 1.0)
                blended = self._home_start_pos + (self._rest_joint_pos - self._home_start_pos) * alpha
                robot.write_joint_state_to_sim(position=blended, velocity=torch.zeros_like(blended))

    def get_action(self, env) -> torch.Tensor:
        """Compute the action tensor for the current step (8D IK pose target)."""
        robot = env.scene["robot"]
        robot.write_joint_damping_to_sim(damping=10.0)

        device = env.device
        num_envs = env.num_envs
        step = self._step_count

        orange_pos_w = env.scene[f"Orange00{self._orange_now}"].data.root_pos_w.clone()
        plate_pos_w = env.scene["Plate"].data.root_pos_w.clone()
        robot_base_pos_w = robot.data.root_pos_w.clone()
        robot_base_quat_w = robot.data.root_quat_w.clone()

        target_quat_w = quat_from_euler_xyz(
            torch.tensor(0.0, device=device),
            torch.tensor(0.0, device=device),
            torch.tensor(0.0, device=device),
        ).repeat(num_envs, 1)
        target_quat = quat_mul(quat_inv(robot_base_quat_w), target_quat_w)

        if self._orange_now == 1 and step == 0:
            self._initial_ee_pos = robot.data.body_pos_w[:, -1, :].clone()

        if self._orange_now == 1 and step < _APPROACH_STEPS:
            target_pos_w, gripper_cmd = self._phase_approach_hover(orange_pos_w, num_envs, device)
        elif step < 180:
            target_pos_w, gripper_cmd = self._phase_move_above_orange(orange_pos_w, num_envs, device)
        elif step < 300:
            target_pos_w, gripper_cmd = self._phase_hover_above_orange(orange_pos_w, num_envs, device)
        elif step < 360:
            target_pos_w, gripper_cmd = self._phase_lower_to_orange(orange_pos_w, num_envs, device)
        elif step < 420:
            target_pos_w, gripper_cmd = self._phase_grasp(orange_pos_w, num_envs, device)
        elif step < 500:
            target_pos_w, gripper_cmd = self._phase_lift_orange(orange_pos_w, num_envs, device)
        elif step < 550:
            target_pos_w, gripper_cmd = self._phase_move_above_plate(plate_pos_w, num_envs, device)
        elif step < 600:
            target_pos_w, gripper_cmd = self._phase_lower_to_plate(plate_pos_w, num_envs, device)
        elif step < 640:
            target_pos_w, gripper_cmd = self._phase_release(plate_pos_w, num_envs, device)
        elif step < 680:
            target_pos_w, gripper_cmd = self._phase_lift_gripper(plate_pos_w, num_envs, device)
        else:
            target_pos_w, gripper_cmd = self._phase_return_home(num_envs, device)

        diff_w = target_pos_w - robot_base_pos_w
        target_pos_local = quat_apply(quat_inv(robot_base_quat_w), diff_w)
        return torch.cat([target_pos_local, target_quat, gripper_cmd], dim=-1)

    def advance(self) -> None:
        """Advance step counter, handle orange transitions, and fast-forward home phase.

        For non-final oranges, the return-home phase (steps 680–979) is skipped
        without simulation — the arm goes straight to the next orange.
        """
        self._step_count += 1
        if self._step_count >= self.MAX_STEPS_PER_ORANGE:
            if self._orange_now >= self._num_oranges:
                self._episode_done = True
            else:
                self._orange_now += 1
                self._step_count = 0
        elif self._orange_now < self._num_oranges and self._step_count >= 680:
            # Fast-forward: skip the home phase for intermediate oranges.
            prev_orange = self._orange_now
            while self._orange_now == prev_orange and not self._episode_done:
                self._step_count += 1
                if self._step_count >= self.MAX_STEPS_PER_ORANGE:
                    self._orange_now += 1
                    self._step_count = 0

    def reset(self) -> None:
        """Reset the state machine to its initial state for a new episode."""
        self._step_count = 0
        self._orange_now = 1
        self._episode_done = False
        self._initial_ee_pos = None
        self._home_start_pos = None

    # ------------------------------------------------------------------
    # Phase methods
    # ------------------------------------------------------------------

    def _phase_approach_hover(self, orange_pos_w, num_envs, device):
        hover_target = orange_pos_w.clone()
        hover_target[:, 0] -= 0.03
        hover_target[:, 1] -= 0.01
        hover_target[:, 2] += 0.1 + _GRIPPER_OFFSET
        alpha = self._step_count / _APPROACH_STEPS
        if self._initial_ee_pos is not None:
            target_pos_w = (1.0 - alpha) * self._initial_ee_pos + alpha * hover_target
        else:
            target_pos_w = hover_target
        return target_pos_w, torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)

    def _phase_move_above_orange(self, orange_pos_w, num_envs, device):
        target_pos_w = orange_pos_w.clone()
        target_pos_w[:, 0] -= 0.03
        target_pos_w[:, 1] -= 0.01
        target_pos_w[:, 2] += 0.15 + _GRIPPER_OFFSET
        return target_pos_w, torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)

    def _phase_hover_above_orange(self, orange_pos_w, num_envs, device):
        target_pos_w = orange_pos_w.clone()
        target_pos_w[:, 0] -= 0.03
        target_pos_w[:, 1] -= 0.01
        target_pos_w[:, 2] += 0.1 + _GRIPPER_OFFSET
        return target_pos_w, torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)

    def _phase_lower_to_orange(self, orange_pos_w, num_envs, device):
        target_pos_w = orange_pos_w.clone()
        target_pos_w[:, 0] -= 0.03
        target_pos_w[:, 1] -= 0.01
        target_pos_w[:, 2] += _GRIPPER_OFFSET
        return target_pos_w, torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)

    def _phase_grasp(self, orange_pos_w, num_envs, device):
        target_pos_w = orange_pos_w.clone()
        target_pos_w[:, 0] -= 0.03
        target_pos_w[:, 1] -= 0.01
        target_pos_w[:, 2] += _GRIPPER_OFFSET
        return target_pos_w, torch.full((num_envs, 1), _GRIPPER_CLOSE, device=device)

    def _phase_lift_orange(self, orange_pos_w, num_envs, device):
        target_pos_w = orange_pos_w.clone()
        target_pos_w[:, 0] -= 0.03
        target_pos_w[:, 1] -= 0.01
        target_pos_w[:, 2] += 0.25
        return target_pos_w, torch.full((num_envs, 1), _GRIPPER_CLOSE, device=device)

    def _phase_move_above_plate(self, plate_pos_w, num_envs, device):
        target_pos_w = plate_pos_w.clone()
        target_pos_w[:, 2] += 0.25
        return target_pos_w, torch.full((num_envs, 1), _GRIPPER_CLOSE, device=device)

    def _phase_lower_to_plate(self, plate_pos_w, num_envs, device):
        target_pos_w = plate_pos_w.clone()
        target_pos_w[:, 2] += _GRIPPER_OFFSET + 0.1
        _apply_triangle_offset(target_pos_w, self._orange_now)
        return target_pos_w, torch.full((num_envs, 1), _GRIPPER_CLOSE, device=device)

    def _phase_release(self, plate_pos_w, num_envs, device):
        target_pos_w = plate_pos_w.clone()
        target_pos_w[:, 2] += _GRIPPER_OFFSET + 0.1
        _apply_triangle_offset(target_pos_w, self._orange_now)
        return target_pos_w, torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)

    def _phase_lift_gripper(self, plate_pos_w, num_envs, device):
        target_pos_w = plate_pos_w.clone()
        target_pos_w[:, 2] += 0.2
        _apply_triangle_offset(target_pos_w, self._orange_now)
        return target_pos_w, torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)

    def _phase_return_home(self, num_envs, device):
        if self._rest_ee_pos_world is not None:
            target_pos_w = self._rest_ee_pos_world.clone()
        elif self._initial_ee_pos is not None:
            target_pos_w = self._initial_ee_pos.clone()
        else:
            target_pos_w = torch.zeros(num_envs, 3, device=device)
        return target_pos_w, torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_episode_done(self) -> bool:
        return self._episode_done

    @property
    def orange_now(self) -> int:
        return self._orange_now

    @property
    def step_count(self) -> int:
        return self._step_count
