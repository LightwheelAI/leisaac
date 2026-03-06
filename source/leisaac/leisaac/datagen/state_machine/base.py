"""Abstract base class for task state machines in LeIsaac."""

from abc import ABC, abstractmethod

import torch


class StateMachineBase(ABC):
    """Abstract base class for task state machines in LeIsaac.

    A state machine encapsulates the step-by-step control logic for a task.
    It is designed to be decoupled from the simulation control flow: the caller
    is responsible for calling :meth:`get_action`, stepping the environment, and
    then calling :meth:`advance` to progress the internal state.

    Typical usage::

        sm = MyStateMachine()
        env.reset()
        sm.setup(env)
        while not sm.is_episode_done:
            sm.pre_step(env)
            actions = sm.get_action(env)
            env.step(actions)
            sm.advance()
        success = sm.check_success(env)
        sm.reset()
    """

    @abstractmethod
    def setup(self, env) -> None:
        """One-time setup after the environment is created.

        Use this for any calibration that requires a live env (e.g. FK
        calibration to compute rest-pose EE positions). Called once before
        the main recording loop starts.
        """
        raise NotImplementedError

    @abstractmethod
    def check_success(self, env) -> bool:
        """Evaluate whether the current episode was successful.

        Called at episode end (when :attr:`is_episode_done` is ``True``)
        before resetting. The implementation may temporarily teleport joints
        to a canonical pose for evaluation.
        """
        raise NotImplementedError

    def pre_step(self, env) -> None:
        """Optional per-step hook called before :meth:`get_action` and ``env.step``.

        Override to inject direct joint-state writes or other per-step
        overrides that must happen before the action is applied (e.g. blended
        home-pose control).  Default implementation is a no-op.
        """

    @abstractmethod
    def get_action(self, env) -> torch.Tensor:
        """Compute and return the action tensor for the current step.

        This method does **not** advance the internal state counter.
        Call :meth:`advance` after :meth:`env.step` to progress the machine.

        Args:
            env: The simulation environment instance. Must expose ``env.device``,
                ``env.num_envs``, and ``env.scene``.

        Returns:
            Action tensor of shape ``(num_envs, action_dim)``.
        """
        raise NotImplementedError

    @abstractmethod
    def advance(self) -> None:
        """Advance the internal step counter and manage state transitions.

        Should be called exactly once after each :meth:`env.step` call.
        Internally handles multi-phase and multi-object transitions,
        including any fast-forward optimisations.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset the state machine to its initial state.

        Should be called before starting a new episode.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def is_episode_done(self) -> bool:
        """Whether the state machine has completed a full episode cycle.

        Returns:
            ``True`` once the state machine has finished all phases of the task.
        """
        raise NotImplementedError
