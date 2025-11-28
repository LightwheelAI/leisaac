"""This file contains temporary fixes for IsaacLab. These issues are typically resolved in IsaacLab version updates,
at which point the corresponding temporary patches should be removed.
To keep leisaac's version dependencies clearer, we choose to retain these temporary fixes until the official IsaacLab
version is released."""


def patch_termination_manager():
    """
    Temporary fix for TerminationManager.compute() method to correctly store each termination condition's result
    in _term_dones when computing termination signals.
    In the current implementation, once self._term_dones is set to True, it cannot be modified back to False.
    This issue was fixed in https://github.com/isaac-sim/IsaacLab/commit/f4982455cb4c3d30e40703fb43b257df18dce2e3.
    TODO: Remove this patch after that commit is officially released.
    """
    import torch
    from isaaclab.managers import TerminationManager

    def compute(self) -> torch.Tensor:
        # reset computation
        self._truncated_buf[:] = False
        self._terminated_buf[:] = False
        # iterate over all the termination terms
        for i, term_cfg in enumerate(self._term_cfgs):
            value = term_cfg.func(self._env, **term_cfg.params)
            # store timeout signal separately
            if term_cfg.time_out:
                self._truncated_buf |= value
            else:
                self._terminated_buf |= value
            # add to episode dones
            self._term_dones[:, i] = value  # [core fix]
            rows = value.nonzero(as_tuple=True)[0]  # indexing is cheaper than boolean advance indexing
            if rows.numel() > 0:
                self._term_dones[rows] = False
                self._term_dones[rows, i] = True
        # return combined termination signal
        return self._truncated_buf | self._terminated_buf
    TerminationManager.compute = compute


def monkey_patch():
    patch_termination_manager()


monkey_patch()
