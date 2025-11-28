import isaaclab.utils.math as math_utils
import torch


def rotvec_to_euler(rotvec: torch.Tensor) -> torch.Tensor:
    """Convert a batch of rotation vectors (axis-angle) into Euler XYZ deltas."""
    # |rotvec| gives the rotation magnitude for each environment (shape: [N, 1])
    rotvec_norm = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    rotvec_norm_clamped = torch.clamp(rotvec_norm, min=1.0e-8)
    axis = rotvec / rotvec_norm_clamped

    # when norm ~ 0, the axis direction is ill-defined; fall back to +X
    default_axis = torch.tensor([1.0, 0.0, 0.0], device=rotvec.device, dtype=axis.dtype).view(1, 3)
    axis = torch.where(rotvec_norm > 1.0e-8, axis, default_axis.repeat(rotvec.shape[0], 1))

    delta_quat = math_utils.quat_from_angle_axis(rotvec_norm.squeeze(-1), axis)
    delta_roll, delta_pitch, delta_yaw = math_utils.euler_xyz_from_quat(delta_quat)
    return torch.cat([delta_roll, delta_pitch, delta_yaw], dim=0)
