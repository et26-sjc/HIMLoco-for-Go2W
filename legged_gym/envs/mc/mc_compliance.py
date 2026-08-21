import torch


class MCContactCompliance:
    """Lightweight contact-aware leg compliance for wheel-legged robots.

    This module is intentionally separated from HIMLoco policy learning. It
    modifies only leg target references when large vertical contact impulses
    appear, providing a first-order passive compliance approximation.
    """

    def __init__(self, cfg):
        self.enable = getattr(cfg, "enable", False)
        self.force_threshold = getattr(cfg, "force_threshold", 150.0)
        self.stiffness = getattr(cfg, "stiffness", 3000.0)
        self.max_knee_offset = getattr(cfg, "max_knee_offset", 0.15)

    def compute_knee_offset(self, contact_forces, feet_indices):
        """Compute knee target offsets from vertical contact force.

        Fz > threshold generates knee flexion. The relation is equivalent to
        a virtual spring:

            Delta q = -(Fz-F0)/K

        and is clipped for stability.
        """
        if not self.enable:
            return None

        fz = contact_forces[:, feet_indices, 2]
        total_fz = torch.clamp(fz, min=0.0)
        excess = torch.mean(torch.clamp(total_fz - self.force_threshold, min=0.0), dim=1)

        offset = -excess / self.stiffness
        offset = torch.clamp(offset, -self.max_knee_offset, 0.0)
        return offset
