import torch


class MCContactCompliance:
    """Contact-aware knee compliance layer for MC wheel-legged locomotion.

    The module approximates a virtual spring:

        Delta q = -(Fz - F_threshold) / K

    It is intentionally separated from HIMLoco. The policy generates nominal
    joint targets, and this module only provides transient knee flexion when
    wheel-ground impact forces increase.
    """

    def __init__(self, cfg):
        self.enable = getattr(cfg, "enable", False)
        self.force_threshold = getattr(cfg, "force_threshold", 150.0)
        self.stiffness = getattr(cfg, "stiffness", 3000.0)
        self.max_knee_offset = getattr(cfg, "max_knee_offset", 0.15)

    def compute_knee_offset(self, contact_forces, feet_indices):
        """Return four wheel/knee compliance offsets.

        Negative offsets correspond to knee flexion, absorbing impact energy.
        """
        num_envs = contact_forces.shape[0]
        device = contact_forces.device

        if not self.enable:
            return torch.zeros(num_envs, 4, device=device)

        fz = torch.clamp(contact_forces[:, feet_indices, 2], min=0.0)
        excess = torch.clamp(fz - self.force_threshold, min=0.0)

        offset = -excess / self.stiffness
        return torch.clamp(offset, -self.max_knee_offset, 0.0)
