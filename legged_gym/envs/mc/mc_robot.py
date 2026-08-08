import torch

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math import get_scale_shift
from .mc_config import MCRoughCfg


class MC(LeggedRobot):
    cfg: MCRoughCfg

    def _init_buffers(self):
        super()._init_buffers()

        # Identify wheel DOFs by joint name rather than relying on fixed indices.
        # This keeps the MC implementation robust to URDF reordering.
        wheel_ids = [i for i, n in enumerate(self.dof_names)
                     if "FOOT_JOINT" in n]
        self.wheel_indices = torch.tensor(
            wheel_ids, dtype=torch.long, device=self.device)

        self.abad_indices = torch.tensor(
            [i for i, n in enumerate(self.dof_names) if "ABAD" in n],
            dtype=torch.long, device=self.device)
        self.hip_indices = torch.tensor(
            [i for i, n in enumerate(self.dof_names) if "HIP" in n],
            dtype=torch.long, device=self.device)
        self.knee_indices = torch.tensor(
            [i for i, n in enumerate(self.dof_names) if "KNEE" in n],
            dtype=torch.long, device=self.device)

    def _wheel_masked_dof_pos(self):
        dof_pos = self.dof_pos.clone()
        dof_pos[:, self.wheel_indices] = 0.0
        return dof_pos

    def _wheel_masked_dof_vel(self):
        dof_vel = self.dof_vel.clone()
        dof_vel[:, self.wheel_indices] = 0.0
        return dof_vel

    def compute_observations(self):
        """Build policy observations without modifying Isaac Gym state."""
        dof_pos_obs = self._wheel_masked_dof_pos()
        dof_err = dof_pos_obs - self.default_dof_pos

        dof_err[:, self.wheel_indices] = 0.0

        current_obs = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            dof_err * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions
        ), dim=-1)

        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:(9 + 3 * self.num_actions)]

        current_obs = torch.cat((
            current_obs,
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.disturbance[:, 0, :]
        ), dim=-1)

        contact_forces_scale, contact_forces_shift = get_scale_shift(self.cfg.normalization.contact_force_range)
        contact_forces = (self.contact_forces[:, self.feet_indices, :].reshape(self.num_envs, -1)
                          - contact_forces_shift) * contact_forces_scale
        current_obs = torch.cat((current_obs, contact_forces), dim=-1)

        self.obs_buf = torch.cat((current_obs[:, :self.num_one_step_obs],
                                  self.obs_buf[:, :-self.num_one_step_obs]), dim=-1)
        self.privileged_obs_buf = torch.cat((current_obs[:, :self.num_one_step_privileged_obs],
                                             self.privileged_obs_buf[:, :-self.num_one_step_privileged_obs]), dim=-1)

    def _reward_dof_vel(self):
        """Penalize leg velocity but ignore wheel rolling velocity."""
        dof_vel = self._wheel_masked_dof_vel()
        return torch.sum(torch.square(dof_vel), dim=1)

    def _reward_hip_default(self):
        """MC hip reward: only the three leg joints are posture joints.

        FOOT_JOINT is a wheel rotational DOF and ABAD is lateral articulation,
        therefore neither should be included in this regularizer.
        """
        hip_err = torch.sum(
            (self.dof_pos[:, self.hip_indices] -
             self.default_dof_pos[:, self.hip_indices]) ** 2,
            dim=1)
        return hip_err
