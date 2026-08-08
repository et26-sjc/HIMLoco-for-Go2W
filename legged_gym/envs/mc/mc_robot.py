import torch

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math import get_scale_shift
from .mc_config import MCRoughCfg


class MC(LeggedRobot):
    cfg: MCRoughCfg

    def compute_observations(self):
        """MC observation.

        Wheel absolute rotation is not useful for policy input, but the simulator
        state must never be modified in-place. Use temporary tensors instead.
        """
        dof_pos_obs = self.dof_pos.clone()
        dof_err = dof_pos_obs - self.default_dof_pos

        dof_err[:, self.wheel_indices] = 0.0
        dof_pos_obs[:, self.wheel_indices] = 0.0

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
