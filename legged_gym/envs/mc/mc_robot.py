import torch

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math import get_scale_shift
from .mc_config import MCRoughCfg


class MC(LeggedRobot):
    cfg: MCRoughCfg

    def _init_buffers(self):
        super()._init_buffers()

        # Identify semantic DOF groups by name rather than relying on hard-coded
        # indices. Isaac Gym may reorder URDF joints during asset import.
        self.wheel_indices = torch.tensor(
            [i for i, name in enumerate(self.dof_names) if "FOOT_JOINT" in name],
            dtype=torch.long,
            device=self.device,
        )
        self.abad_indices = torch.tensor(
            [i for i, name in enumerate(self.dof_names) if "ABAD_JOINT" in name],
            dtype=torch.long,
            device=self.device,
        )
        self.hip_indices = torch.tensor(
            [i for i, name in enumerate(self.dof_names) if "HIP_JOINT" in name],
            dtype=torch.long,
            device=self.device,
        )
        self.knee_indices = torch.tensor(
            [i for i, name in enumerate(self.dof_names) if "KNEE_JOINT" in name],
            dtype=torch.long,
            device=self.device,
        )

        expected = {
            "wheel": self.wheel_indices,
            "abad": self.abad_indices,
            "hip": self.hip_indices,
            "knee": self.knee_indices,
        }
        for group_name, indices in expected.items():
            if indices.numel() != 4:
                raise RuntimeError(
                    f"MC expected 4 {group_name} joints, found {indices.numel()}: "
                    f"DOFs={self.dof_names}"
                )

        print("###MC wheel_indices:", self.wheel_indices.tolist())
        print("###MC abad_indices:", self.abad_indices.tolist())
        print("###MC hip_indices:", self.hip_indices.tolist())
        print("###MC knee_indices:", self.knee_indices.tolist())

    def _wheel_masked_dof_vel(self):
        """Return a temporary velocity tensor for regularizers.

        Rolling speed is useful to the policy and therefore remains in the normal
        observation. It is masked only for the leg-velocity regularizer.
        """
        dof_vel = self.dof_vel.clone()
        dof_vel[:, self.wheel_indices] = 0.0
        return dof_vel

    def _build_current_obs(self):
        """Build one MC observation frame without mutating simulator state.

        Wheel absolute angle has no useful state meaning for a continuously
        rolling wheel, so only its position error is masked. Wheel velocity is
        intentionally retained. Privileged observations keep the same layout as
        the Go2W HIMLoco baseline, including 187 terrain-height samples when
        height sensing is enabled.
        """
        dof_err = (self.dof_pos - self.default_dof_pos).clone()
        dof_err[:, self.wheel_indices] = 0.0

        current_obs = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            dof_err * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
        ), dim=-1)

        # Actor-side observation noise (57 dimensions for the 16-DOF MC).
        if self.add_noise:
            current_obs += (
                (2 * torch.rand_like(current_obs) - 1)
                * self.noise_scale_vec[0:(9 + 3 * self.num_actions)]
            )

        # Privileged information: base linear velocity and external disturbance.
        current_obs = torch.cat((
            current_obs,
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.disturbance[:, 0, :],
        ), dim=-1)

        # Preserve the original HIMLoco/Go2W critic layout on rough terrain.
        # measured_points_x (17) * measured_points_y (11) = 187 samples.
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                -1,
                1,
            ) * self.obs_scales.height_measurements

            if self.add_noise:
                height_noise_start = 9 + 3 * self.num_actions
                height_noise_end = height_noise_start + heights.shape[1]
                heights += (
                    (2 * torch.rand_like(heights) - 1)
                    * self.noise_scale_vec[height_noise_start:height_noise_end]
                )

            current_obs = torch.cat((current_obs, heights), dim=-1)

        # Four wheel contact-force vectors: 4 * xyz = 12 dimensions.
        contact_forces_scale, contact_forces_shift = get_scale_shift(
            self.cfg.normalization.contact_force_range
        )
        contact_forces = (
            self.contact_forces[:, self.feet_indices, :].reshape(self.num_envs, -1)
            - contact_forces_shift
        ) * contact_forces_scale
        current_obs = torch.cat((current_obs, contact_forces), dim=-1)

        if current_obs.shape[1] != self.num_one_step_privileged_obs:
            raise RuntimeError(
                "MC privileged-observation size mismatch: "
                f"built {current_obs.shape[1]}, configured "
                f"{self.num_one_step_privileged_obs}."
            )

        return current_obs

    def compute_observations(self):
        """Update actor history and the current privileged critic observation."""
        current_obs = self._build_current_obs()

        self.obs_buf = torch.cat((
            current_obs[:, :self.num_one_step_obs],
            self.obs_buf[:, :-self.num_one_step_obs],
        ), dim=-1)

        self.privileged_obs_buf = torch.cat((
            current_obs[:, :self.num_one_step_privileged_obs],
            self.privileged_obs_buf[:, :-self.num_one_step_privileged_obs],
        ), dim=-1)

    def get_current_obs(self):
        """Return one complete actor+privileged observation frame safely."""
        return self._build_current_obs()

    def compute_termination_observations(self, env_ids):
        """Build terminal critic observations without modifying DOF state."""
        current_obs = self._build_current_obs()
        termination_obs = torch.cat((
            current_obs[:, :self.num_one_step_privileged_obs],
            self.privileged_obs_buf[:, :-self.num_one_step_privileged_obs],
        ), dim=-1)
        return termination_obs[env_ids]

    def _reward_dof_vel(self):
        """Penalize leg velocity while ignoring normal wheel rolling velocity."""
        dof_vel = self._wheel_masked_dof_vel()
        return torch.sum(torch.square(dof_vel), dim=1)

    def _reward_hip_default(self):
        """Penalize only MC HIP_JOINT displacement from its nominal pose."""
        hip_err = torch.sum(
            (
                self.dof_pos[:, self.hip_indices]
                - self.default_dof_pos[:, self.hip_indices]
            ) ** 2,
            dim=1,
        )
        return hip_err
