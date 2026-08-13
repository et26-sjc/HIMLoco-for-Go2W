"""Deterministic quiet-evaluation wrapper for the MC wheeled quadruped.

This v2 wrapper keeps the trained policy, reward and controller unchanged while
improving evaluation repeatability and the leg-compression measurement:

* resets use exact nominal joint positions and zero base velocity;
* custom-origin resets no longer add random +/-1 m XY offsets;
* the quiet "leg compression" signal is the change in hip-to-wheel geometric
  distance rather than wheel-center Z relative to the base.

It inherits all physics-rate impact / vibration instrumentation from QuietMC.
"""

from isaacgym import gymtorch
import torch

from .quiet_mc_robot import QuietMC


class QuietMCV2(QuietMC):
    """QuietMC with deterministic reset and geometric leg compression."""

    _HIP_BODY_NAMES = [
        "FR_HIP_LINK",
        "FL_HIP_LINK",
        "RR_HIP_LINK",
        "RL_HIP_LINK",
    ]

    def _init_quiet_metrics(self):
        self.hip_body_indices = torch.tensor(
            [
                self.gym.find_actor_rigid_body_handle(
                    self.envs[0], self.actor_handles[0], name
                )
                for name in self._HIP_BODY_NAMES
            ],
            dtype=torch.long,
            device=self.device,
        )
        if torch.any(self.hip_body_indices < 0):
            raise RuntimeError(
                "QuietMCV2 could not resolve all MC hip rigid bodies: "
                f"{self._HIP_BODY_NAMES} -> {self.hip_body_indices.tolist()}"
            )
        if self.hip_body_indices.numel() != self.feet_indices.numel():
            raise RuntimeError(
                "QuietMCV2 hip/wheel body count mismatch: "
                f"hips={self.hip_body_indices.numel()}, "
                f"wheels={self.feet_indices.numel()}"
            )
        super()._init_quiet_metrics()

    def _wheel_kinematics(self):
        """Return wheel velocities and -hip-to-wheel length.

        QuietMC treats the third return value as a scalar that increases when a
        leg compresses during a contact event.  Returning ``-leg_length`` makes
        the inherited expression

            current_scalar - touchdown_scalar

        equal to

            leg_length_at_touchdown - current_leg_length,

        i.e. positive geometric compression of the whole leg.
        """
        states = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
        wheel_pos_world = states[:, self.feet_indices, 0:3]
        hip_pos_world = states[:, self.hip_body_indices, 0:3]
        wheel_vel_world = states[:, self.feet_indices, 7:10]
        wheel_vel_body = self._to_base_frame(wheel_vel_world)
        leg_length = torch.norm(wheel_pos_world - hip_pos_world, dim=-1)
        return wheel_vel_world, wheel_vel_body, -leg_length

    def _reset_dofs(self, env_ids):
        """Reset exactly to the nominal MC configuration for reproducible eval."""
        if len(env_ids) == 0:
            return
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_pos[env_ids[:, None], self.wheel_indices[None, :]] = 0.0
        self.dof_vel[env_ids] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_root_states(self, env_ids):
        """Reset to the terrain origin with zero linear/angular velocity."""
        if len(env_ids) == 0:
            return
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, 7:13] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
