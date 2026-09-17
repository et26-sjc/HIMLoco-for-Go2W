from types import SimpleNamespace

import isaacgym  # noqa: F401  # must be imported before torch
import torch

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.mc.mc_learned_admittance_100hz_robot import (
    MCLearnedAdmittance100Hz,
)
from legged_gym.envs.mc.mc_learned_admittance_100hz_config import (
    MCLearnedAdmittance100HzCfg,
)
from legged_gym.envs.mc.mc_learned_admittance import MCLearnedAdmittance


class _ZeroAdmittance:
    def step(
        self,
        compliance_actions,
        estimated_contact,
        q_nominal,
        hip_indices,
        knee_indices,
        dt,
    ):
        return torch.zeros(
            q_nominal.shape[0],
            hip_indices.numel(),
            2,
            dtype=q_nominal.dtype,
            device=q_nominal.device,
        )


class _FixedResidualAdmittance:
    def __init__(self, residual):
        self.residual = residual

    def step(self, *args):
        return self.residual


def _controller(admittance):
    controller = SimpleNamespace()
    controller.cfg = SimpleNamespace(
        control=SimpleNamespace(
            action_scale=0.5,
            vel_scale=2.0,
            control_type="P",
        )
    )
    controller.default_dof_pos = torch.linspace(-0.35, 0.35, 16).repeat(2, 1)
    controller.dof_pos = torch.linspace(-0.2, 0.2, 32).reshape(2, 16)
    controller.dof_vel = torch.linspace(0.3, -0.3, 32).reshape(2, 16)
    controller.p_gains = torch.linspace(20.0, 35.0, 16)
    controller.d_gains = torch.linspace(0.5, 1.2, 16)
    controller.Kp_factors = torch.tensor([[1.0], [0.9]])
    controller.Kd_factors = torch.tensor([[1.0], [1.1]])
    controller.torque_limits = torch.full((16,), 100.0)
    controller.wheel_indices = torch.tensor([3, 7, 11, 15])
    controller.adm_hip_indices = torch.tensor([5, 1, 9, 13])
    controller.adm_knee_indices = torch.tensor([6, 2, 10, 14])
    controller.dof_pos_limits = torch.tensor([[-0.4, 0.4]] * 16)
    controller.sim_params = SimpleNamespace(dt=0.005)
    controller.admittance = admittance
    return controller


def test_zero_compliance_torque_exactly_matches_baseline_even_outside_limits():
    controller = _controller(_ZeroAdmittance())
    actions = torch.linspace(-2.0, 2.0, 32).reshape(2, 16)
    compliance = torch.zeros(2, 4)
    estimate = torch.zeros(2, 8)

    expected = LeggedRobot._compute_torques(controller, actions.clone())
    actual = MCLearnedAdmittance100Hz._compute_adaptive_torques(
        controller, actions.clone(), compliance, estimate
    )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_real_admittance_zero_endpoint_is_exact_baseline():
    admittance = MCLearnedAdmittance(
        MCLearnedAdmittance100HzCfg().learned_admittance, 2, "cpu"
    )
    controller = _controller(admittance)
    actions = torch.linspace(-2.0, 2.0, 32).reshape(2, 16)
    compliance = torch.zeros(2, 4)
    estimate = torch.tensor(
        [
            [2.0, 1.0, 0.5, 3.0, 4.0, -2.0, 1.0, 0.0],
            [1.0, 2.0, 3.0, 0.5, -1.0, 3.0, 0.0, 2.0],
        ]
    )

    expected = LeggedRobot._compute_torques(controller, actions.clone())
    actual = MCLearnedAdmittance100Hz._compute_adaptive_torques(
        controller, actions.clone(), compliance, estimate
    )

    torch.testing.assert_close(admittance.alpha, torch.zeros(2, 4))
    torch.testing.assert_close(admittance.effective_alpha, torch.zeros(2, 4))
    torch.testing.assert_close(admittance.drive_force, torch.zeros(2, 4))
    torch.testing.assert_close(admittance.delta_l, torch.zeros(2, 4))
    torch.testing.assert_close(admittance.last_joint_offsets, torch.zeros(2, 4, 2))
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_compliance_residual_cannot_increase_nominal_limit_violation():
    residual = torch.zeros(2, 4, 2)
    residual[:, :, 0] = torch.tensor([-0.3, 0.3, -0.3, 0.3])
    residual[:, :, 1] = torch.tensor([-0.3, 0.3, -0.3, 0.3])
    controller = _controller(_FixedResidualAdmittance(residual))
    actions = torch.zeros(2, 16)
    # Put two nominal targets outside opposite limits.
    actions[:, 5] = -2.0
    actions[:, 1] = 2.0

    MCLearnedAdmittance100Hz._compute_adaptive_torques(
        controller, actions, torch.ones(2, 4), torch.zeros(2, 8)
    )

    # The exact target is easiest to recover before PD gains by setting q/dq=0
    # and dividing the unclipped torque by Kp. Torques remain below the limit.
    controller.dof_pos.zero_()
    controller.dof_vel.zero_()
    torque = MCLearnedAdmittance100Hz._compute_adaptive_torques(
        controller, actions, torch.ones(2, 4), torch.zeros(2, 8)
    )
    target = torque / (controller.p_gains * controller.Kp_factors)
    nominal = controller.default_dof_pos + 0.5 * actions
    torch.testing.assert_close(target[:, 5], nominal[:, 5])
    torch.testing.assert_close(target[:, 1], nominal[:, 1])
    for index in [2, 6, 9, 10, 13, 14]:
        assert torch.all(target[:, index] >= -0.4)
        assert torch.all(target[:, index] <= 0.4)
