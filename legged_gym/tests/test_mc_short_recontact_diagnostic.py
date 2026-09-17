"""CPU checks for the diagnostic-only short re-contact event counter."""

from types import SimpleNamespace

import isaacgym  # noqa: F401; must precede torch
import torch

from legged_gym.envs.mc.mc_learned_admittance_100hz_robot import (
    MCLearnedAdmittance100Hz,
)


def _event_counter():
    counter = SimpleNamespace(
        sim_params=SimpleNamespace(dt=0.005),
        gt_short_contact_state=torch.zeros((1, 1), dtype=torch.bool),
        gt_short_release_age=torch.zeros((1, 1), dtype=torch.long),
        gt_step_short_recontact=torch.zeros((1, 1)),
        gt_step_touchdown=torch.zeros((1, 1)),
    )

    def advance(force):
        MCLearnedAdmittance100Hz._update_gt_short_recontact_substep(
            counter, torch.tensor([[float(force)]])
        )

    return counter, advance


def test_short_recontact_window_crosses_policy_step_boundary():
    counter, advance = _event_counter()
    for force in (0, 0, 6):
        advance(force)
    assert counter.gt_step_touchdown.item() == 1
    assert counter.gt_step_short_recontact.item() == 0
    advance(3)  # hysteresis keeps contact active above the 2 N off threshold
    advance(0)
    counter.gt_step_short_recontact.zero_()
    advance(0)
    advance(6)
    assert counter.gt_step_short_recontact.item() == 1


def test_exact_20ms_boundary_and_reset_state():
    counter, advance = _event_counter()
    advance(6)
    advance(0)
    for _ in range(3):
        advance(0)
    advance(6)
    assert counter.gt_step_short_recontact.item() == 1

    counter.gt_short_contact_state.zero_()
    counter.gt_short_release_age.zero_()
    counter.gt_step_short_recontact.zero_()
    advance(6)
    assert counter.gt_step_short_recontact.item() == 0
