"""Quiet-evaluation wrapper for the 100 Hz MC axial-admittance controller."""

from .quiet_mc_v2_robot import QuietMCV2
from .mc_admittance_100hz_robot import MCAdmittance100Hz


class QuietMCAdmittance100HzV2(QuietMCV2, MCAdmittance100Hz):
    """V2 quiet instrumentation plus the axial-admittance torque target.

    QuietMCV2 supplies deterministic resets and physics-rate quiet metrics.
    MCAdmittance100Hz supplies the modified ``_compute_torques`` method.  The
    shared quiet step already refreshes contact/body states every physics
    substep, so the admittance state runs at the 200 Hz physics rate.
    """

    def get_quiet_metrics(self):
        summary = super().get_quiet_metrics()
        summary.update(
            {
                "controller": self.quiet_controller_name,
                "admittance_virtual_mass_kg": self.admittance_mass,
                "admittance_virtual_damping_ns_per_m": self.admittance_damping,
                "admittance_virtual_stiffness_n_per_m": self.admittance_stiffness,
                "admittance_force_bias_tau_s": self.admittance_force_bias_tau,
                "admittance_force_deadband_n": self.admittance_force_deadband,
                "admittance_max_force_input_n": self.admittance_max_force_input,
                "admittance_max_compression_m": self.admittance_max_compression,
                "admittance_max_compression_velocity_mps": self.admittance_max_compression_vel,
                "admittance_jacobian_damping": self.admittance_jacobian_damping,
            }
        )
        return summary
