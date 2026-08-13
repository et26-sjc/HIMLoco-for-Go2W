"""Quiet-evaluation wrapper for the contact-buffered 100 Hz MC controller."""

from .quiet_mc_v2_robot import QuietMCV2
from .mc_buffered_100hz_robot import MCBuffered100Hz


class QuietMCBuffered100HzV2(QuietMCV2, MCBuffered100Hz):
    """V2 quiet instrumentation + contact-buffered torque computation.

    The MRO intentionally selects QuietMC's physics-rate evaluation step while
    resolving ``_compute_torques`` through MCBuffered100Hz.  QuietMC already
    refreshes contact forces after every physics substep, so the buffer detector
    receives 200 Hz contact information during evaluation.
    """

    pass
