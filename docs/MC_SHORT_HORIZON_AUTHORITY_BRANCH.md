# MC Same-State Short-Horizon Authority Branch

## Research question

Does the physical compliance torque residual still have enough local causal
authority, immediately before contact release, to reduce upward release motion
and short re-contact? Or has the absorption phase already established the
release geometry before a late authority intervention can help?

This is an evaluation-only experiment on the existing replay-stabilized,
iteration-500 checkpoint. No estimator, replay, PPO, reward, impact mapping,
M/D/K, action/observation dimension, or production controller was changed.

## Why this experiment

Previous full-rollout counterfactuals were difficult to interpret because GPU
PhysX trajectories diverged and the reference intervention schedule agreed
with branched contact state only about 77% of the time. This experiment instead
restores one pre-release state, replays the same 50 ms policy and ContactEstimator
outputs, and changes only the target leg's physical compliance torque-residual
scale.

GPU PhysX does not expose contact-manifold or solver warm-start snapshots.
Therefore every experiment includes two independent, unchanged `1.0x` arms.
Intervention results are accepted only after this A/A reproducibility gate
passes.

## Protocol

* checkpoint: seed-1 replay-0.25 policy at iteration 500;
* scenario: down stairs, 0.14 m step height, 0.5 m/s command;
* source rollout: 64 environments, 2 s warm-up and 20 s collection;
* evaluation seeds: 123, 11 and 22;
* 96 sampled release events per seed, 288 events total;
* event eligibility: completed contact loading peak at least 5000 N/s and
  pre-release admittance compression at least 0.01 mm;
* branch state: root and DOF state, observation/action histories, commands,
  complete admittance state, and physics-rate quiet diagnostic state;
* branch horizon: 50 ms at 200 Hz;
* high-level motion action, compliance action, and ContactEstimator output are
  replayed identically in all arms;
* target-leg torque is
  `tau_PD + scale * (tau_adaptive - tau_PD)`, with the normal actuator torque
  limit retained.

The five arms are `A1=1.0`, `A2=1.0`, `half=0.5`, `zero=0.0`, and
`double=2.0`. Scaling is applied only to the releasing leg's hip/knee physical
residual. The admittance internal state continues to evolve unchanged.

GT contact is used only to select and score diagnostic events. It is never
passed to the actor, ContactEstimator, controller, or training path.

## A/A reproducibility

All three runs pass the declared gate.

| Eval seed | dRelease rate | dShort rate | dRelease wheel vz | Base-RMS relative difference |
|---:|---:|---:|---:|---:|
| 123 | 0 | 0 | 0.0000003 m/s | 0.0005% |
| 11 | 0 | 0 | 0.0000012 m/s | 0.0004% |
| 22 | 0 | 0 | 0.0000001 m/s | 0.0037% |

The two unchanged arms also have identical compression trajectories. The
remaining A/A error is orders of magnitude below the intervention effects, so
the short-horizon comparison is materially cleaner than the earlier
reference-timed full rollout.

## Combined result

Values below are event-level means across 288 restored states.

| Arm | Short/release | Release wheel vz (m/s) | Local force peak (N) | Local loading peak (N/s) | Base-acc RMS | Compliance work (J) |
|---|---:|---:|---:|---:|---:|---:|
| A1, 1.0x | 0.6424 | 0.5011 | 79.70 | 14830 | 3.2199 | 0.01763 |
| A2, 1.0x | 0.6424 | 0.5011 | 79.71 | 14830 | 3.2199 | 0.01763 |
| 0.5x | 0.6389 | 0.5061 | 80.84 | 14924 | 3.2191 | 0.00741 |
| 0.0x | 0.6354 | 0.5152 | 80.86 | 14772 | 3.2297 | 0.00000 |
| 2.0x | 0.6424 | 0.5144 | 79.77 | 14685 | 3.2158 | 0.04830 |

Paired effects relative to A1 use an event-level bootstrap over all three
evaluation seeds.

| Arm | dShort [95% CI] | dRelease vz (m/s) [95% CI] | dForce (N) [95% CI] | dLoading (N/s) [95% CI] | dBase RMS [95% CI] |
|---|---:|---:|---:|---:|---:|
| A2 | 0.0000 [0, 0] | -0.0000005 [-0.0000014, 0.0000000] | +0.00 [-0.00, +0.01] | +1 [-0, +2] | +0.00004 [-0.00003, +0.00015] |
| 0.5x | -0.0035 [-0.0243, +0.0174] | +0.00496 [-0.00228, +0.01606] | +1.14 [-0.04, +2.41] | +94 [-175, +378] | -0.00076 [-0.00872, +0.00641] |
| 0.0x | -0.0069 [-0.0278, +0.0139] | +0.01407 [-0.00016, +0.03243] | +1.16 [-0.27, +2.63] | -58 [-438, +304] | +0.00980 [-0.00170, +0.02183] |
| 2.0x | 0.0000 [-0.0104, +0.0104] | **+0.01332 [+0.00884, +0.01919]** | +0.07 [-1.40, +1.55] | -145 [-512, +216] | -0.00410 [-0.01474, +0.00755] |

The source-event loading distribution spans 5220 to 178102 N/s. Snapshot
compression has median 0.215 mm and interquartile range 0.061 to 1.155 mm, so
the test is not limited to zero-authority contacts.

## Interpretation

Late scalar authority is not an actionable short-recontact control in these
states:

* removing or halving the residual does not reliably reduce upward release
  velocity or short re-contact;
* doubling the residual does not reduce short re-contact and **consistently
  increases upward release velocity** across all three seeds;
* force, loading, and local base acceleration do not show a consistent
  Pareto improvement under any scale;
* compliance work changes strongly with scale, confirming that the physical
  intervention is active rather than numerically ineffective.

This rejects the useful form of the "millimeter residual is simply too small"
hypothesis for the **late pre-release window**. More of the same axial residual
pushes release kinematics in the wrong direction. It also agrees with the
earlier negative late-authority and post-release state counterfactuals.

The experiment does not prove that every possible lifecycle controller is
impossible. It branches only 0-10 ms before the observed release, after the
original impact peak. Consequently the local force/loading values in the table
are post-peak quantities; this experiment cannot test whether an intervention
at impact onset could alter both absorption and the later geometry.

## Decision

Do not tune the late residual scale, promote `2x/3x` residual amplification,
or start 1024/4096-environment training from this result. The evidence now
places the remaining mechanism earlier, during absorption and formation of
the release geometry, and raises a concrete action-expressivity question:
one nonnegative per-leg compliance scalar controls force absorption but cannot
independently command late leg posture or contact trajectory.

The next highest-information experiment is the same validated branch framework
at **impact onset**, still with one authority-scale factor. It should compare
local impact force/loading, 50 ms wheel/leg kinematics, and compliance work.
If stronger axial authority lowers impact force while increasing upward wheel
motion or leg extension, the structural trade-off is confirmed and a small,
independent signed leg-length/knee-shaping action becomes justified. If the
impact-onset branch finds a scale that improves both, retain the current action
space and test timing instead.

Only after one of those mechanisms passes a fixed down-stair force,
short-recontact, vibration, and tracking gate should 1024 environments be used
for scale validation.

## Artifacts

The reproducible evaluator is
`legged_gym/scripts/evaluate_mc_short_horizon_branch.py`. Generated tensors and
JSON remain under ignored `logs/mc_short_horizon_branch/` directories and are
not part of Git.
