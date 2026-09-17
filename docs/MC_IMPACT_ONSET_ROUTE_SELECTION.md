# MC Impact-Onset Branching and Final Route Selection

## Decision question

This is the final mechanism experiment before freezing the research route. It
asks whether the existing nonnegative axial-compliance scalar has an
impact-onset operating point that improves both impact severity and the later
release geometry. If it does not, one signed leg-posture counterfactual tests
whether an independent kinematic direction is locally useful.

No training or production-controller change was made. The experiments use the
existing replay-stabilized seed-1 iteration-500 checkpoint. Policy actions,
ContactEstimator outputs, admittance states, M/D/K, impact gain, rewards and
observation/action dimensions are identical between branch arms.

## Identical-state protocol

The validated short-horizon branch framework was moved from the pre-release
state to the instant immediately before touchdown. For each event it restores
root and DOF states, policy/history tensors, commands, complete admittance
state and diagnostic state, then replays the same future policy actions and
ContactEstimator outputs for 300 ms at 200 Hz.

Protocol:

* down stairs, 0.14 m steps and 0.5 m/s command;
* 64 source environments and 96 events for each evaluation seed;
* seeds 123, 11 and 22, for 288 paired events in total;
* event state captured immediately before the first `>=5 N` contact sample;
* all branches retain normal actuator limits and residual-only joint limits;
* GT contact is used only for event selection and scoring.

Each experiment includes two unchanged A arms. Across all seeds the A/A gates
passed: contact outcomes matched, release-wheel-velocity differences were
below `1.3e-6 m/s`, base-acceleration RMS differed by at most `0.004%`, and
force/loading differences were below `0.23%`. The intervention effects are
therefore measured above the restored-state reproducibility floor.

## Axial authority branch

Only the target leg's hip/knee physical compliance torque residual is scaled.
The arms are `0x`, `0.5x`, `1x`, and `2x`; the admittance internal state still
evolves identically. Values below combine all 288 events.

| Scale | Force peak (N) | Loading peak (N/s) | Release wheel vz (m/s) | Leg-length change (mm) | Short/release | Base-acc RMS | Compliance work (J) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x | 568.20 | 110672 | -0.24885 | -4.959 | 0.5336 | 3.2258 | 0.12369 |
| 0.5x | 569.24 | 110939 | -0.24723 | -5.039 | 0.5391 | 3.2250 | 0.05184 |
| 0x | 570.14 | 111006 | -0.24781 | -5.042 | 0.5366 | 3.2275 | 0.00000 |
| 2x | 566.11 | 110175 | -0.24910 | -5.012 | 0.5322 | 3.2355 | 0.34782 |

Paired bootstrap effects relative to `1x`:

| Scale | dForce (N), 95% CI | dLoading (N/s), 95% CI | dRelease vz (m/s), 95% CI | dShort, 95% CI | dBase RMS, 95% CI |
|---:|---:|---:|---:|---:|---:|
| 0.5x | +1.04 [-0.60, +2.40] | +268 [-220, +729] | +0.00162 [-0.00063, +0.00449] | +0.0055 [-0.0182, +0.0297] | -0.0008 [-0.0050, +0.0031] |
| 0x | +1.94 [+0.71, +3.04] | +335 [-37, +664] | +0.00104 [-0.00123, +0.00365] | +0.0030 [-0.0200, +0.0260] | +0.0017 [-0.0050, +0.0090] |
| 2x | -2.09 [-3.93, -0.44] | -496 [-1104, +72] | -0.00025 [-0.00238, +0.00222] | -0.0015 [-0.0243, +0.0219] | +0.0096 [+0.0029, +0.0168] |

The `2x` branch gives a small `0.37%` force reduction, but no measurable
release-geometry or short-recontact benefit and a repeatable base-vibration
increase. For contacts lasting more than 20 ms (`n=82`), even the force effect
changes sign and all geometry/recontact effects remain uncertain.

The source residual norm at impact onset has quantiles
`[0, 0, 0, 0.234, 2.073] Nm`: at least half of the events have no active
compliance residual at the pre-contact state. This is expected for a
current-event proprioceptive classifier. The first collision often occurs
before the classifier-driven residual has causal authority. Scaling the same
scalar therefore cannot provide a useful joint force/geometry operating point
at impact onset.

## Signed leg-posture counterfactual

Because the axial-authority gate failed, exactly one independent kinematic
counterfactual was run. The normal adaptive residual remains active and a
50-ms sine pulse applies a signed `5 mm` leg-length target through the existing
hip/knee Jacobian. Positive `+5 mm` retracts the wheel toward the hip;
negative `-5 mm` extends it. This is diagnostic-only and does not add a policy
action or modify the production controller.

| Pulse | Force peak (N) | Loading peak (N/s) | Release wheel vz (m/s) | Leg-length change (mm) | Short/release | Base-acc RMS |
|---|---:|---:|---:|---:|---:|---:|
| None | 566.62 | 110296 | -0.26704 | -4.626 | 0.5333 | 3.2544 |
| Retract 5 mm | 560.15 | 108988 | -0.24450 | -4.890 | 0.5062 | 3.2886 |
| Extend 5 mm | 567.81 | 110317 | -0.28561 | -4.200 | 0.6151 | 3.3054 |

Paired effects relative to no pulse:

| Pulse | dForce (N), 95% CI | dLoading (N/s), 95% CI | dRelease vz (m/s), 95% CI | dShort, 95% CI | dBase RMS, 95% CI |
|---|---:|---:|---:|---:|---:|
| Retract | -6.47 [-11.94, -1.21] | -1307 [-2639, -3] | +0.02045 [+0.01228, +0.02927] | -0.0313 [-0.0891, +0.0245] | +0.0342 [+0.0169, +0.0524] |
| Extend | +1.20 [-5.17, +7.56] | +22 [-1445, +1493] | -0.01890 [-0.03571, -0.00168] | +0.0825 [+0.0005, +0.1649] | +0.0510 [+0.0342, +0.0687] |

Retraction consistently improves force/loading but increases upward release
velocity and base vibration. Extension moves release velocity in the desired
direction, yet increases short re-contact and base vibration without a force
benefit. The signed direction is physically effective, but neither direction
moves the complete force/geometry/vibration Pareto front outward.

## Final interpretation

The result does not show a hidden scalar-compliance setting that solves the
remaining lifecycle issue. It also does not support adding a signed posture
action on the basis of one counterfactual: the independent kinematic direction
has authority, but its local benefits split across opposite signs and its
vibration cost is consistent.

The current method should therefore be frozen as **Route 1: replay-stabilized,
proprioceptive impact-aware admittance for impact-severity attenuation**. Its
claim must remain scoped: it reduces force/loading peaks and can preserve
locomotion, but it does not eliminate all micro contact switching. No further
alpha, residual, lifecycle, posture-amplitude or timing sweep is justified by
the present evidence.

Route 2, impact-aware posture/contact-trajectory shaping, remains future work
rather than the current method. Pursuing it credibly would require a new
method-design phase with a phase/timing representation and new training, not
another local controller patch.

## Stop decision and next stage

Mechanism exploration stops here. The final method keeps the validated
architecture and parameters: 16D HIMLoco motion plus 4D compliance action,
8D force/logit ContactEstimator, replay ratio `0.25`, warm-up/online estimator
LR `1e-3/3e-4`, correction scale `0.10`, impact gain `2.0`, and unchanged
M/D/K and rewards.

The next experiment is no longer a mechanism search. It is a predeclared
`1024`-environment scale-validation run of the frozen Route-1 method, with
checkpoints evaluated by the paired 20-second down-stair protocol. Acceptance
requires retained force/loading reduction, no regression in vibration or
tracking relative to the validated iteration-500 operating point, stable
fixed-validation estimator metrics, and no increase in the established
short-recontact trade-off. A `4096`-environment run is justified only after
that gate passes across training/evaluation seeds.

## Artifacts and safety

The evaluator is
`legged_gym/scripts/evaluate_mc_impact_onset_branch.py`. Generated tensors and
JSON are under ignored `logs/mc_impact_onset_branch/` directories. GT contact
is evaluation-only and never enters the actor, critic, ContactEstimator,
admittance controller, reward or training storage.
