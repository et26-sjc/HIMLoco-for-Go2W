# MC Admittance Baseline-Equivalence and Chatter Diagnostic

## Question

This diagnostic isolates two issues before any further estimator or controller
tuning:

1. Does the adaptive torque path exactly preserve the original fixed-PD
   controller when the four compliance actions are zero?
2. During active compliance, are the additional contact transitions and base
   vibration associated primarily with activation/compression, recovery, or
   release/re-contact?

No estimator, reward, M/D/K, impact gain, impact mapping, network, action, or
observation setting was changed. The active traces use the two existing
replay-0.25, 256-environment, 500-iteration checkpoints.

## Alpha-zero baseline preservation

The previous adaptive path added the compliance offsets to the complete
HIMLoco target and then clamped that complete hip/knee target to the URDF joint
limits. The baseline fixed-PD path does not clamp its target. Consequently, a
nominal HIMLoco target outside a URDF limit produced different torque even when
the compliance residual was exactly zero.

The corrected path limits only the compliance residual. A nominal target that
is inside its limits remains inside after adding the residual. A nominal target
that is already outside a limit may be preserved or moved toward the valid
range, but the residual cannot make the violation larger. The baseline
position-error expression and floating-point operation order are retained, so
a zero residual does not introduce an algebraically equivalent but numerically
different expression.

Two levels of validation passed:

* CPU same-state/action tests cover zero residual with nominal targets outside
  the limits and nonzero residuals at both limit directions.
* A 32-environment, 30-second, 200 Hz alpha-zero rollout recorded the original
  fixed-PD torque as a same-state/action counterfactual at every physics
  substep. Across 192,000 environment-substeps (3,072,000 torque elements),
  `adaptive_torque - baseline_torque` was exactly zero. Alpha, beta, drive,
  compression, and compression velocity were also exactly zero.

Separate Isaac Gym processes are not used as the exactness criterion because
GPU PhysX trajectories can diverge after tiny reset/simulation differences.
The evaluator nevertheless re-seeds immediately before the paired reset and
synchronizes observation history. Exact controller equivalence is established
inside one rollout using the same state and action.

Normal alpha-positive safety is retained: only the residual is bounded, and it
cannot push an in-range target outside its limits or worsen an existing nominal
violation.

## Chatter protocol

Both adaptive checkpoints were evaluated on down stairs at 0.5 m/s with 32
environments, a 2-second warm-up, and 30 measured seconds. Domain
randomization and observation noise were disabled. The evaluator sampled at
200 Hz:

* impact probability, alpha, beta, drive force, and transient force;
* admittance compression and compression velocity;
* wheel contact state, touchdown/release, and base vertical acceleration;
* the adaptive torque residual relative to original fixed PD.

Phase definitions are diagnostic only. Fast compression is
`compression_velocity > 0.1 m/s`, recovery is
`compression_velocity < -0.002 m/s`, and neutral is
`abs(compression_velocity) < 0.002 m/s`. A short re-contact is a touchdown
after at most four 200 Hz samples (20 ms) without contact.

## Contact-switching evidence

| Trace | Touchdowns | Short re-contacts (<=20 ms) | Contact duty |
|---|---:|---:|---:|
| Baseline | 15,553 | 8,256 | 0.775 |
| Adaptive train seed 1 | 19,577 | 10,906 | 0.729 |
| Adaptive train seed 2 | 17,973 | 10,070 | 0.766 |

Short re-contacts account for 66% of seed 1's additional touchdowns and 75%
of seed 2's. The increase is therefore dominated by brief release/re-contact
cycles rather than by a proportional increase in normal gait contacts.

## Phase evidence

| Checkpoint | Phase | Sample ratio | Mean abs base acc. (m/s2) | Base acc. p95 (m/s2) | Touchdowns/sample |
|---|---|---:|---:|---:|---:|
| Seed 1 | Neutral | 76.10% | 1.838 | 6.512 | 0.0203 |
| Seed 1 | Fast compression | 2.39% | 4.018 | 10.547 | 0.0532 |
| Seed 1 | Recovery | 15.80% | 1.976 | 6.507 | 0.0364 |
| Seed 2 | Neutral | 76.79% | 1.726 | 6.411 | 0.0203 |
| Seed 2 | Fast compression | 3.06% | 3.752 | 10.452 | 0.1229 |
| Seed 2 | Recovery | 16.38% | 1.875 | 5.991 | 0.0166 |

Fast positive compression has the largest and cross-checkpoint-consistent
base-acceleration increase. Its touchdown rate is 2.6 times the neutral rate
for seed 1 and 6.1 times for seed 2. Recovery has neither the same acceleration
increase nor a consistent touchdown-rate increase across checkpoints.

Alpha rising edges (`delta alpha > 0.05` per 5 ms sample) occupy only
1.37%/1.21% of samples but have mean absolute base acceleration
2.998/4.410 m/s2 and touchdown rates 0.0659/0.0764 for seeds 1/2. Fast
compression reaches an average 0.143 m/s in both runs, close to the configured
0.15 m/s compression-velocity limit.

For comparison, impact-probability rising edges occupy 5.78%/4.87% of samples,
with lower mean acceleration 2.563/2.779 m/s2 and lower touchdown rates
0.0439/0.0453. Impact recognition participates in the drive, but the abrupt
compliance-authority/positive-compression transition is the tighter correlate.

Touchdown-aligned traces also place the acceleration response after contact,
while the controller is still compressing. For seeds 1/2, mean absolute base
acceleration is 2.86/3.00 m/s2 on the touchdown sample and rises to 3.57/3.87
m/s2 one 5 ms sample later. Mean compression velocity remains positive over
that interval. It does not become a negative recovery event.

Seed 2 shows the clearest release mechanism: at release, mean drive is 12.1 N
and compression velocity is +0.0177 m/s. After 50 ms, while only 23.5% of
those legs have contact, mean drive is still 8.4 N and compression has grown
from 1.32 to 2.00 mm. This is consistent with deployable force-estimate/control
state persistence retracting a lightly loaded or airborne leg and creating
grazing release/re-contact. Seed 1 has weaker persistence, but retains the same
fast-compression/base-acceleration association.

## Diagnosis

The strongest current explanation is **activation/fast-compression chatter**,
followed by brief release/re-contact and a base-acceleration peak immediately
after re-contact. A spring recovery/rebound mechanism is not supported as the
primary cause: recovery acceleration is near neutral and its touchdown
association changes direction across the two checkpoints.

The classifier is not the present bottleneck. Impact probability changes are
less tightly associated with the high-acceleration samples than alpha rising
edges and compression velocity. The relevant control issue is that the
unsmoothed 100 Hz compliance command, amplified by the concave beta mapping,
can produce a rapid positive drive/compression transition; estimated transient
force can then persist briefly after physical release.

## Next minimal experiment

Do not change M/D/K, impact gain, reward, estimator, or recovery behavior
together. The single highest-information modification is a zero-preserving
slew limit on the existing alpha state (or equivalently its beta authority),
with alpha zero still mapping exactly to baseline torque. This directly tests
the observed activation/fast-compression mechanism without adding sensors or
controller state dimensions.

Run one small fresh A/B training comparison with the corrected residual-only
limit path: current unsmoothed alpha versus one conservative alpha slew limit.
Use the same replay-0.25 estimator setup and paired down-stair protocol. Accept
the change only if short re-contacts, fast-compression occupancy, and 5-100 Hz
base acceleration decrease while force/loading improvements and tracking are
retained.

The controller fix means existing checkpoints remain useful for diagnosis but
were trained under the old target-clamp dynamics. A short retraining experiment
is therefore justified. A 1024/4096-environment production run is not yet
justified until this chatter gate passes.

## Artifacts

Evaluation artifacts remain ignored by git:

* alpha-zero: `logs/mc_quiet_eval/stairs_down/20260911_193837`
* adaptive train seed 1: `logs/mc_quiet_eval/stairs_down/20260911_194607`
* adaptive train seed 2: `logs/mc_quiet_eval/stairs_down/20260911_194046`

GT contact is used only by evaluation instrumentation. It is not exposed to
the policy, ContactEstimator, or admittance controller.
