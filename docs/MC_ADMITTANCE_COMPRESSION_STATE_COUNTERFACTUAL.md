# MC Admittance Compression-State Counterfactual

## Question

Does the second-order admittance state retained after wheel release cause the
excess short release/re-contact cycles?

This is an evaluation-only causal diagnostic. It uses the existing 256-env
long-run checkpoint at iteration 500, where impact attenuation is strongest
but excess short re-contact remains clear. No training, estimator, replay,
reward, policy, impact gain, M/D/K, observation, or action definition changes.
GT wheel contact identifies a confirmed release window only; it is not a
candidate deployment input.

## Interventions

All branches start from the same checkpoint and seeded reset. Intervention is
applied before generation of the compliance joint residual:

1. `compression_velocity_reset`: clear only `delta_l_dot` on the first
   controllable 5 ms substep after confirmed release;
2. `compression_state_reset`: clear both `delta_l` and `delta_l_dot` once on
   that release edge, while leaving the existing drive active;
3. `compression_state_clamp`: clamp both state variables to zero throughout a
   50 ms confirmed-off-contact window. This is a deliberately strong causal
   upper bound; it is not a deployable controller proposal.

The reference and every branch use 32 environments, 20 measured seconds,
2 seconds warm-up, down stairs at 0.5 m/s, 0.14 m stair height, evaluation seed
123, no observation noise, and no domain randomization.

## Initial branch screen

| Branch | Touchdowns | Short re-contact <=20 ms | Event force mean/p95/p99 (N) | Loading mean/p95/p99 (N/s) | Base RMS/p95 | 5-20 / 20-50 / 50-100 Hz RMS |
|---|---:|---:|---:|---:|---:|---:|
| Actual | 15353 | 9676 | 162.6 / 627.9 / 873.1 | 30441 / 123439 / 174530 | 2.797 / 5.959 | 1.606 / 1.062 / 1.089 |
| Velocity reset | 15520 | 9744 | 163.1 / 634.7 / 881.8 | 30600 / 123549 / 176168 | 2.838 / 6.024 | 1.628 / 1.080 / 1.100 |
| State reset | 15312 | 9641 | 164.5 / 634.2 / 885.7 | 30816 / 125408 / 176822 | 2.859 / 6.050 | 1.664 / 1.091 / 1.104 |
| State clamp 50 ms | 15161 | 9527 | 165.6 / 645.0 / 884.1 | 31036 / 126764 / 176164 | 2.855 / 6.056 | 1.639 / 1.090 / 1.117 |

Velocity reset is directionally worse. A one-shot full reset changes short
re-contact by only -0.4% and raises vibration. The strong clamp changes short
re-contact by -1.5%, while worsening mean force/loading by 1.8%/2.0% and all
three vibration bands by 2.1-2.8%. This is not the required coupled quiet
improvement.

## Three-seed paired clamp validation

Because the seed-123 clamp produced a small touchdown reduction, the strongest
intervention was repeated with evaluation seeds 11, 22, and 33. Values below
are paired percentage change from the unmodified checkpoint; negative is
better except tracking reward.

| Metric | Seed 11 | Seed 22 | Seed 33 | Mean +/- sample std |
|---|---:|---:|---:|---:|
| Touchdowns | -1.57% | -0.03% | +0.59% | -0.34 +/- 1.11% |
| Short re-contact | -1.48% | +0.06% | +1.56% | +0.05 +/- 1.52% |
| Mean event force | +1.48% | +0.59% | -0.90% | +0.39 +/- 1.20% |
| Force p95 | +0.64% | +1.39% | -0.66% | +0.46 +/- 1.04% |
| Force p99 | +1.59% | -0.55% | +0.84% | +0.63 +/- 1.09% |
| Mean event loading | +1.79% | +0.81% | -1.17% | +0.48 +/- 1.51% |
| Loading p95 | +2.55% | +1.32% | +0.50% | +1.46 +/- 1.03% |
| Loading p99 | +1.53% | -0.46% | +0.75% | +0.61 +/- 1.00% |
| Base acceleration RMS | +0.41% | +0.58% | -0.89% | +0.03 +/- 0.80% |
| Base acceleration p95 | +0.96% | +0.95% | -0.00% | +0.64 +/- 0.55% |
| 5-20 Hz RMS | +2.16% | +2.80% | -0.48% | +1.50 +/- 1.74% |
| 20-50 Hz RMS | -0.14% | +0.24% | -0.16% | -0.02 +/- 0.22% |
| 50-100 Hz RMS | +0.02% | +0.80% | +0.53% | +0.45 +/- 0.40% |
| Compression mean | -10.23% | -8.57% | -10.87% | -9.89 +/- 1.19% |
| Compression p95 | -9.91% | -7.08% | -10.07% | -9.02 +/- 1.68% |
| Absolute tracking error | -2.82% | -1.01% | -1.14% | -1.66 +/- 1.01% |

The intervention strongly and consistently changes its intended internal
state, but does not consistently change re-contact, force, or vibration. The
small tracking improvement is not accompanied by the target quiet response.

## Release-aligned state check

In the seed-123 reference, still-airborne legs carry approximately 0.36 mm
compression and 0.136 Nm maximum HIP/KNEE torque residual 5 ms after release.
At 50 ms they retain 0.27 mm and 0.100 Nm. During the strong-clamp branch these
values are zero at 5 ms and remain approximately 0.007 mm and 0.003 Nm at
50 ms, while estimated drive remains about 2.4-2.6 N. Thus the negative result
is not caused by a failed intervention: stored compression and its physical
joint residual were actually removed.

## Causal conclusion

Persistent compression state is **not a principal sufficient cause** of the
observed chatter. If it were, a 50 ms state clamp that eliminates the residual
through the entire re-contact horizon should have produced a large,
directionally consistent reduction in short re-contact and vibration. It did
not.

The evidence instead points upstream in the lifecycle. Compliance applied
during impact absorption has already changed leg configuration, wheel vertical
velocity, load distribution, and the subsequent contact trajectory by the time
release is observed. Clearing internal state after release cannot undo that
physical state. This interpretation is also consistent with the earlier
negative transient-only, residual-only, and alpha-slew counterfactuals.

This experiment does not prove a single geometric cause; it rules out the most
direct stored-state hypothesis. A deployable release detector/decay mechanism
is therefore **not justified now**, even though such a detector could be built
from estimator force, impact probability, and internal state without GT.

## Next highest-information step

Do not retrain and do not scale to 1024/4096 environments. The next diagnostic
should localize the physical divergence before release by comparing baseline
and adaptive touchdown-aligned traces of wheel vertical position/velocity,
leg length, contact duration, load transfer, compliance torque work, and the
first release time. In particular, split contacts that later become <=20 ms
re-contacts from stable contacts and determine whether their adverse kinematics
are already present during peak compression/absorption.

If pre-release kinematics separate the two groups, the smallest subsequent
counterfactual is to attenuate compliance **late within the same contact after
peak absorption**, using only a deployable unloading proxy such as declining
estimated force together with positive compression and compression-velocity
sign. That must be tested evaluation-only before a fresh retrain. If no
pre-release separation exists, the remaining limitation is likely contact
geometry/observability and another controller state rule would be speculative.

## Safety and artifacts

Production leaves the diagnostic mode unset. GT contact appears only in the
quiet-evaluation wrapper and is not added to actor, estimator, reward, PPO,
controller state, or the deployable torque path. Alpha-zero baseline
equivalence remains covered by the exact torque regression test.

The seed-123 artifacts are:

* actual: `logs/mc_quiet_eval/stairs_down/20260912_210558`
* velocity reset: `logs/mc_quiet_eval/stairs_down/20260912_224502`
* state reset: `logs/mc_quiet_eval/stairs_down/20260912_224551`
* 50 ms state clamp: `logs/mc_quiet_eval/stairs_down/20260912_224640`

Three-seed paired actual/clamp artifacts span
`logs/mc_quiet_eval/stairs_down/20260912_224810` through
`logs/mc_quiet_eval/stairs_down/20260912_225214`. All artifacts remain ignored
by git.
