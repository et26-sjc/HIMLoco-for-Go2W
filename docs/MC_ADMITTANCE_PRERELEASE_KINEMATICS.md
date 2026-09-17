# MC Admittance Pre-Release Kinematics Diagnostic

## Question

After post-release transient, residual, and compression-state interventions all
failed to remove chatter, this experiment asks whether the adverse physical
state is already present before release, during impact absorption.

No controller, estimator, replay, PPO, reward, action, observation, or model
parameter was changed. The experiment uses the original
`MC_100Hz/model_9000.pt` baseline and the existing adaptive iteration-500
checkpoint from the 256-env/1000-iteration run. GT contact is used only for
offline event labeling.

## Evaluation and event definition

Both policies were evaluated down stairs with 32 environments, seed 123, a
2-second warm-up, 30 measured seconds, a 0.5 m/s command, 0.14 m stair height,
no noise, no domain randomization, and 200 Hz physics traces.

For each wheel, a contact episode begins at touchdown and ends at the first
subsequent release. The outcome is:

* **short re-contact:** another touchdown occurs within four physics samples
  (20 ms) after release;
* **stable:** no touchdown occurs in that window.

The impact sample is the maximum 3-D loading-rate sample in the first 25 ms of
the contact. Kinematics are compared from that sample through the last contact
sample before release. One-sample contact episodes are counted separately
because they contain no resolvable impact-to-release trajectory.

New evaluation-only traces are wheel-center world/base-relative height,
base-to-wheel-center length, and signed compliance power
`(tau_adaptive - tau_baseline) * dq`. Adaptive controller quantities are
explicitly permuted from their semantic `FL,FR,RR,RL` order into the quiet
contact trace's `FR,FL,RR,RL` order before event association.

## Event population

| Policy | Raw touchdowns | Paired episodes | One-sample episodes | Analyzed stable | Analyzed short | Short ratio |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 15338 | 15219 | 5277 (34.7%) | 4699 | 5241 | 52.7% |
| Adaptive | 21284 | 21195 | 7927 (37.4%) | 5068 | 8189 | 61.8% |

Over the longer 30-second descent, adaptive control increases raw touchdowns
by 38.8%, one-sample contacts by 50.2%, and the short outcome fraction by
9.1 percentage points. This is consistent with the previously observed
release/re-contact issue and makes it visible before inspecting controller
state.

## Pre-release feature separation

`Effect` is `(short mean - stable mean) / pooled standard deviation`.
`AUC*` is directionless scalar ranking AUC: `max(AUC, 1-AUC)`. Values around
0.63-0.66 represent useful but overlapping distributions, not a perfect event
classifier.

| Feature | Baseline stable / short | Baseline effect / AUC* | Adaptive stable / short | Adaptive effect / AUC* |
|---|---:|---:|---:|---:|
| Contact duration (ms) | 260.2 / 170.3 | -0.267 / 0.631 | 239.7 / 156.0 | -0.225 / 0.612 |
| Wheel dz, impact to release (mm) | -12.70 / -1.90 | +0.730 / 0.658 | -10.28 / -0.34 | +0.702 / 0.649 |
| Wheel vz at impact (m/s) | 0.187 / 0.201 | +0.029 / 0.546 | 0.226 / 0.255 | +0.055 / 0.555 |
| Wheel vz at release (m/s) | -0.041 / 0.218 | +0.387 / 0.655 | 0.038 / 0.297 | +0.376 / 0.636 |
| Wheel vz, last 20 ms (m/s) | -0.017 / 0.195 | +0.346 / 0.657 | 0.054 / 0.260 | +0.324 / 0.642 |
| Leg-length change (mm) | -11.94 / +2.04 | +0.348 / 0.624 | -12.50 / +4.44 | +0.437 / 0.632 |
| Physical compression max (mm) | 28.49 / 14.88 | -0.489 / 0.641 | 26.82 / 13.04 | -0.505 / 0.634 |
| Physical compression at release (mm) | 24.01 / 13.22 | -0.444 / 0.629 | 22.40 / 11.39 | -0.456 / 0.630 |
| Load fraction at release | 0.192 / 0.201 | +0.046 / 0.532 | 0.166 / 0.188 | +0.131 / 0.546 |

The strongest repeatable signature is upward wheel motion before release.
Adaptive short events release at `+0.297 m/s`, 36% faster upward than baseline
short events (`+0.218 m/s`). Their mean upward velocity over the last 20 ms is
also 33% higher. Adaptive stable contacts are shifted upward as well
(`+0.038 m/s` at release versus `-0.041 m/s` for baseline), so compliance
changes the general release kinematics rather than only a small pathological
tail.

Short episodes already begin from a somewhat shorter leg configuration and
then extend toward release, while stable contacts shorten substantially during
the same normalized phase. The wheel-position, velocity, leg-length, contact-
duration, and physical-compression features therefore agree on one physical
picture: short events are shallow/grazing contacts that unload while the wheel
is moving upward.

## Adaptive-state and work evidence

| Adaptive feature | Stable | Short re-contact | Effect | AUC* |
|---|---:|---:|---:|---:|
| Max admittance compression (mm) | 1.990 | 1.875 | -0.027 | 0.545 |
| Admittance compression at release (mm) | 0.123 | 0.306 | +0.224 | 0.502 |
| Admittance velocity at release (m/s) | -0.0034 | -0.0089 | -0.221 | 0.523 |
| Mean torque residual (Nm) | 0.253 | 0.439 | +0.174 | 0.530 |
| Torque residual at release (Nm) | 0.045 | 0.137 | +0.232 | 0.518 |
| Net compliance work (J) | +0.171 | -0.098 | -0.395 | 0.636 |
| Positive compliance work (J) | 0.218 | 0.060 | -0.286 | 0.595 |
| Absolute compliance work (J) | 0.265 | 0.217 | -0.068 | 0.549 |

Stored compression magnitude by itself remains a weak ranker, matching the
negative post-release state counterfactual. In contrast, net compliance work
has useful separation. The normalized trajectory changes from positive work
near impact to negative work late in the short-contact group, whereas the
stable group remains net positive on average. Torque residual at release is
three times larger in the short-group mean, but its AUC is only 0.518 because
the distributions have heavy overlap; it is not a sufficient scalar gate.

Signed work must be interpreted carefully: negative residual power is evidence
that the compliance torque and joint motion have entered a different late-
contact regime, but this observational comparison alone does not prove whether
that work causes upward release or reacts to it.

## Normalized impact-to-release trajectory

The five values are at 0/25/50/75/100% of each episode's impact-to-release
interval.

| Adaptive trace | Stable | Short re-contact |
|---|---|---|
| Wheel vz (m/s) | 0.226, 0.246, 0.255, 0.247, 0.038 | 0.255, 0.257, 0.284, 0.300, 0.297 |
| Leg length (m) | 0.368, 0.362, 0.356, 0.352, 0.356 | 0.350, 0.350, 0.350, 0.351, 0.354 |
| Physical compression (mm) | 0.52, 10.76, 16.61, 23.23, 22.40 | 0.40, 4.62, 7.75, 10.49, 11.39 |
| Admittance compression (mm) | 0.61, 0.92, 0.66, 0.38, 0.12 | 0.42, 1.25, 1.19, 0.79, 0.31 |
| Torque residual (Nm) | 0.251, 0.343, 0.267, 0.161, 0.045 | 0.170, 0.556, 0.565, 0.425, 0.137 |
| Compliance power (W) | 1.604, 0.509, 0.338, -0.121, -0.015 | 0.256, -0.144, -0.074, -0.837, -0.207 |

The short group is distinguishable before release, particularly over the last
half of the contact. This answers the diagnostic question affirmatively, while
the moderate AUC values show that a hard one-variable state rule would still
produce many false interventions.

## Causal interpretation

The findings support an **absorption-to-unloading transition problem**, not a
post-release stored-state problem:

```text
shallow impact / reduced leg shortening
-> compliance torque remains material while load falls
-> leg begins extending and wheel retains upward velocity
-> release occurs from an adverse physical state
-> short re-contact
```

Some of this geometry exists in the baseline, so compliance did not create the
mode from nothing. It increases its prevalence and amplifies the upward
pre-release velocity. Clearing controller state after release is ineffective
because the wheel/leg position and velocity have already diverged.

## Next minimal counterfactual design

Do not implement a permanent controller rule or retrain yet. The next single-
factor experiment should attenuate compliance authority **late within a still-
active contact**, after useful impact absorption but before release.

For the causal diagnostic only, GT contact and the event force trajectory may
mark this phase. Suppress or smoothly decay the existing compliance joint
residual when all of the following are true:

1. the contact has passed its local peak loading/force;
2. force is declining into an unloading band;
3. compression is positive and the leg is entering extension/unloading.

Do not change M/D/K, impact gain, estimator, policy output, or recovery at the
same time. Compare short re-contact, pre-release upward velocity, force/loading
peaks, vibration, and tracking using the same checkpoint.

If that counterfactual works, a deployable version can replace GT quantities
with signals already available without external sensors: estimated transient
force and its causal decline, internal compression/velocity, and existing
alpha. It would require a small per-leg peak/unloading state but no new policy
observation or GT input. A candidate conceptual gate is:

```text
unloading = compression > x_min
            and estimated_transient_force < rho * causal_peak_force
            and estimated_transient_force is decreasing
residual_authority *= smooth_decay(unloading)
```

This is a design for the next evaluation counterfactual, not an accepted
controller modification. If the GT-marked late-contact attenuation fails,
contact geometry/terrain interaction is the more likely limit and adding a
deployable controller state rule is not justified.

## Decision

* **Pre-release kinematic separation:** YES, moderate and physically coherent.
* **Is stored compression the primary issue:** NO.
* **Is a late-absorption counterfactual justified:** YES.
* **Is a permanent release rule justified now:** NO.
* **Retraining required now:** NO.
* **1024/4096-env training:** NO, until the late-contact causal test passes.

## Artifacts and safety

* baseline trace: `logs/mc_quiet_eval/stairs_down/20260913_000217`
* adaptive trace and offline JSON:
  `logs/mc_quiet_eval/stairs_down/20260913_000333`

The JSON is produced by
`legged_gym/scripts/analyze_mc_prerelease_kinematics.py`. All generated traces
and analysis tensors remain ignored by git. GT contact is read only after
rollout for event labels and never enters actor, estimator, controller, reward,
or PPO storage.
