# MC Late-Absorption Authority Counterfactual

## Question and scope

Can attenuation of the **physical compliance torque residual** during the final
unloading part of an existing contact reduce short re-contacts and body
vibration without giving up the original impact-force reduction? This is an
evaluation-only intervention on an existing seed-1, iteration-500 checkpoint.
There was no retraining or change to the production controller, estimator,
replay, PPO, rewards, M/D/K, impact gain, network, or observation/action sizes.

The user's visual inspection found no large-scale instability or striking
chatter, and the compliance displacement is typically millimeters. Here
"chatter" means the previously measured excess **200 Hz short contact
switching and base vibration**, not a claim of visibly unstable walking.

## Offline diagnostic protocol

Original adaptive reference: `logs/mc_quiet_eval/stairs_down/20260913_000333`
and its seed-123 200 Hz physics trace. The policy checkpoint is
`logs/MC_ImpactClassifier_Admittance_100Hz/Sep12_20-42-12_impact_map_corr_0p100_gain_2p00_alpha_rise_0p00_online_3em04_replay_0p25_seed_1_env_256_iter_1000/model_500.pt`.
Evaluation: 32 environments, same reset seed 123, down stairs at difficulty
0.5, forward command 0.5 m/s, 2 s warm-up and 30 measured seconds, no domain
randomization/noise. No measured rollout reset.

The **offline** schedule uses the completed reference contact episode to
identify its final 40 ms. At each eligible reference sample the wheel is
contacting, the past per-contact force and loading-rate peaks have fallen to
at most 80% of peak, the past loading peak exceeds the *unchanged* 5000 N/s
impact threshold, and the admittance compression is positive. Reference GT
contact/force/loading and the *future reference release time* are used **only
to construct this evaluation schedule**: this is neither a deployable rule nor
future-impact prediction. The evaluator receives only fixed authority scales;
it does not receive or gate on reference or branched GT contact. The original
contact window covers 50,245 environment-leg-substeps, 9.0% of reference
contact samples. Its timing has a one-physics-sample lag relative to the GT
reference trace, because torques precede the current physics observation.

At each substep the evaluator computes both adaptive and original fixed-PD
hip/knee torques on **the same state and 16D motion action** and applies
`baseline_torque + scale * (adaptive_torque - baseline_torque)` only at those
hip/knee joints. A leaves the original path unchanged; B uses 0.5 inside the
reference unloading window; C linearly decays the scale to zero over 20 ms
after entering that window (60.4% of C's attenuated samples reach zero; short
contacts may release before the full decay). The compliance model's
compression and velocity states still evolve normally, so this tests residual
**authority**, not state
reset. The alpha-zero baseline-equivalence remains exact by construction.

Every branch has the same initial reset but is a separate GPU PhysX rollout:
these are not identical *entire* trajectories. Fixed reference timestamps
become progressively less aligned after branch divergence; alignment is
reported rather than silently attributing differences to the intervention.

## Paired branch measurements

Here "short" denotes a GT wheel re-contact within 20 ms of release. The
filtered short-event count omits single-sample contact episodes; the raw
20-ms count includes them. The additional no-intervention A repeat measures
process-to-process variation under the same evaluation parameters.

| Branch | Raw touchdowns | <=20 ms re-contacts | Filtered short events | Short release wheel vz (m/s) | F3D event p95/p99 (N) | dF3D event p95/p99 (N/s) | Base acc RMS (m/s2) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A original reference | 21,284 | 12,987 | 8,189 | 0.2971 | 676.3 / 884.5 | 134,066 / 176,784 | 2.9467 |
| A repeat, no intervention | 21,049 | 12,727 | 7,934 | 0.2939 | 676.5 / 881.7 | 134,487 / 176,147 | 2.9591 |
| B authority x0.5 | 21,124 | 12,677 | 7,989 | 0.3030 | 669.9 / 877.8 | 132,710 / 175,253 | 2.9495 |
| C authority decay to 0 | 21,178 | 12,756 | 8,048 | 0.2956 | 678.2 / 887.0 | 134,257 / 177,048 | 2.9753 |

Relative to the **original** A, B reduces short re-contacts by 310 and C by
231, but the second unmodified A independently differs by 260. Relative to
the contemporaneous A repeat, B is lower by only 50, whereas C is higher by
29. B's short-event release velocity **rises** by 0.006 m/s; C reduces it by
only 0.002 m/s versus the original A and raises it versus the A repeat.
Neither branch reduces the overall base-acceleration RMS versus the original A.

| Branch | 5-20 Hz | 20-50 Hz | 50-100 Hz (base vertical acc band RMS, m/s2) | Mean x error (m/s) | Mean tracking linear reward |
|---|---:|---:|---:|---:|---:|
| A original | 1.7934 | 1.1543 | 1.0974 | 0.05346 | 0.01460 |
| A repeat | 1.8121 | 1.1449 | 1.0896 | 0.05354 | 0.01460 |
| B x0.5 | 1.7887 | 1.1552 | 1.1024 | 0.05386 | 0.01459 |
| C decay | 1.8255 | 1.1646 | 1.1118 | 0.05485 | 0.01458 |

Reference baseline (`20260913_000217`) p95/p99 3-D contact-force peaks
were 761.8/917.4 N and loading peaks 151,458/183,389 N/s. Thus B and C
retain the *existing* force/loading attenuation against the original fixed-PD
baseline. C does **not** retain A's exact peak or vibration values, though;
baseline-to-adaptive impact reduction alone is not the lifecycle success gate.

## Release geometry, authority work and window fidelity

Filtered short-event mean release leg length is 0.3543 m (A), 0.3560 m (B),
0.3544 m (C). Short-event mean net compliance work is -0.0976 J (A),
-0.1028 J (B), -0.0864 J (C). Stable-event net work is +0.1715 J (A),
+0.1737 J (B), +0.1608 J (C). Mean compression is 0.912/0.905/0.906 mm
respectively; mean absolute all-DOF torque residual is
0.148/0.144/0.142 Nm. Full five-point normalized impact-to-release leg-length
trajectories and both short/stable release-velocity/torque-work summaries are
included in the ignored `late_absorption_seed123_results.json` analysis file.

The fixed reference contact and each branch's current contact agree in only
76.8% (B) and 77.9% (C) of leg-substeps. During actual attenuation, 27.1%
(B) and 28.4% (C) of samples occur when the branched leg no longer contacts.
Even the independently reproduced unmodified A agrees with the reference
contact in just 76.9% of leg-substeps. This is a **major limitation** of
reference-timed full-rollout counterfactuals under GPU PhysX: the study cannot
estimate a precise per-release treatment effect or disprove *all* possible
release-aware control. C also restores full residual authority at the end of
the *reference* contact; it does not modify post-release persistence, by design.

## Decision and next experiment

The requested three-way gate does **not** pass: the raw re-contact decreases
are on the scale of the unmodified A repeat, short-release vz does not
consistently fall, and 5-100 Hz base acceleration is not reduced. Late-contact
residual attenuation is therefore **not established as the principal causal
source** of chatter. Do not implement a deployable lifecycle rule from this
result and do not retrain a new controller yet.

The next focused analysis is contact geometry/gait interaction **during
absorption**, using the already captured wheel height, leg length, load
transfer and per-contact trajectories to determine how stance geometry changes
the propensity for near-immediate re-contact. A better causal test, if later
needed, must branch on identical saved physics states for *short* horizons or
validate a deployable proprioceptive late-absorption signal independently;
the GT-window schedule in this study must never be deployed.

The advisor's smaller-PPO-batch explanation is plausible but untested: the
current 256-env setting has 48 steps/env, i.e. 12,288 transitions/iteration;
these are correlated and may miss rare difficult states. Increasing to 1024
environments could improve rare-state coverage and policy exploration, but
**cannot automatically repair** a repeatable force-versus-contact-switching
tradeoff or this evaluation's causal-window misalignment. Do not begin
1024/4096-environment training on these results alone. First establish a
mechanistic/evaluation gate that retains force/loading reduction *and* reduces
short re-contacts and 5-100 Hz vibration across matched evaluation seeds.

## Artifacts and safety

* Offline builder: `legged_gym/scripts/build_mc_late_absorption_schedule.py`;
  branch evaluator: `legged_gym/scripts/evaluate_mc_impact_quiet.py`;
  analysis: `legged_gym/scripts/analyze_mc_late_absorption.py`.
* Ignored schedule/results: `logs/mc_quiet_eval/stairs_down/late_absorption_seed123_last40ms_schedule.npz`
  and `late_absorption_seed123_results.json`.
* B evaluation: `logs/mc_quiet_eval/stairs_down/20260915_155609`;
  C evaluation: `20260915_155750`;
  A repeat: `20260915_160057`.
* Reference GT never enters actor/critic/ContactEstimator inputs, estimator
  optimizer, PPO storage, rewards or the formal admittance controller. The
  evaluation environment holds only precomputed authority scales; GT events
  are processed offline for schedule construction and metric evaluation.
