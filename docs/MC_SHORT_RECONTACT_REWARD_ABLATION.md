# MC Short-Recontact Reward A/B

## Research decision

This experiment asks whether the remaining contact fragmentation is primarily
an objective problem: PPO is rewarded for lower force, loading and base
acceleration, but the original objective does not explicitly distinguish a
normal touchdown from another touchdown within 20 ms of release.

The reward A/B was selected ahead of a 1024-environment run. In the corrected
20-second paired descent data, all 32 adaptive robots reduced mean event force
and loading relative to baseline, while all 32 increased short re-contact.
This widespread trade-off does not look like a few missing extreme samples.
Moreover, the 30-second protocol mixes descent phases after about 20 seconds:
baseline and adaptive cover different forward distances/contact regimes in the
last 10 seconds. The 20-second protocol is therefore the fair primary endpoint.

## Single-factor implementation

The only objective addition is `quiet_short_recontact`. Physics-rate GT wheel
contact uses the evaluator's unchanged hysteresis: contact begins at 5 N and
ends below/equal to 2 N. A touchdown is penalized only when it occurs within
four 5 ms physics substeps (20 ms) of a preceding release. Initial and ordinary
touchdowns have zero cost. The event state survives the two physics substeps in
one 100 Hz policy transition and is cleared on episode reset.

GT contact enters only this privileged reward and diagnostics. It does not
enter actor/critic observations, ContactEstimator input, controller state or
admittance control. The default reward scale is zero, preserving the validated
configuration. The experiment launcher alone sets B to `-0.01`; after the
environment's normal `dt=0.01` reward scaling this is `-0.0001` per event.
Its observed mean logged episode contribution is about `-0.00193`, so it does
not dominate the existing quiet objective.

Everything else is identical: fresh initialization from
`MC_100Hz/model_9000.pt`, seed 1, 256 environments, 48 steps/environment,
300 iterations, estimator warm-up LR `1e-3`, online LR `3e-4`, replay `0.25`,
impact correction `0.10`, impact gain `2.0`, and no alpha slew. Network,
losses, PPO/GAE, M/D/K, impact label, observation/action dimensions and formal
controller are unchanged.

| Experiment | Short-recontact reward scale |
|---|---:|
| A original objective | 0 |
| B short-only penalty | -0.01 |

## Training-distribution response

Metrics are means over the indicated 100-iteration windows. Counts are per
environment per 100 Hz policy transition.

| Window | A short / touchdown | B short / touchdown | A alpha / comp p95 | B alpha / comp p95 |
|---|---:|---:|---:|---:|
| 0-99 | 0.1965 / 0.3230 | 0.1974 / 0.3241 | 0.0731 / 3.78 mm | 0.0683 / 3.59 mm |
| 100-199 | 0.1966 / 0.3223 | 0.1959 / 0.3223 | 0.0888 / 4.36 mm | 0.0882 / 5.09 mm |
| 200-299 | 0.2115 / 0.3363 | 0.1972 / 0.3255 | 0.0875 / 4.68 mm | 0.1110 / 5.38 mm |

In the last window B lowers the training-distribution short count by 6.75%
and touchdown count by 3.23%. It does not achieve this by shutting off
compliance: mean alpha is 27% higher than A. The objective signal is therefore
learnable and changes the policy solution.

Fixed ContactEstimator validation remains stable. At iteration 300 A/B raw
F1 is 0.3686/0.3739, PR-AUC 0.2876/0.2966, ROC-AUC 0.7284/0.7342 and raw
probability separation 0.1652/0.1713. Final-60 online F1 is 0.3271/0.3289.
Estimator behavior does not explain the control result.

## Fixed down-stair evaluation

Both checkpoints were evaluated at 200 Hz for 20 seconds after a 2-second
warm-up, with 32 environments, 0.5 m/s forward command, 0.14 m steps, no
noise/domain randomization and evaluation seeds 123, 11 and 22. Each seed has
a separately paired original HIMLoco baseline.

The table reports B relative to A, mean percentage change +/- sample standard
deviation across the three evaluation seeds. Negative is favorable except
alpha/beta, for which it only means less use.

| Metric | B vs A |
|---|---:|
| Touchdowns | +1.29 +/- 1.02% |
| Short re-contact <=20 ms | +0.55 +/- 1.83% |
| Mean event force | -2.91 +/- 0.83% |
| Event force p95 / p99 | -0.28 +/- 0.92% / -0.64 +/- 0.07% |
| Mean event loading | -3.47 +/- 0.98% |
| Event loading p95 / p99 | -0.26 +/- 0.91% / -0.69 +/- 0.08% |
| Base acceleration RMS / p95 | +1.67 +/- 0.50% / +2.03 +/- 0.37% |
| 5-20 / 20-50 / 50-100 Hz RMS | +4.96 / +1.63 / -0.83% |
| Absolute x tracking error | -1.15 +/- 1.28% |
| Linear tracking reward | -0.01 +/- 0.08% |
| Mean alpha / beta | +32.37 / +10.01% |
| Mean / p95 compression | -38.13 / -47.93% |

Short re-contact changes are directionally inconsistent: B changes the count
by -1.56%, +1.46%, and +1.74% for seeds 123, 11, and 22. Thus the target event
does not improve beyond evaluation variation. The policy instead learns a
different compliance pattern with larger raw alpha but much smaller realized
compression, slightly lower event severity, and worse low/mid-band vibration.

Relative to the original fixed-PD baseline, B still reduces mean force/loading
by `11.5 +/- 1.0%` / `12.4 +/- 1.1%`, their p95 values by about 5.4%/5.3%,
base RMS by `1.65 +/- 1.00%`, and every 5-100 Hz band. However, it increases
touchdowns by `10.1 +/- 1.1%` and short re-contact by `20.6 +/- 2.0%`.
The original A has a better vibration trade-off: mean force/loading improve
8.9%/9.3%, base RMS improves 3.3%, while short re-contact increases 19.9%.

## Interpretation

The result rejects the useful strong form of the reward hypothesis: **one
small, explicit short-event count penalty does not move the target down-stair
Pareto front outward at 256 environments and 300 iterations.** It is not a
proof that reward design never matters. It shows that this event reward is
optimized on the on-policy training distribution but does not generalize to
the fixed priority scenario, and that stronger/blind weight sweeps would be a
poor next step.

The event is also not rare in the training batch. In the final A window its
rate is 0.2115 per environment-step, approximately 2,600 events per
`256 x 48` iteration. Raising the batch to 1024 can lower gradient variance
and cover more randomized conditions, but does not introduce a missing event
signal or new lifecycle expressivity. The advisor's rare-*extreme*-state
hypothesis remains plausible for tails, but it is not a good explanation for
the broad short-recontact trade-off measured here.

## Decision

The current bottleneck is more consistent with **structural contact/lifecycle
credit and geometry** than simple rare-state coverage. Do not promote the new
reward, sweep its weight, or combine it with a 1024-environment change.

The highest-information next experiment is the previously specified
identical-saved-state, 20-50 ms short-horizon branch. First establish A/A
reproducibility from restored simulator state. Then change one compliance
authority factor after absorption and measure local release wheel velocity,
re-contact and acceleration. This avoids the long-rollout PhysX divergence
that invalidated precise causal interpretation of the late-authority schedule.

Proceed to a 1024-environment **diagnostic** only if either:

1. short-horizon branching shows no actionable controller mechanism and a
   predeclared 1024-vs-256 same-objective experiment is needed to test gradient
   variance; or
2. a single mechanism/reward candidate passes force, short-recontact,
   vibration and tracking gates at 256 environments and needs scale validation.

Proceed toward 4096 only after 1024 improves or preserves those gates across
training and evaluation seeds. If an identical-state intervention changes
release velocity/re-contact while preserving the already-achieved impact
absorption, return to the controller lifecycle mechanism before scale-up.

## Checks and artifacts

Both fresh GPU trainings completed 300 iterations without non-finite logged
scalars. All six adaptive and three paired-baseline evaluations completed with
no measured resets. The original actor, HIM estimator and zero-scale motion
adapter are bit-identical between A and B; only the trainable compliance head
and on-policy-supervised ContactEstimator differ. Checkpoint output dimensions
remain 16D actor, 4D compliance and 8D ContactEstimator.

Direct CPU event tests cover initial touchdown, hysteresis, a 20 ms
cross-policy-step event, expiration and reset. Alpha-zero torque equivalence
tests still pass. `pytest` is unavailable in the HIM environment, so these
test functions were invoked directly. `py_compile` and `git diff --check`
pass. Logs and checkpoints remain ignored by Git.

Training runs:

* A: `logs/MC_ImpactClassifier_Admittance_100Hz/Sep15_16-53-31_impact_map_corr_0p100_gain_2p00_alpha_rise_0p00_online_3em04_replay_0p25_seed_1_env_256_iter_300_shortrec_0p00`
* B: `logs/MC_ImpactClassifier_Admittance_100Hz/Sep15_17-00-38_impact_map_corr_0p100_gain_2p00_alpha_rise_0p00_online_3em04_replay_0p25_seed_1_env_256_iter_300_shortrec_m0p01`
