# MC Paired Down-Stair Quiet Validation

## Question

This experiment asks whether the replay-stabilized impact-aware policy has a
repeatable quiet advantage on the priority scenario, descending stairs, and
whether a 1024/4096-environment training run is the next most informative use
of compute.

The core method and trained checkpoints were not changed. Evaluation used the
baseline `MC_100Hz/model_9000.pt` and both existing replay-0.25 adaptive
checkpoints trained with seeds 1 and 2.

## Fair paired protocol

The earlier evaluator seeded environment creation, but baseline and adaptive
runner construction consumed different amounts of RNG state before the fixed
terrain reset. The evaluator now accepts `eval_seed` and re-seeds immediately
before that reset. This gives baseline and adaptive runs the same terrain,
command, randomized DOF state, root state, and root velocity for each robot.

Protocol:

* scenario: down stairs, difficulty 0.5 (0.14 m step height);
* command: 0.5 m/s forward;
* 32 environments, 2 s warm-up, 30 s measured duration;
* evaluation seeds: 11, 22, 33;
* domain randomization and observation noise disabled;
* physics-rate metrics sampled at 200 Hz;
* no resets occurred in any measured run.

GT contact quantities are evaluation-only and never enter the actor,
ContactEstimator, or controller.

## Paired results

Values are mean percentage change from each paired baseline over the three
evaluation seeds. Negative is better for all rows except tracking reward.

| Metric | Adaptive train seed 1 | Adaptive train seed 2 | Direction consistency |
|---|---:|---:|---:|
| Mean 3-D event force peak | -14.45 +/- 0.56% | -7.00 +/- 1.02% | better in 6/6 pairs |
| 3-D event force p95 | -6.30 +/- 1.00% | -2.94 +/- 0.51% | better in 6/6 pairs |
| 3-D event force p99 | -2.91 +/- 0.45% | -3.33 +/- 0.58% | better in 6/6 pairs |
| Mean 3-D event loading peak | -15.32 +/- 0.59% | -7.16 +/- 1.13% | better in 6/6 pairs |
| 3-D event loading p95 | -6.48 +/- 1.03% | -3.00 +/- 0.65% | better in 6/6 pairs |
| 3-D event loading p99 | -2.90 +/- 0.42% | -3.30 +/- 0.64% | better in 6/6 pairs |
| Mean contact fz | +3.71 +/- 0.75% | +1.61 +/- 0.16% | worse in 6/6 pairs |
| Base acceleration mean | +10.38 +/- 0.97% | +7.36 +/- 0.35% | worse in 6/6 pairs |
| Base acceleration RMS | +4.02 +/- 0.92% | +4.21 +/- 0.24% | worse in 6/6 pairs |
| Base acceleration p95 | +4.18 +/- 1.19% | +5.47 +/- 0.84% | worse in 6/6 pairs |
| Touchdown count | +24.90 +/- 2.40% | +16.24 +/- 1.95% | worse in 6/6 pairs |
| Absolute x tracking error | +12.78 +/- 2.64% | +5.81 +/- 2.91% | worse in 6/6 pairs |
| Linear tracking reward | -0.57 +/- 0.27% | -0.27 +/- 0.19% | worse in 6/6 pairs |

The baseline averages were 222.50 N mean 3-D event peak, 752.60 N p95,
920.03 N p99, 42469 N/s mean loading peak, 149695 N/s loading p95,
1.724 m/s2 mean base acceleration, 2.891 m/s2 base-acceleration RMS, and
15450 touchdowns per 32-robot/30-second run. Both adaptive training seeds
show the same trade-off: individual contact events get milder, but contacts
switch more often and the body vibrates more.

## Frequency diagnostic

FFT band RMS was calculated per robot from the signed 200 Hz vertical base
acceleration trace and then averaged. Relative to paired baseline:

| Frequency band | Adaptive train seed 1 | Adaptive train seed 2 |
|---|---:|---:|
| 0.5-5 Hz | +1.68% | +7.52% |
| 5-20 Hz | +6.99% | +2.14% |
| 20-50 Hz | +2.42% | +3.39% |
| 50-100 Hz | +2.76% | +3.04% |

Every band is worse for both checkpoints. Together with the increased
touchdown rate, this supports a contact-chatter/compliance-dynamics mechanism
rather than an estimator-stability failure.

## Zero-compliance control

For adaptive training seed 2 and evaluation seed 11, a diagnostic run forced
the four compliance outputs to zero. The original actor and HIM estimator
tensors in the baseline and adaptive checkpoints were verified exactly equal.
Nevertheless, the zero-compliance run did not reproduce baseline exactly:
tracking error was 8.16% higher and touchdown count was 2.88% higher.

Code inspection explains the non-equivalence. The adaptive torque path clamps
the complete hip/knee `q_target` to joint limits even when compliance is zero;
the original fixed-PD baseline torque path does not. Consequently the current
baseline/adaptive comparison includes both learned compliance and this target
clamp. This must be isolated before making a final causal quiet claim. No
controller change was made in this diagnostic stage.

### Follow-up resolution

The clamp confound has since been removed by limiting only the compliance
residual and preserving the original fixed-PD position-error operation order.
A same-state/action counterfactual over 32 environments and 30 seconds found
exactly zero torque difference at alpha zero across all 3,072,000 recorded
torque elements. Normal alpha-positive residual limiting remains active. See
`MC_ADMITTANCE_CHATTER_DIAGNOSTIC.md` for the controller test and the subsequent
phase-aligned chatter analysis.

## Decision

The down-stair result is promising but not yet unambiguously good. It reduces
typical and p95/p99 force/loading peaks consistently, which supports the core
impact-absorption idea. It simultaneously raises touchdown frequency and base
vibration consistently, which conflicts with the stated quiet objective.

* **Most important unresolved issue:** determine why selective compliance
  creates more contact switching and body acceleration, while isolating the
  adaptive target-clamp difference from the compliance residual.
* **1024 environments:** not yet justified as a 500-iteration production run.
  It may be used after the zero-compliance path is baseline-equivalent and the
  paired descent gates are defined. More environments cannot repair an
  evaluation confound or a systematic closed-loop vibration trade-off.
* **4096 environments:** not justified now.
* **Up-stair behavior:** keep as a secondary robustness check. Since the
  priority down-stair scenario itself has a force-versus-vibration trade-off,
  diagnose this mechanism before changing the method or spending scale-up
  compute.

The next highest-value work is limited to two items: first, establish an exact
alpha-zero baseline-preservation test (including the joint-target clamp); then
localize the extra touchdown/base-acceleration energy relative to compression
and impact probability timing. Only after those diagnostics should a method
change or 1024-environment run be selected.
