# MC Compliance-Activation Slew Ablation

## Objective

The previous paired descent diagnostic localized chatter to rapid compliance
activation and fast positive compression. This experiment tests the smallest
corresponding intervention: limit only the rising edge of the existing raw
compliance authority. Estimator, replay, impact mapping, PPO, reward, impact
gain, virtual M/D/K, labels, and action/observation dimensions were unchanged.

## Implementation

`MCLearnedAdmittance` now accepts
`compliance_alpha_rise_rate_per_s`. With rate `r > 0` and physics step `dt`,

```text
requested_alpha = clip(policy_compliance, 0, 1)
alpha = min(requested_alpha, previous_alpha + r * dt)
```

When the request decreases, it is followed immediately; only activation is
limited. Rate `0` preserves the unsmoothed legacy behavior. The filtered alpha
is then passed through the existing beta mapping. Since alpha is reset to zero,
`alpha=0 -> beta=0 -> zero residual -> baseline PD` remains exact. The existing
residual joint-limit safety is unchanged.

The ablation launcher exposes `--alpha_rise_rate`, plus explicit overrides for
the already validated `--online_lr=3e-4` and `--replay_ratio=0.25` so that both
runs are fresh and comparable.

## Experiment protocol

Both policies were initialized fresh from `MC_100Hz/model_9000.pt`, seed 1,
64 environments, 50 iterations, warm-up 500 steps, online LR `3e-4`, replay
ratio `0.25`, correction scale `0.10`, and impact gain `2.0`.

* A: `alpha_rise_rate=0` (no slew)
* B: `alpha_rise_rate=10 s^-1` (at 200 Hz, maximum raw-alpha rise is 0.05 per
  physics step)

Quiet evaluation used the same baseline protocol for all three traces: 32
environments, down stairs, command 0.5 m/s, 2 s warm-up, 20 s measured motion,
fixed seed 123, no domain randomization/noise, and 200 Hz instrumentation.

## Results

| Run | Touchdowns | Short re-contacts <=20 ms | Mean event force (N) | Mean loading (N/s) | Base acc RMS (m/s2) | Base acc p95 (m/s2) | Tracking error (m/s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 13,441 | 8,256 | 207.39 | 39,453 | 3.118 | 6.718 | 0.04206 |
| No slew | 13,622 | 7,733 | 190.36 | 36,182 | 2.874 | 6.351 | 0.04486 |
| Slew 10/s | 13,739 | 7,783 | 205.05 | 38,882 | 3.185 | 6.945 | 0.04331 |

Relative to baseline, the no-slew fresh policy reduced mean event force by
8.2%, mean loading by 8.3%, and base-acceleration RMS by 7.8%, with touchdown
count only +1.3%. The 10/s slew policy retained only about 1.1% force and 1.4%
loading reduction, increased base-acceleration RMS by 2.2% and p95 by 3.4%,
and increased touchdown count by 2.2%. Tracking error was between baseline and
no-slew (`0.04331 m/s`).

## Activation/compression traces

| Run | Fast-compression occupancy (`v > 0.1 m/s`) | Fast-compression mean velocity | Fast-compression base acc. mean | Alpha-rise base acc. mean |
|---|---:|---:|---:|---:|
| No slew | 1.038% | 0.1366 m/s | 3.872 m/s2 | 7.512 m/s2 |
| Slew 10/s | 1.104% | 0.1321 m/s | 5.314 m/s2 | 4.370 m/s2 |

The slew reduces the acceleration specifically on alpha rising edges, but does
not reduce fast-compression occupancy. It instead produces slightly larger
overall compression (`p95 1.93 mm` vs `1.26 mm`) and larger base vibration.
This indicates that the compliance policy adapts to the delayed authority,
rather than simply becoming quieter. The short re-contact count is also not
reduced.

## Decision

The 10/s one-sided activation slew **does not support the hypothesis** in this
fresh 50-iteration test. The no-slew control happened to provide the strongest
force/loading and vibration result, while the slew candidate was close to
baseline and worse on vibration. This is not evidence that every possible
slew rate is harmful; it does show that a fixed raw-alpha slew at 10/s is not a
safe default and should not be promoted by blind rate tuning.

The likely reason is closed-loop compensation: delaying alpha changes the
compliance policy's effective timing, so it increases or sustains authority
later in the contact cycle. The dominant mechanism remains the coupled
alpha/beta/compression/contact dynamics, not estimator failure. A slew applied
only to the policy action also does not directly gate already stored
compression or transient-force state.

## Recommendation

Do not enter 1024/4096-environment training based on this negative candidate.
Do not change estimator, replay, impact gain, M/D/K, reward, or recovery yet.
The next minimal diagnostic should be a **no-training, same-checkpoint causal
counterfactual** that gates the compliance residual during the measured
release/re-contact window (or logs residual torque and contact phase with a
controlled alpha schedule). This will distinguish policy compensation from
admittance-state persistence before selecting another controller modification.

If a method change is eventually justified, retrain after the controller path
is finalized; the two ablation checkpoints here were trained with the corrected
residual-limit implementation but remain short-horizon research runs.

## Checks and artifacts

* `py_compile`: PASS for admittance, config, launcher, and torque tests.
* Direct CPU tests: PASS (baseline equivalence, residual safety, slew endpoint).
* Fresh A/B training: PASS, no NaN or dimension errors.
* Quiet evaluations: PASS, no resets; logs remain ignored under `logs/`.
* `git diff --check`: PASS.

Training runs:

* no slew: `Sep11_21-58-02_impact_map_corr_0p100_gain_2p00_alpha_rise_0p00_online_3em04_replay_0p25_seed_1_env_64_iter_50`
* 10/s slew: `Sep11_22-00-03_impact_map_corr_0p100_gain_2p00_alpha_rise_10p00_online_3em04_replay_0p25_seed_1_env_64_iter_50`
