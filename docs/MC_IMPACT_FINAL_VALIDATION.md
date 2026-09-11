# MC Impact-aware Admittance Final Validation

## Scope

This validation uses the fixed recommended configuration: `pos_weight=3`,
`impact_logit_correction_scale=0.10`, `impact_threshold=5000 N/s`,
`impact_gain=2.0`, warm-up LR `1e-3`, online LR `3e-4`, and Stage-0 replay
ratio `0.25`.  The estimator remains `force(4) + raw impact logits(4)`;
the locomotion/compliance/action dimensions and admittance law are unchanged.

The adaptive checkpoint was trained fresh from
`MC_100Hz/model_9000.pt` for 500 iterations with 256 environments.  Seed 2
was evaluated against the earlier seed-1 run.  Quiet evaluation uses the
same fixed-command, 200 Hz physics-rate instrumentation for the baseline and
adaptive checkpoint; GT contact force is collected only for evaluation
metrics, not passed to the policy or controller.

## Replay=0.25 cross-seed estimator stability

| Seed | Fixed F1 (warm/final) | PR-AUC (warm/final) | ROC-AUC (warm/final) | Final logit separation | Final online F1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.3317 / 0.3939 | 0.2803 / 0.3213 | 0.7280 / 0.7547 | 1.3445 | 0.3271 |
| 2 | 0.3471 / 0.4187 | 0.2817 / 0.3717 | 0.7293 / 0.7780 | 1.5777 | 0.3261 |
| Mean +/- std | 0.4063 +/- 0.0175 final | 0.3465 +/- 0.0356 final | 0.7663 +/- 0.0165 final | 1.4611 +/- 0.1649 | 0.3266 +/- 0.0007 |

Both seeds improve rather than lose fixed-validation separation during the
500-iteration run.  This is the opposite of the no-replay trajectory, where
fixed PR/ROC degraded by about 0.10/0.12.  Seed 2 therefore supports the
stability conclusion, although two seeds are not a production uncertainty
estimate.

The seed-2 final control metrics were compression-after-impact/no-impact
`1.771/1.142 mm`, `R_comp=1.551`, compression p95 `7.124 mm`, saturation
ratio `0.000601`, peak 3-D force `63.67 N`, peak loading `4962.7 N/s`, and
base-acceleration peak `5.125 m/s^2`.  Seed 1 was `1.082/0.723 mm`,
`R_comp=1.496`, p95 `4.213 mm`, saturation `0.000600`, force `63.85 N`,
loading `4969.6 N/s`, and base acceleration `5.117 m/s^2`.

## Baseline versus adaptive quiet evaluation

The following tables use 16 environments, a two-second warm-up, and 20
seconds of measured motion.  `fz_mean`/`fz_peak` are continuous contact
normal-force mean/max; `f3d_peak` and `df3d_peak` are event peak norms and
loading-rate peaks.  There is no single calibrated `quiet_score` scalar in
the current evaluator, so the physical quiet result is reported as this
metric vector.

### Down stairs

| Metric | Baseline | Replay adaptive | Change |
|---|---:|---:|---:|
| fz_mean (N) | 66.11 | 64.89 | -1.9% |
| fz_peak (N) | 1257.24 | 1200.00 | -4.6% |
| f3d_peak mean (N) | 206.62 | 192.86 | -6.7% |
| f3d_peak max (N) | 1257.24 | 1200.00 | -4.6% |
| df3d_peak mean (N/s) | 32669 | 29712 | -9.1% |
| df3d_peak max (N/s) | 251449 | 240000 | -4.6% |
| base acc p95 (m/s2) | 6.6197 | 6.4904 | -2.0% |
| base acc max (m/s2) | 38.17 | 30.02 | -21.3% |
| touchdown count | 6741 | 6908 | +2.5% |
| tracking error x (m/s) | 0.0422 | 0.0464 | +0.0042 |
| tracking linear reward | 0.01474 | 0.01471 | -0.2% |

### Up stairs

| Metric | Baseline | Replay adaptive | Change |
|---|---:|---:|---:|
| fz_mean (N) | 64.46 | 62.95 | -2.4% |
| fz_peak (N) | 1102.92 | 1094.02 | -0.8% |
| f3d_peak mean (N) | 217.35 | 225.88 | +3.9% |
| f3d_peak max (N) | 1102.92 | 1094.02 | -0.8% |
| df3d_peak mean (N/s) | 34455 | 35203 | +2.2% |
| df3d_peak max (N/s) | 220583 | 218804 | -0.8% |
| base acc p95 (m/s2) | 6.6834 | 6.7991 | +1.7% |
| base acc max (m/s2) | 38.21 | 39.96 | +4.6% |
| touchdown count | 5937 | 5531 | -6.8% |
| tracking error x (m/s) | 0.0479 | 0.0515 | +0.0036 |
| tracking linear reward | 0.01473 | 0.01470 | -0.2% |

The adaptive layer reduces the worst down-stair impact measures, while the
up-stair mean event force/loading and base-acceleration p95 are statistically
similar or slightly worse.  It does not simply make every contact softer;
the effect is contact- and direction-dependent.  Tracking reward remains
essentially unchanged, but touchdown count and tracking error should be
checked with more seeds before claiming a universal quiet improvement.

## Acceptance decision

* **Estimator stability across seed:** PASS for this two-seed check. Replay
  ratio 0.25 prevents the fixed-validation degradation seen without replay.
* **Selective compliance:** PASS qualitatively: both seeds have
  `compression_after_impact > compression_after_noimpact` and near-zero
  saturation. The absolute compression and `R_comp` vary by seed.
* **Impact reduction:** PARTIAL. Down stairs shows lower force/loading/base
  acceleration; up stairs does not improve every peak statistic.
* **Locomotion retention:** PASS for this short evaluation: tracking reward
  is effectively unchanged, with a small increase in x tracking error.
* **4096-env readiness:** NOT YET as a final acceptance claim. The replay
  mechanism is a viable candidate for a 1024-env scale-up, but the quiet
  result still needs repeated-seed stair evaluation and a baseline protocol
  with confidence intervals.  A 4096-env run is justified as the next
  validation scale, not as proof of sim-to-real readiness.

Recommended next step: run replay `0.25` at 1024 environments for 500
iterations with at least one additional seed, then repeat the same up/down
stairs quiet evaluation.  Do not change the estimator loss, action dimensions,
impact mapping, or admittance law in that run.

## Artifacts and checks

* Seed-2 training run:
  `logs/MC_ImpactClassifier_Admittance_100Hz/Sep10_18-54-39_contact_stability_warm_1em03_online_3em04_replay_0p25_continual_seed_2_env_256_iter_500`
* Baseline/adaptive quiet summaries are stored under
  `logs/mc_quiet_eval/{flat,stairs_up,stairs_down}/` and are ignored by git.
* The adaptive evaluator uses policy-produced contact estimates; GT contact
  forces are read only by the quiet metrics collector.
