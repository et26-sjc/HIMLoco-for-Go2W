# MC ContactEstimator Long-Horizon Stability Diagnostic

## Research question

Does a `1e-3` warm-up LR followed by a lower Stage-1 LR prevent long-term fixed-validation drift without degrading selective admittance control?

Both runs start fresh from `MC_100Hz/model_9000.pt`, use the same fixed baseline validation tensors, and preserve the existing model, loss, PPO, reward, label, and admittance definitions.

Configuration: `256` envs, `500` iterations, seed `1`, warm-up LR `1e-3`, warm-up steps `500`, validation every `20` iterations.

## Fixed-validation trajectory

| Online LR | Iteration | F1 | PR-AUC | ROC-AUC | Prob. sep. | Logit sep. | Force MAE (N) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3e-04 | 0 | 0.3131 | 0.2809 | 0.7283 | 0.1335 | 0.7416 | 27.57 |
| 3e-04 | 100 | 0.3022 | 0.2066 | 0.6396 | 0.1051 | 0.6866 | 25.88 |
| 3e-04 | 200 | 0.2785 | 0.1795 | 0.6032 | 0.0869 | 0.6624 | 24.78 |
| 3e-04 | 300 | 0.2769 | 0.1786 | 0.6029 | 0.0884 | 0.7236 | 24.46 |
| 3e-04 | 400 | 0.2789 | 0.1792 | 0.6072 | 0.0933 | 0.7772 | 24.26 |
| 3e-04 | 500 | 0.2778 | 0.1767 | 0.6053 | 0.0928 | 0.7829 | 24.50 |
| 1e-04 | 0 | 0.3247 | 0.2792 | 0.7261 | 0.1367 | 0.7524 | 27.62 |
| 1e-04 | 100 | 0.3247 | 0.2540 | 0.6928 | 0.1209 | 0.6505 | 26.34 |
| 1e-04 | 200 | 0.3141 | 0.2313 | 0.6648 | 0.1132 | 0.6542 | 25.87 |
| 1e-04 | 300 | 0.2938 | 0.1975 | 0.6264 | 0.0966 | 0.6395 | 25.47 |
| 1e-04 | 400 | 0.2849 | 0.1883 | 0.6131 | 0.0892 | 0.6271 | 25.01 |
| 1e-04 | 500 | 0.2769 | 0.1801 | 0.6027 | 0.0823 | 0.5944 | 24.80 |

## Drift summary

A positive drop means the final fixed-validation metric is worse than its post-warm-up value. Slopes are least-squares changes per 100 Stage-1 iterations over all validation snapshots.

| Online LR | Initial/final F1 | F1 drop | Initial/final PR-AUC | PR drop | PR slope | Initial/final ROC-AUC | ROC drop | ROC slope |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3e-04 | 0.3131 / 0.2778 | +0.0353 | 0.2809 / 0.1767 | +0.1042 | -0.0154 | 0.7283 / 0.6053 | +0.1231 | -0.0180 |
| 1e-04 | 0.3247 / 0.2769 | +0.0478 | 0.2792 / 0.1801 | +0.0992 | -0.0206 | 0.7261 / 0.6027 | +0.1234 | -0.0252 |

## Early versus late closed-loop behavior

The early window is iterations 0-99 and the late window is iterations 400-499.

| Online LR | Online F1 early/late | R_comp early/late | Comp p95 early/late (mm) | Base acc. early/late (m/s^2) | Reward early/late |
|---:|---:|---:|---:|---:|---:|
| 3e-04 | 0.4170 / 0.2974 | 1.498 / 1.635 | 3.313 / 8.575 | 4.847 / 5.144 | 19.008 / 17.140 |
| 1e-04 | 0.4415 / 0.2808 | 1.426 / 1.502 | 3.913 / 7.534 | 4.823 / 5.076 | 18.914 / 18.862 |

## Final 20% online and control metrics

| Online LR | Online P/R/F1 | Comp impact/no-impact (mm) | R_comp | Comp p95 (mm) | Saturation | dF peak (N/s) | F peak (N) | Base acc. (m/s^2) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3e-04 | 0.2456 / 0.3780 / 0.2974 | 2.507 / 1.534 | 1.635 | 8.575 | 0.001135 | 5031.5 | 63.36 | 5.144 |
| 1e-04 | 0.2302 / 0.3610 / 0.2808 | 1.847 / 1.230 | 1.502 | 7.534 | 0.001170 | 4994.0 | 63.91 | 5.076 |

## Locomotion retention (final 20%)

| Online LR | Reward | Tracking lin | Tracking yaw | Base height |
|---:|---:|---:|---:|---:|
| 3e-04 | 17.140 | 0.996 | 0.524 | -0.238 |
| 1e-04 | 18.862 | 1.140 | 0.582 | -0.142 |

## Automated reading

- Neither LR solves long-term estimator drift. Both lose about `0.10` PR-AUC and `0.12` ROC-AUC by iteration 500 and converge to a similar fixed F1 near `0.277`.
- `1e-4` delays degradation through roughly 200 iterations, but then continues declining. `3e-4` degrades earlier and reaches a plateau near iteration 200.
- Highest final-window compliance selectivity: `3e-04`.
- Lowering Stage-1 LR therefore mitigates short-horizon parameter drift but does not remove the underlying continual-learning problem.
- A separate Stage-0 replay training set is now justified as the next minimal experiment. The fixed validation set must remain strictly evaluation-only.
- Stage-specific LR is considered a full drift solution only if fixed PR/ROC remain approximately stable across the complete 500-iteration trajectory, not merely at iteration 100.
- This single-seed diagnostic can select the next candidate, but a production decision still requires a second seed and comparison against the baseline quiet/tracking evaluation protocol.

## Distance to readiness

Two simulation gates remain before large-scale training. First, the estimator must retain fixed PR/ROC performance for 500 iterations using an independent replay-training set or an equivalent continual-learning control. Second, with estimator semantics stabilized, the compliance policy must avoid the observed long-horizon compression growth while demonstrating quiet improvement against the baseline without tracking loss.

Only after both gates pass across seeds is 1024/4096-env training well justified. Hardware deployment then remains a separate sim-to-real validation stage.

## Safety

Fixed validation is forward-only under `torch.inference_mode()` and never enters PPO storage, estimator optimization, reward, or the admittance controller.
Both runs completed all 500 iterations without non-finite TensorBoard scalars; checkpoint inspection confirmed the expected online LR and unchanged baseline actor, HIM estimator, and motion adapter.

## Run directories

- online `3e-04`: `/home/sjc/HIMLoco-for-Go2W/logs/MC_ImpactClassifier_Admittance_100Hz/Sep09_17-31-54_contact_stability_warm_1em03_online_3em04_continual_seed_1_env_256_iter_500`
- online `1e-04`: `/home/sjc/HIMLoco-for-Go2W/logs/MC_ImpactClassifier_Admittance_100Hz/Sep09_17-43-18_contact_stability_warm_1em03_online_1em04_continual_seed_1_env_256_iter_500`
