# MC ContactEstimator Stage-Specific LR Ablation

## Experiment

All runs start fresh from `MC_100Hz/model_9000.pt`. Stage 0 uses `1e-3` for 500 deterministic baseline steps; only the existing Adam learning rate changes before Stage-1 continual updates. Optimizer state is preserved.

Configuration: `256` envs, `100` iterations, seed `1`, validation steps `200`.

`Drop = warm-up - final`; positive values indicate degradation on identical fixed-validation tensors. Online/control metrics use the final 20% of iterations.

## Fixed-validation drift and control selectivity

| Warmup LR | Online LR | Warmup F1 | Final F1 | F1 drop | PR-AUC drop | ROC drop | R_comp |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e-03 | 1e-03 | 0.3070 | 0.2828 | +0.0242 | +0.0980 | +0.1166 | 1.521 |
| 1e-03 | 3e-04 | 0.3179 | 0.3020 | +0.0160 | +0.0701 | +0.0841 | 1.533 |
| 1e-03 | 1e-04 | 0.3088 | 0.3194 | -0.0106 | +0.0272 | +0.0341 | 1.484 |

## Complete fixed-validation trajectory endpoints

| Online LR | PR-AUC warm/final | ROC-AUC warm/final | Probability sep. warm/final | Logit sep. warm/final | Force MAE warm/final (N) |
|---:|---:|---:|---:|---:|---:|
| 1e-03 | 0.2792 / 0.1812 | 0.7270 / 0.6104 | 0.1323 / 0.0941 | 0.7476 / 0.7675 | 27.64 / 25.32 |
| 3e-04 | 0.2774 / 0.2074 | 0.7252 / 0.6411 | 0.1353 / 0.1086 | 0.7529 / 0.7027 | 27.30 / 25.97 |
| 1e-04 | 0.2794 / 0.2522 | 0.7240 / 0.6899 | 0.1312 / 0.1200 | 0.7369 / 0.6509 | 27.49 / 26.37 |

## Online and physical-control metrics

| Online LR | Online P | Online R | Online F1 | Comp impact (mm) | Comp no-impact (mm) | R_comp | Comp p95 (mm) | dF peak (N/s) | F peak (N) | Base acc. (m/s^2) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e-03 | 0.2296 | 0.3578 | 0.2794 | 1.368 | 0.900 | 1.521 | 4.745 | 4853.0 | 64.13 | 4.767 |
| 3e-04 | 0.2733 | 0.4670 | 0.3445 | 0.737 | 0.481 | 1.533 | 2.647 | 4886.1 | 64.18 | 4.873 |
| 1e-04 | 0.3316 | 0.6546 | 0.4398 | 1.148 | 0.773 | 1.484 | 4.030 | 4961.5 | 64.29 | 4.931 |

## Stage-specific versus prior single-LR runs

Prior single-LR references use the same LR in warm-up and Stage 1. They are included only to expose warm-up confounding.

| Online LR | Stage-specific warmup F1 | Stage-specific final F1 | Single-LR warmup F1 | Single-LR final F1 |
|---:|---:|---:|---:|---:|
| 1e-03 | 0.3070 | 0.2828 | 0.3083 | 0.2822 |
| 3e-04 | 0.3179 | 0.3020 | 0.1999 | 0.3063 |
| 1e-04 | 0.3088 | 0.3194 | 0.0444 | 0.3038 |

## Conclusions

- Stage-specific LR is useful because it removes the severe low-LR warm-up under-convergence seen in the single-LR runs. It does not improve every final metric monotonically; the benefit is a better-defined warm-up snapshot and an independent drift knob.
- `1e-4` minimizes drift: PR-AUC drop `+0.0272` versus `+0.0980` at `1e-3`, and its final-20% online F1 is `0.4398`.
- `3e-4` is the more balanced online adaptation candidate: R_comp `1.533`, no-impact compression `0.481 mm`, and p95 compression `2.647 mm`. It retains faster adaptation than `1e-4` without the large drift of `1e-3`.
- Replay is not the next justified change. First repeat the `3e-4` and `1e-4` online-LR candidates across seeds or a longer horizon. Replay becomes justified only if fixed PR/ROC degradation persists at the selected lower LR.
- These are single-seed, 100-iteration diagnostic results and do not change the production default automatically.

## Verification

- Fixed validation is evaluated with `torch.inference_mode()` and is never passed to PPO storage or a ContactEstimator optimizer.
- Checkpoints confirm warm-up LR `1e-3` and Stage-1 LRs `1e-3`, `3e-4`, and `1e-4` respectively.
- All runs logged validation at iterations 0, 20, 40, 60, 80, and 100 with no non-finite scalar values.
- The baseline actor, HIM estimator, and motion adapter were unchanged between warm-up and final checkpoints.

## Run directories

- online `1e-03`: `/home/sjc/HIMLoco-for-Go2W/logs/MC_ImpactClassifier_Admittance_100Hz/Sep09_16-57-45_contact_stability_warm_1em03_online_1em03_continual_seed_1_env_256_iter_100`
- online `3e-04`: `/home/sjc/HIMLoco-for-Go2W/logs/MC_ImpactClassifier_Admittance_100Hz/Sep09_17-00-22_contact_stability_warm_1em03_online_3em04_continual_seed_1_env_256_iter_100`
- online `1e-04`: `/home/sjc/HIMLoco-for-Go2W/logs/MC_ImpactClassifier_Admittance_100Hz/Sep09_17-03-01_contact_stability_warm_1em03_online_1em04_continual_seed_1_env_256_iter_100`
