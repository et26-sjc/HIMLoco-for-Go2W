# MC ContactEstimator Learning-Rate Ablation

All runs use fresh `MC_100Hz/model_9000.pt` initialization, continual Stage-1 ContactEstimator updates, and the same fixed validation set.

Configuration: `256` envs, `100` iterations, seed `1`, warm-up `500`, validation steps `200`.

The online values are means over the final 20% of iterations; fixed warm-up values are measured at iteration 0 before any Stage-1 update; final values are measured at the requested final iteration. `Drop` is defined as `warm-up - final`, so a positive value means degradation.

## Fixed-validation drift

| LR | Warm-up F1 | Final F1 | F1 drop | Warm-up PR-AUC | Final PR-AUC | AUC drop | Warm-up ROC-AUC | Final ROC-AUC | ROC-AUC drop |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e-03 | 0.3083 | 0.2822 | +0.0261 | 0.2812 | 0.1801 | +0.1011 | 0.7273 | 0.6084 | +0.1189 |
| 5e-04 | 0.2657 | 0.2787 | -0.0130 | 0.2720 | 0.1779 | +0.0941 | 0.7079 | 0.5996 | +0.1083 |
| 3e-04 | 0.1999 | 0.3063 | -0.1064 | 0.2590 | 0.2223 | +0.0367 | 0.6910 | 0.6566 | +0.0344 |
| 1e-04 | 0.0444 | 0.3038 | -0.2595 | 0.2062 | 0.2386 | -0.0325 | 0.6152 | 0.6612 | -0.0460 |

## Separation and force drift

| LR | Prob sep. warm/final/drop | Logit sep. warm/final/drop | Force MAE warm/final (N) |
|---:|---:|---:|---:|
| 1e-03 | 0.1347 / 0.0931 / +0.0415 | 0.7698 / 0.7469 / +0.0229 | 27.26 / 25.51 |
| 5e-04 | 0.0965 / 0.0818 / +0.0147 | 0.4858 / 0.5900 / -0.1042 | 28.69 / 25.88 |
| 3e-04 | 0.0733 / 0.1110 / -0.0377 | 0.3554 / 0.6483 / -0.2929 | 29.69 / 26.26 |
| 1e-04 | 0.0298 / 0.0949 / -0.0651 | 0.1410 / 0.4732 / -0.3322 | 42.49 / 26.99 |

## Online and control metrics (final 20%)

| LR | Online P | Online R | Online F1 | Comp impact (mm) | Comp no-impact (mm) | R_comp | Comp p95 (mm) | dF peak (N/s) | F peak (N) | Base acc. (m/s²) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e-03 | 0.2250 | 0.3497 | 0.2735 | 1.165 | 0.741 | 1.572 | 3.709 | 4835.6 | 63.92 | 4.843 |
| 5e-04 | 0.2328 | 0.3777 | 0.2877 | 1.032 | 0.707 | 1.460 | 3.765 | 4910.3 | 64.09 | 4.841 |
| 3e-04 | 0.3078 | 0.5643 | 0.3978 | 0.925 | 0.619 | 1.496 | 3.399 | 4993.7 | 64.32 | 4.880 |
| 1e-04 | 0.3294 | 0.6579 | 0.4387 | 0.801 | 0.546 | 1.466 | 3.118 | 4936.3 | 64.18 | 4.900 |

## Drift interpretation

- Highest final fixed F1: `3e-04` (`0.3063`).
- Highest final fixed PR-AUC: `1e-04` (`0.2386`).
- Highest final-20% online F1: `1e-04` (`0.4387`).
- Positive drop denotes forgetting on identical fixed tensors; negative drop denotes improvement after Stage-1 updates.
- `1e-3` shows clear fixed-set drift. Reducing LR to `5e-4` does not materially protect PR/ROC ranking, although thresholded F1 recovers slightly from its weaker warm-up snapshot. At `3e-4`, PR-AUC/ROC-AUC degradation is much smaller. At `1e-4`, both AUCs improve during Stage 1, but the 500-step warm-up is severely under-converged.
- One config LR is deliberately used by both the 500-step warm-up and Stage 1. Consequently, within-run initial-to-final differences measure drift cleanly, but final values across LRs also contain different warm-up convergence quality. This ablation does not isolate a Stage-1-only LR schedule.
- `3e-4` is the best conservative follow-up candidate under the fixed 500-step protocol: it has the highest final fixed F1, substantially less AUC drift, and better selectivity than `1e-4`. The single-seed result is not sufficient to change the production default automatically.
- Lower LR did not improve all quiet proxies relative to `1e-3`; classifier stability alone is therefore not evidence of final quiet-performance improvement.

## Run directories

- `1e-03`: `/home/sjc/HIMLoco-for-Go2W/logs/MC_ImpactClassifier_Admittance_100Hz/Sep08_17-50-25_contact_stability_lr_1em03_continual_seed_1_env_256_iter_100`
- `5e-04`: `/home/sjc/HIMLoco-for-Go2W/logs/MC_ImpactClassifier_Admittance_100Hz/Sep08_17-53-06_contact_stability_lr_5em04_continual_seed_1_env_256_iter_100`
- `3e-04`: `/home/sjc/HIMLoco-for-Go2W/logs/MC_ImpactClassifier_Admittance_100Hz/Sep08_17-55-46_contact_stability_lr_3em04_continual_seed_1_env_256_iter_100`
- `1e-04`: `/home/sjc/HIMLoco-for-Go2W/logs/MC_ImpactClassifier_Admittance_100Hz/Sep08_17-58-28_contact_stability_lr_1em04_continual_seed_1_env_256_iter_100`
