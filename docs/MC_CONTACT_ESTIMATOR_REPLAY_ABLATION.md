# MC ContactEstimator Stage-0 Replay Ablation

## Method

Stage 0 collects a separate deterministic-baseline replay training set containing `obs_history`, `controller_state`, and the existing 8-D force/impact target. It is stored under `logs/replay/` and is distinct from the fixed validation set.

During each existing Stage-1 supervised update, replay samples are appended to the online minibatch before the unchanged force-MSE plus impact-BCE loss. A configured ratio `r` appends `r * online_batch` samples, giving an effective replay fraction `r / (1 + r)`.

Configuration: `256` envs, `500` iterations, seed `1`, warm-up LR `1e-3`, online LR `3e-4`, replay capacity `8192`.

## Fixed-validation retention

| Replay ratio | Effective fraction | F1 initial/final | PR-AUC initial/final/drop | ROC-AUC initial/final/drop | Prob. sep. initial/final | Logit sep. initial/final | Force MAE initial/final (N) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.000 | 0.3131 / 0.2778 | 0.2809 / 0.1767 / +0.1042 | 0.7283 / 0.6053 / +0.1231 | 0.1335 / 0.0928 | 0.7416 / 0.7829 | 27.57 / 24.50 |
| 0.25 | 0.200 | 0.3317 / 0.3939 | 0.2803 / 0.3213 / -0.0411 | 0.7280 / 0.7547 / -0.0267 | 0.1387 / 0.1965 | 0.7629 / 1.3445 | 27.37 / 21.26 |
| 1.00 | 0.500 | 0.3059 / 0.5064 | 0.2806 / 0.5250 / -0.2444 | 0.7264 / 0.8224 / -0.0959 | 0.1326 / 0.3978 | 0.7479 / 5.2211 | 27.64 / 18.74 |

## Fixed-validation trajectory

| Replay ratio | Iteration | F1 | PR-AUC | ROC-AUC | Prob. sep. | Logit sep. | Force MAE (N) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0 | 0.3131 | 0.2809 | 0.7283 | 0.1335 | 0.7416 | 27.57 |
| 0.00 | 100 | 0.3022 | 0.2066 | 0.6396 | 0.1051 | 0.6866 | 25.88 |
| 0.00 | 200 | 0.2785 | 0.1795 | 0.6032 | 0.0869 | 0.6624 | 24.78 |
| 0.00 | 300 | 0.2769 | 0.1786 | 0.6029 | 0.0884 | 0.7236 | 24.46 |
| 0.00 | 400 | 0.2789 | 0.1792 | 0.6072 | 0.0933 | 0.7772 | 24.26 |
| 0.00 | 500 | 0.2778 | 0.1767 | 0.6053 | 0.0928 | 0.7829 | 24.50 |
| 0.25 | 0 | 0.3317 | 0.2803 | 0.7280 | 0.1387 | 0.7629 | 27.37 |
| 0.25 | 100 | 0.3723 | 0.2958 | 0.7246 | 0.1458 | 0.8813 | 23.61 |
| 0.25 | 200 | 0.3634 | 0.2868 | 0.7205 | 0.1539 | 0.9779 | 22.20 |
| 0.25 | 300 | 0.3722 | 0.2928 | 0.7303 | 0.1666 | 1.0798 | 21.83 |
| 0.25 | 400 | 0.3783 | 0.3065 | 0.7429 | 0.1779 | 1.1949 | 21.37 |
| 0.25 | 500 | 0.3939 | 0.3213 | 0.7547 | 0.1965 | 1.3445 | 21.26 |
| 1.00 | 0 | 0.3059 | 0.2806 | 0.7264 | 0.1326 | 0.7479 | 27.64 |
| 1.00 | 100 | 0.4496 | 0.4480 | 0.7951 | 0.2100 | 1.2220 | 21.18 |
| 1.00 | 200 | 0.4917 | 0.5158 | 0.8226 | 0.2962 | 2.0292 | 19.85 |
| 1.00 | 300 | 0.5200 | 0.5494 | 0.8370 | 0.3712 | 3.2005 | 19.17 |
| 1.00 | 400 | 0.5165 | 0.5403 | 0.8311 | 0.3960 | 4.3129 | 18.91 |
| 1.00 | 500 | 0.5064 | 0.5250 | 0.8224 | 0.3978 | 5.2211 | 18.74 |

## Final 20% online and closed-loop metrics

| Replay ratio | Online P/R/F1 | Comp impact/no-impact (mm) | R_comp | Comp p95 (mm) | Saturation | dF peak (N/s) | F peak (N) | Base acc. (m/s^2) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.2456 / 0.3780 / 0.2974 | 2.507 / 1.534 | 1.635 | 8.575 | 0.001135 | 5031.5 | 63.36 | 5.144 |
| 0.25 | 0.2727 / 0.4100 / 0.3271 | 1.082 / 0.723 | 1.496 | 4.213 | 0.000600 | 4969.6 | 63.85 | 5.117 |
| 1.00 | 0.2473 / 0.3846 / 0.3007 | 1.845 / 1.250 | 1.476 | 6.979 | 0.000274 | 4907.8 | 63.85 | 5.048 |

## Locomotion retention (final 20%)

| Replay ratio | Reward | Tracking lin | Tracking yaw | Base height |
|---:|---:|---:|---:|---:|
| 0.00 | 17.140 | 0.996 | 0.524 | -0.238 |
| 0.25 | 19.337 | 1.122 | 0.578 | -0.154 |
| 1.00 | 20.123 | 1.170 | 0.596 | -0.126 |

## Automated assessment

- Best fixed-validation PR-AUC: replay ratio `1.00` (`0.5250`).
- Online F1 retained within 10% of no replay: `YES`.
- Replay ratio `0.25` reverses the no-replay fixed PR-AUC drop from `+0.1042` to `-0.0411` while improving late online F1 from `0.2974` to `0.3271`.
- Replay ratio `1.00` is the estimator-retention upper bound, but uses a 50% effective replay fraction and has higher non-impact compression and compression p95 than ratio `0.25`.
- The balanced Stage-1 candidate is replay ratio `0.25`: it is the smallest tested replay setting that removes fixed-validation drift, gives the best late online F1, and keeps compression lowest.
- A seed-2 repeat of ratio `0.25` is the next decision experiment. A 1024-env run is justified only after that repeat; 4096-env long training is not yet justified without a baseline quiet/tracking comparison across seeds.
- The lower absolute compression at ratio `0.25` comes with a slightly lower `R_comp` than no replay, so this run establishes estimator retention rather than final quiet superiority.

## Decision

- Experiment campaign: `PASS`.
- Does minimal replay prevent long-horizon fixed-validation drift: `YES`.
- Does replay preserve online adaptation: `YES` for both tested ratios; ratio `0.25` has the best final-window online F1.
- Recommended replay candidate: `0.25` (20% of the mixed supervised batch).
- Recommend 1024 env now: `NO`, pending the ratio `0.25` seed-2 replication.
- Recommend 4096 env now: `NO`.

## Experimental limits

The replay runs use one seed. The no-replay row reuses the prior 500-iteration run with identical training settings; a CPU compatibility check confirms that replay ratio zero leaves the estimator update and parameters exactly unchanged. Initial F1 varies slightly across fresh GPU runs, while initial PR/ROC are nearly identical. Quiet metrics show trends only: no dedicated baseline quiet-evaluation protocol was run in this ablation.

## Information-flow safety

Replay contains only Stage-0 proprioceptive estimator inputs and their supervised targets. The fixed validation file remains forward-only and is never used by the optimizer. Replay does not enter PPO storage, actor/critic observations, reward, or the admittance controller.

## Run directories

- replay `0.00`: `/home/sjc/HIMLoco-for-Go2W/logs/MC_ImpactClassifier_Admittance_100Hz/Sep09_17-31-54_contact_stability_warm_1em03_online_3em04_continual_seed_1_env_256_iter_500`
- replay `0.25`: `/home/sjc/HIMLoco-for-Go2W/logs/MC_ImpactClassifier_Admittance_100Hz/Sep10_00-21-18_contact_stability_warm_1em03_online_3em04_replay_0p25_continual_seed_1_env_256_iter_500`
- replay `1.00`: `/home/sjc/HIMLoco-for-Go2W/logs/MC_ImpactClassifier_Admittance_100Hz/Sep10_00-45-51_contact_stability_warm_1em03_online_3em04_replay_1p00_continual_seed_1_env_256_iter_500`
