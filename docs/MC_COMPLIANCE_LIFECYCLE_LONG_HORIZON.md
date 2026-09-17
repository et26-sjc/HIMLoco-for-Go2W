# MC Compliance Lifecycle Long-Horizon Diagnostic

## Research question

This experiment tests whether PPO can learn a complete compliance lifecycle
without a controller change:

```text
impact detection -> activation -> absorption -> timely release
```

The experiment was selected before adding another controller mechanism because
the compliance policy already observes the 16-D controller state (compression,
compression velocity, alpha, and force bias), while the existing reward
contains impact-force, loading-rate, base-acceleration, compliance-usage, and
admittance-displacement terms. A long run on the corrected, alpha-zero
baseline-equivalent torque path can therefore distinguish insufficient policy
training from a structural lifecycle limitation.

No estimator, replay, reward, impact mapping, impact gain, M/D/K, action,
observation, or controller equation was changed. The only tooling change was
exposing the existing checkpoint save interval in the ablation launcher.

## Experiment

The policy was initialized fresh from `MC_100Hz/model_9000.pt` and trained with:

* 256 environments and 1000 Stage-1 iterations, seed 1;
* estimator warm-up LR `1e-3`, online LR `3e-4`, Stage-0 replay ratio `0.25`;
* impact correction scale `0.10`, impact gain `2.0`, no alpha slew;
* checkpoints every 100 iterations and fixed validation every 20 iterations.

Checkpoints 100, 300, 500, 700, and 1000 were evaluated on the same paired
down-stair protocol: 32 environments, evaluation seed 123, 2 seconds warm-up,
20 seconds measured motion, 0.5 m/s command, 0.14 m stairs, noise and domain
randomization disabled, and 200 Hz traces. The baseline is the original
`MC_100Hz/model_9000.pt` under the identical protocol. Each policy starts from
the same seeded terrain/reset/history state; trajectories are allowed to
diverge naturally after control begins.

## Estimator stability

Replay remains effective beyond the previously tested 500-iteration horizon.

| Iteration | Fixed F1 | PR-AUC | ROC-AUC | Probability separation | Logit separation | Force MAE (N) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.3044 | 0.2746 | 0.7242 | 0.1273 | 0.7034 | 28.02 |
| 100 | 0.3667 | 0.2908 | 0.7203 | 0.1410 | 0.8355 | 24.24 |
| 300 | 0.3716 | 0.2983 | 0.7350 | 0.1679 | 1.0978 | 21.86 |
| 500 | 0.4192 | 0.3797 | 0.7784 | 0.2192 | 1.4657 | 20.53 |
| 700 | 0.5029 | 0.5208 | 0.8298 | 0.3057 | 2.0765 | 19.66 |
| 1000 | 0.5340 | 0.5761 | 0.8462 | 0.3717 | 2.8638 | 19.20 |

The lifecycle result cannot be attributed to ContactEstimator forgetting.

## Paired down-stair lifecycle trajectory

Short re-contact means a touchdown within 20 ms of a contact release. Band RMS
is computed per robot from the signed vertical base-acceleration trace.

| Iter. | Touchdowns | Short re-contact | Event force (N) | Loading (N/s) | Base RMS | Base p95 | 5-20 Hz | 20-50 Hz | 50-100 Hz | Tracking error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 13441 | 7418 | 207.39 | 39453 | 3.118 | 6.718 | 1.875 | 1.202 | 1.251 | 0.0421 |
| 100 | 14798 | 9080 | 172.35 | 32341 | 2.906 | 6.236 | 1.683 | 1.110 | 1.150 | 0.0464 |
| 300 | 15805 | 10044 | 161.47 | 30081 | 2.884 | 6.123 | 1.691 | 1.083 | 1.089 | 0.0496 |
| 500 | 15353 | 9676 | 162.63 | 30441 | 2.797 | 5.959 | 1.606 | 1.062 | 1.089 | 0.0502 |
| 700 | 14848 | 8999 | 183.89 | 34635 | 3.016 | 6.555 | 1.771 | 1.146 | 1.143 | 0.0500 |
| 1000 | 14204 | 8307 | 198.51 | 37760 | 3.029 | 6.490 | 1.747 | 1.179 | 1.200 | 0.0458 |

Relative to baseline, iteration 500 provides the strongest physical quiet
vector: event force/loading are lower by 21.6%/22.8%, base RMS and p95 by
10.3%/11.3%, and all 5-100 Hz bands by 11.7-14.4%. However, touchdowns and
short re-contacts are higher by 14.2% and 30.4%, and absolute tracking error is
19.3% higher (the scaled linear tracking reward changes by only -0.4%).

By iteration 1000, touchdowns and short re-contacts recover to +5.7% and
+12.0% versus baseline, but event force/loading improvement also collapses to
only 4.3%. This is not a clean learned release behavior.

## Authority and compression evidence

| Iter. | Mean alpha | Mean beta | Mean drive (N) | Compression mean/p95 (mm) | Fast-compression occupancy | Fast velocity (m/s) |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.029 | 0.128 | 4.72 | 0.663 / 4.471 | 2.56% | 0.1437 |
| 300 | 0.055 | 0.230 | 7.25 | 1.007 / 6.782 | 4.01% | 0.1441 |
| 500 | 0.087 | 0.277 | 7.04 | 0.920 / 7.344 | 4.10% | 0.1453 |
| 700 | 0.082 | 0.290 | 6.68 | 0.986 / 6.988 | 4.40% | 0.1443 |
| 1000 | 0.061 | 0.238 | 4.88 | 0.741 / 5.176 | 4.09% | 0.1446 |

The reduction in late chatter coincides with lower alpha, drive, and
compression, while fast positive compression remains close to the 0.15 m/s
limit. The force/loading benefit disappears at the same time. PPO has learned
the trade-off by reducing effective use of compliance, not by preserving
absorption and adding a timely release phase.

## Interpretation

The long run rejects the strong hypothesis that more PPO iterations alone will
solve the lifecycle problem. It demonstrates that PPO can move along the
force-versus-contact-switching trade-off, but does not demonstrate a state at
which impact absorption is retained while short re-contact is removed. The
best quiet checkpoint is around iteration 500 rather than the final checkpoint,
which is another sign that longer optimization is not monotonically solving
the intended mechanism.

This result also changes the earlier vibration interpretation. On the corrected
residual-only clamp path, all evaluated adaptive checkpoints have lower
5-100 Hz vertical vibration than the paired baseline. Chatter is still present
as excess short re-contact, but it is no longer evidence that every vibration
metric must be worse. Both event severity and event frequency must remain
separate acceptance gates.

## Next minimal experiment

Do not scale this run to 1024/4096 environments yet. The highest-information
next test is an evaluation-only, existing-checkpoint counterfactual on the
second-order compression state at confirmed release: reset or rapidly decay
compression and compression velocity as one coupled admittance state, while
leaving transient-force estimation, alpha, M/D/K, and policy output unchanged.
The prior transient-only and torque-residual-only counterfactuals were negative;
this remaining test directly asks whether stored compression state closes the
release/re-contact loop.

GT contact may define the counterfactual window for diagnosis only. If and only
if that counterfactual reduces short re-contact and 5-100 Hz energy without
losing force/loading attenuation, the deployable follow-up should replace GT
with a proprioceptive unloading confidence and apply the same action to the
compression state. If it fails, the remaining evidence points to gait/contact
geometry altered during absorption rather than a persistent controller state,
and a lifecycle controller modification is not yet justified.

## Decision

* **Can PPO learn some adaptation?** Yes: it changes compliance usage and later
  reduces excess contact switching.
* **Does PPO learn the desired full lifecycle?** No: the late improvement is
  explained by reduced compliance authority and lost impact attenuation.
* **Is the estimator the cause?** No: fixed validation improves throughout.
* **Is retraining immediately useful?** No. One existing-checkpoint compression-
  state counterfactual should precede another fresh run.
* **1024/4096 readiness:** No. More parallel data cannot by itself resolve this
  structural force-versus-chatter trade-off.

## Artifacts

Training run:

`logs/MC_ImpactClassifier_Admittance_100Hz/Sep12_20-42-12_impact_map_corr_0p100_gain_2p00_alpha_rise_0p00_online_3em04_replay_0p25_seed_1_env_256_iter_1000`

Paired evaluations:

* baseline: `logs/mc_quiet_eval/stairs_down/20260911_220428`
* iteration 100: `logs/mc_quiet_eval/stairs_down/20260912_210423`
* iteration 300: `logs/mc_quiet_eval/stairs_down/20260912_210510`
* iteration 500: `logs/mc_quiet_eval/stairs_down/20260912_210558`
* iteration 700: `logs/mc_quiet_eval/stairs_down/20260912_210646`
* iteration 1000: `logs/mc_quiet_eval/stairs_down/20260912_210735`

GT contact quantities remain evaluation/supervision/reward-only and are not
added to policy, estimator, or deployable controller inputs.
