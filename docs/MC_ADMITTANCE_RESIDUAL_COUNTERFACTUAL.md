# MC Release-Window Residual Counterfactual

## Scope

This is an evaluation-only causal diagnostic on the fresh no-slew checkpoint
(`replay=0.25`, online ContactEstimator LR `3e-4`, 50-iteration seed-1 run).
No estimator, reward, PPO, impact mapping, M/D/K, or deployment controller was
changed. The diagnostic uses GT wheel contact only to identify a confirmed
release: a leg must have been observed in contact and then be observed out of
contact. A 50 ms window is applied only in the counterfactual evaluator.

Two interventions were compared with the same reset seed and protocol:

* `transient`: zero the transient-driven acceleration term during the release
  window, while allowing the persistent compression state to evolve;
* `residual`: keep admittance state evolving but zero only the HIP/KNEE joint
  compliance residual during the window.

Neither intervention is a proposed deployment method. The unmodified rollout
is the causal reference; each intervention is a separate branch initialized
from the same checkpoint/reset, so future PhysX trajectories are not assumed to
remain bitwise identical.

## Release-window state evidence

In the unmodified rollout, release samples had nonzero controller state:
impact probability `0.385`, beta `0.315`, drive `12.1 N`, compression `1.32 mm`,
and compression velocity `+0.0177 m/s`. The adaptive torque residual was
nonzero (`0.86 Nm` mean max-per-leg magnitude). Fifty milliseconds after
release, drive remained `8.4 N`, beta `0.390`, and compression had increased to
`2.00 mm`, while only 23.5% of the legs had re-contacted. This confirms that
transient/compression/residual authority persists after physical release.

At release, the next-four-sample re-contact probability was 56.7%. Release-time
scalar state was only a weak predictor: correlations with four-sample
re-contact were 0.075 for impact probability, 0.004 for beta, 0.003 for drive,
0.017 for compression, and -0.041 for torque residual. Compression velocity
was more informative in the tail: release samples with `v > 0.02 m/s` had a
future base-acceleration maximum of `8.87 m/s²`, compared with `6.42 m/s²` for
all re-contact events. This supports a coupled state/timing mechanism rather
than a single scalar trigger.

## Counterfactual results

| Mode | Touchdowns | Short re-contacts <=20 ms | Mean event force (N) | Mean loading (N/s) | Base acc RMS (m/s2) | Tracking error (m/s) |
|---|---:|---:|---:|---:|---:|---:|
| Actual | 13,622 | 7,733 | 190.36 | 36,182 | 2.874 | 0.04486 |
| Zero transient drive | 13,697 | 7,776 | 189.44 | 35,943 | 2.889 | 0.04423 |
| Zero compliance residual | 13,675 | 7,723 | 189.47 | 35,955 | 2.887 | 0.04422 |

The separate branches changed touchdowns by less than 0.6% and changed base
acceleration RMS by only `+0.014`/`+0.013 m/s²`; force/loading changes were
about `-0.5%`. Neither intervention provides a decisive reduction in chatter.
The result is therefore **not** support for transient-force persistence alone,
nor for the final joint residual alone, as the unique causal source.

The small effect is expected from a 50 ms branch intervention: contact state,
policy action, and PhysX trajectory immediately diverge, and the policy can
compensate. It does establish that removing one stored term in this short
window is insufficient to explain or eliminate the long-horizon vibration.

## Conclusion and next step

The best-supported causal description is a coupled release/re-contact loop:

```text
nonzero alpha/beta + transient estimate
→ compression velocity / stored compression
→ residual torque during light-load release
→ grazing re-contact
→ post-touchdown acceleration
```

The dominant quantity to modify next should therefore be the **release-time
authority of the complete compliance residual**, with transient force and
compression state logged as separate diagnostics. A deployable candidate must
use only proprioception/controller state (for example, a state-based residual
hold/decay rule); GT-contact gating must not be promoted into control.

Before changing the method, run one same-checkpoint deterministic policy replay
with a controlled alpha schedule or a state-based residual decay, and compare
the exact release-window and 5–100 Hz metrics. Do not tune M/D/K, impact gain,
estimator, replay, or recovery simultaneously.

No retraining is required to interpret this diagnostic. Retraining is required
after any accepted release-authority change because the compliance policy was
trained against the old dynamics. 1024/4096-environment training is not yet
justified: the residual counterfactual has not produced a clear quiet gate.

## Artifacts

* actual reference: `logs/mc_quiet_eval/stairs_down/20260911_221649`
* corrected transient branch: `logs/mc_quiet_eval/stairs_down/20260912_134201`
* corrected residual branch: `logs/mc_quiet_eval/stairs_down/20260912_134300`

All logs remain ignored by git. `counterfactual_active` is an evaluation trace
only; no GT contact tensor enters actor, estimator, reward, or production
admittance inputs.
