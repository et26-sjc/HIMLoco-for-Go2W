# MC Impact-Aware Admittance Release Audit

## Release identity

Frozen method: replay-stabilized proprioceptive impact-aware admittance.

* physical/motion action: 16D;
* compliance action: 4D;
* policy action: 20D;
* ContactEstimator output: normalized axial force 4D plus raw impact logits 4D;
* controller state: 16D deployable admittance state;
* baseline actor, HIM estimator, motion adapter and policy std frozen;
* Stage-0/Stage-1 estimator LR: `1e-3 / 3e-4`;
* Stage-0 replay ratio: `0.25`;
* impact correction scale/gain: `0.10 / 2.0`.

## Controller integrity

The production controller consumes only the 4D compliance action, the 8D
proprioceptive ContactEstimator output, nominal joint targets and its own 16D
state. It contains no GT contact read, evaluation schedule, posture pulse,
counterfactual reset/clamp, late-authority gate, alpha slew or short-recontact
reward branch.

The only GT contact paths in the training environment create estimator
targets, the frozen privileged quiet rewards and read-only diagnostics. They
do not enter actor observations, ContactEstimator inputs or the admittance
controller.

The residual-only joint-limit guard preserves the original PD expression.
With alpha zero, beta, drive, compression, joint residual and torque residual
are zero; the direct CPU equivalence test requires bit-exact torque equality.

## Experimental isolation

Counterfactual and branching logic lives under `legged_gym/scripts/` in
evaluation-only environment subclasses. These classes are registered only by
their corresponding scripts and are not imported by the default task registry
or training launcher. Generated schedules, traces and restored-state tensors
are written under ignored `logs/` paths.

Historical negative-result reports and their analysis scripts remain in the
repository for research traceability. They are not training entry points.

## Reproducibility and scale

The release launcher asserts the frozen architecture and parameters before
constructing Isaac Gym. It starts fresh from
`logs/MC_100Hz/Aug08_18-20-03_baseline/model_9000.pt` and rejects `--resume`.

Rollout tensors are allocated from runtime `env.num_envs`; at 1024 environments
the PPO batch is `1024 x 48 = 49,152` transitions and is divided by the
unchanged four minibatches. The Stage-0 replay buffer remains capped at 8,192
samples. Fixed validation is deterministically capped at 51,200 samples so its
memory and evaluation cost do not grow fourfold relative to the validated
256-environment protocol.

Adaptive checkpoints include model weights, PPO/HIM/ContactEstimator optimizer
states, iteration/timing state and the Stage-0 replay tensors/cursor. Resume is
therefore self-contained for checkpoints produced by this release.

## Release gate

The release gate consists of Python compilation, whitespace validation, the
direct controller tests, a fresh baseline-initialized training smoke run and a
deterministic quiet-evaluation smoke run.

All release checks passed on 2026-09-17 in the `HIM` conda environment:

| Check | Result |
|---|---|
| Repository Python compilation | PASS |
| `git diff --check` | PASS |
| Frozen config/action/JIT/storage/evaluator contract tests | PASS (4) |
| Real-controller alpha-zero and residual-limit tests | PASS (3) |
| Short re-contact diagnostic reset/window tests | PASS (2) |
| Fresh GPU train: 16 env, full 500-step warm-up, one PPO iteration | PASS |
| Replay checkpoint payload: 8192 x (342 + 16 + 8) | PASS |
| Adaptive checkpoint resume without repeated warm-up | PASS |
| Baseline stairs-down evaluator smoke | PASS |
| Adaptive stairs-down evaluator smoke | PASS |

The fresh smoke installed 8,192 replay samples, evaluated the fixed validation
set, switched the estimator from `1e-3` to `3e-4`, completed one finite PPO
update and saved a self-contained checkpoint. Smoke logs and checkpoints are
under ignored `logs/release_audit_smoke/` and are not release artifacts.

## Issues found and fixed

1. The default task still selected `online_lr=1e-3`, replay ratio `0`, and the
   latest baseline checkpoint. Defaults now select the frozen `3e-4`, `0.25`
   and the explicit `model_9000.pt` baseline.
2. Archived alpha-slew and state-counterfactual branches were present in the
   production controller. They were removed; counterfactual behavior now
   exists only in an evaluator-private controller class.
3. Adaptive checkpoints did not contain the Stage-0 replay set, so resume with
   replay enabled required an external tensor path. New checkpoints contain
   replay tensors and cursor state and resume without repeating warm-up.
4. Fixed-validation memory scaled linearly with `num_envs`. Deterministic
   collection is now capped at 51,200 samples without changing the 200-step
   baseline trajectory.
5. W&B validation and training used inconsistent zero/one-based steps after
   resume. They now share one-based iteration steps and one committed row.

Checkpoints created before this release remain valid for inference. Resuming
an older adaptive checkpoint that does not embed replay still requires its
external Stage-0 replay file; release checkpoints are self-contained.
