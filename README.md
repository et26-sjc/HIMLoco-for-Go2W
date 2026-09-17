# HIMLoco for Go2W and MC

This repository extends HIMLoco to the Go2W and MC wheel-legged robots in
Isaac Gym. The frozen MC research method is **replay-stabilized,
proprioceptive impact-aware admittance** at a 100 Hz policy rate.

## Method overview

The MC robot has 16 physical control DOFs: 12 leg joints and four wheel
joints. The adaptive policy keeps the trained HIMLoco locomotion path intact
and adds one compliance action per leg:

```text
6-frame proprioceptive history ----> frozen HIM estimator ----> frozen 16D actor
              |                                      |
              +----> ContactEstimator (force4 + raw impact logits4)
                                                     |
                                      4D compliance head
                                                     |
16D HIMLoco target + sensorless admittance residual -> 16D physical PD command
```

The actor therefore emits `16D motion + 4D compliance = 20D`, while physical
actuation remains 16D. Contact force is used only for estimator supervision,
privileged training rewards and diagnostics. It is not an actor,
ContactEstimator or controller input. A zero compliance action gives exactly
the original fixed-PD HIMLoco torque.

The frozen configuration is:

| Setting | Value |
|---|---:|
| Impact BCE positive weight | 3.0 |
| Impact logit correction scale | 0.10 |
| Impact gain | 2.0 |
| Estimator warm-up / online LR | 1e-3 / 3e-4 |
| Stage-0 replay ratio | 0.25 |
| Compliance initialization std | 0.15 |
| Maximum compression | 0.020 m |
| Policy / physics rate | 100 / 200 Hz |

## Environment

Use the local `HIM` conda environment with the repository's Isaac Gym setup:

```bash
conda activate HIM
pip install -e rsl_rl
pip install -e legged_gym
```

Place the baseline checkpoint at:

```text
logs/MC_100Hz/Aug08_18-20-03_baseline/model_9000.pt
```

Generated logs, checkpoints, validation/replay tensors, W&B state and caches
are ignored by Git.

## Training

The release launcher imports the fixed configuration from
`legged_gym/envs/mc/mc_learned_admittance_100hz_config.py`, verifies the
frozen dimensions and parameters, then initializes fresh from the baseline.

Reproduce the 256-environment training protocol:

```bash
conda run -n HIM python legged_gym/scripts/train_mc_impact_release.py \
  --task=mc_learned_admittance_100hz \
  --num_envs=256 \
  --seed=1 \
  --max_iterations=500 \
  --save_interval=20 \
  --baseline_run=Aug08_18-20-03_baseline \
  --baseline_checkpoint=9000 \
  --log_root=logs/MC_ImpactClassifier_Admittance_100Hz \
  --run_name=route1_release_env256_seed1 \
  --wandb_mode=online \
  --headless
```

Run the formal 1024-environment scale validation by changing only
`num_envs` and the run name:

```bash
conda run -n HIM python legged_gym/scripts/train_mc_impact_release.py \
  --task=mc_learned_admittance_100hz \
  --num_envs=1024 \
  --seed=1 \
  --max_iterations=500 \
  --save_interval=20 \
  --baseline_run=Aug08_18-20-03_baseline \
  --baseline_checkpoint=9000 \
  --log_root=logs/MC_ImpactClassifier_Admittance_100Hz \
  --run_name=route1_scale1024_seed1 \
  --wandb_mode=online \
  --headless
```

The output directory is
`logs/MC_ImpactClassifier_Admittance_100Hz/<timestamp>_<run_name>/`; the same
directory name is used for the W&B run.

## Quiet evaluation

The primary fixed protocol is descending stairs at 0.5 m/s, 0.14 m step
height, 32 environments, a 2-second warm-up and a 20-second measurement at
the 200 Hz physics rate. Domain randomization and observation noise are off.
Use the same `eval_seed` for the paired baseline and adaptive runs.

Baseline:

```bash
conda run -n HIM python legged_gym/scripts/evaluate_mc_quiet.py \
  --experiment_name=MC_100Hz \
  --load_run=Aug08_18-20-03_baseline \
  --checkpoint=9000 \
  --num_envs=32 \
  --eval_scenario=stairs_down \
  --stair_difficulty=0.5 \
  --eval_command_x=0.5 \
  --warmup_seconds=2 \
  --eval_seconds=20 \
  --eval_seed=123 \
  --headless
```

Adaptive:

```bash
conda run -n HIM python legged_gym/scripts/evaluate_mc_impact_quiet.py \
  --checkpoint_path=logs/MC_ImpactClassifier_Admittance_100Hz/<run>/model_500.pt \
  --num_envs=32 \
  --eval_scenario=stairs_down \
  --stair_difficulty=0.5 \
  --eval_command_x=0.5 \
  --warmup_seconds=2 \
  --eval_seconds=20 \
  --eval_seed=123 \
  --headless
```

Report paired force/loading event mean, p95 and p99; base-acceleration RMS,
p95 and 5-100 Hz bands; touchdown and short re-contact counts; velocity
tracking error and tracking reward. The method is accepted at scale only when
impact attenuation and estimator stability are retained without degrading
vibration or tracking relative to the validated 256-environment operating
point.

## References

- [HIMLoco](https://github.com/OpenRobotLab/HIMLoco)
- [legged_gym](https://github.com/leggedrobotics/legged_gym)
