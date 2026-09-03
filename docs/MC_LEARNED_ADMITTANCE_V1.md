# MC Learned Admittance v1

Branch: `mc-learned-admittance-v1`

This branch extends the trained `mc_100hz` HIMLoco baseline with a learned,
sensorless per-leg admittance controller.  The baseline branch is intentionally
left unchanged.

## 1. Hard information-flow boundary

### Deployable signals

The deployed policy/controller may use only:

- original 6-frame HIM proprioceptive history (342D), built from the unchanged
  57D MC actor frame;
- original HIM estimator outputs: base velocity estimate (3D) + latent (16D);
- learned contact estimator output (8D): normalized axial compressive force for
  four legs + positive axial loading rate for four legs;
- internal admittance/controller state (12D): leg compression (4), compression
  velocity (4), previous/current compliance activation (4);
- policy outputs: original motion action (16D) + compliance action (4D).

### Training-only privileged signals

Isaac Gym ground-truth contact force may be used only for:

- ContactEstimator supervision;
- quiet-impact rewards;
- critic privileged information already present in the MC baseline;
- offline/evaluation metrics.

Ground-truth contact force MUST NOT be passed to the actor or the deployed
admittance controller.

## 2. Dimensions

```text
original one-step proprioception       57
original HIM history              57 x 6 = 342
original HIM estimate             3 + 16 = 19
controller state                         12
contact estimate                           8

physical/motion action                    16
compliance policy action                   4
policy action                             20
physical MC DOFs                          16
```

`env.num_actions` intentionally remains 16.  `env.num_policy_actions` is 20.
This avoids corrupting the existing DOF-sized torque, PD-gain, motor-delay and
observation buffers.

## 3. Policy/state flow

```text
proprio history 342
      |
      +--> original HIM estimator --> v_hat(3), z(16)
      |
      +--> ContactEstimator(history, controller_state)
                              --> F_axial_hat(4), dF_axial_hat(4)

current proprio 57 + v_hat + z + contact_hat + controller_state
      |
      +--> original baseline actor ------------------> motion 16
      |
      +--> compliance head --------------------------> compliance 4
      |
      +--> optional motion adapter (disabled in v1 Stage 1)
```

The ContactEstimator receives no force sensor signal at inference time.

## 4. Ground-truth target definition

At every 200 Hz physics substep, training computes both:

1. full 3-D wheel-force magnitude, used for quiet-impact reward/metrics;
2. compressive force projected onto the current hip-to-wheel leg axis, used as
   the ContactEstimator target because this is the force that physically drives
   axial leg compression.

Positive loading rates are computed at 200 Hz.  The estimator target for each
100 Hz transition is the per-leg peak axial force and peak positive axial
loading rate over the two physics substeps, normalized and clipped.

This makes the estimator a short-horizon impact predictor rather than a direct
force sensor replacement at one instantaneous sample.

## 5. Admittance law

For each leg:

```text
M * x_ddot + D(alpha) * x_dot + K(alpha) * x
    = alpha * gate(dF_hat) * F_transient_hat
```

with

```text
K(alpha) = K_max - alpha * (K_max - K_min)
D(alpha) = 2 * zeta * sqrt(M * K(alpha))
```

`alpha in [0, 1]` is the learned compliance action.  A slowly varying estimated
support-force baseline is removed before the transient force drives the virtual
system.  Therefore normal static/routine support load should not continuously
compress the leg.

The 100 Hz policy holds `alpha` and the estimated contact state for two 200 Hz
physics substeps.  The virtual second-order system is integrated at 200 Hz.

The resulting axial compression is mapped to HIP/KNEE target offsets using a
damped least-squares Jacobian based on the MC URDF leg lengths (0.20 m, 0.22 m).
Offsets are added to the original HIM target and clipped to the existing joint
safety envelope before the original fixed-PD controller computes torque.

## 6. Stage-1 training policy

The default v1 task intentionally preserves locomotion:

- initialize original actor/HIM/critic weights from latest `logs/MC_100Hz`;
- freeze original 16D locomotion actor (`base_actor_lr_scale = 0`);
- freeze original HIM estimator (`update_him_estimator = False`);
- disable motion adapter contribution (`motion_adapter_scale = 0`);
- train ContactEstimator with supervised GT axial contact targets;
- train 4D compliance head with PPO;
- train critic and policy action std;
- retain all original HIMLoco rewards and add quiet/compliance penalties.

Thus Stage 1 asks one clean question:

> Can the robot reduce impact/vibration by learning when and how much admittance
> to enable, while leaving the trained 16D locomotion command unchanged?

Only after this is validated should a Stage-2 co-adaptation experiment enable a
small motion adapter / small baseline-actor learning rate.

## 7. Default training command

The latest trained 100 Hz baseline must exist under `logs/MC_100Hz` because the
new runner migrates compatible baseline actor/HIM/critic weights automatically.

```bash
python legged_gym/scripts/train.py \
  --task=mc_learned_admittance_100hz \
  --headless
```

To resume an adaptive run, use the normal `--resume --load_run ... --checkpoint ...`
arguments.  Resume loads the adaptive checkpoint, not the baseline initializer.

## 8. Checks before a long training run

Use a small number of environments/iterations first and verify:

1. startup prints four semantic leg mappings with valid foot/hip/knee handles;
2. physical action dimension is 16 and policy action dimension is 20;
3. baseline checkpoint migration copies actor/HIM/critic and first 16 std values;
4. deterministic compliance mean begins at zero;
5. estimated force begins near zero rather than spuriously activating admittance;
6. original 16D motion output matches the loaded baseline when Stage-1 adapter is off;
7. ContactEstimator force/loading losses decrease;
8. compliance remains sparse on flat terrain and activates around stair impacts;
9. velocity/yaw tracking remains close to `mc_100hz`;
10. 3-D impact peak/loading rate/base acceleration improve on stairs.

Recommended first smoke run:

```bash
python legged_gym/scripts/train.py \
  --task=mc_learned_admittance_100hz \
  --num_envs=64 \
  --max_iterations=20 \
  --headless
```

## 9. Evaluation and deployment status

The existing `QuietMC` infrastructure remains the authoritative physics-rate
quiet evaluator.  The adaptive task currently computes the training signals it
needs, but a shared/adaptive wrapper for the full existing touchdown/impulse/
quiet-score evaluation should still be added before final experiments.

The JIT exporter now exports `(policy_action20, contact_estimate8)` from only
342D proprioceptive history + 12D controller state.

`mujoco/pdandrl.py` is still the old 16D deployment loop and must be upgraded to:

1. maintain the 12D admittance state locally;
2. call the adaptive JIT policy with history + controller state;
3. split policy action into motion16/compliance4;
4. integrate the exact same `MCLearnedAdmittance` dynamics locally;
5. apply resulting HIP/KNEE offsets before fixed PD;
6. never read MuJoCo contact force as a controller input.

Do not claim sim-to-sim or hardware deployability until that path has been
implemented and tested.

## 10. Known research point

The estimator target is the impact generated over the upcoming 10 ms policy
transition.  This gives the controller predictive value, but part of the target
is action-dependent.  In Stage 1 the baseline motion command is fixed, making
this relationship relatively clean.  A later co-adaptation stage should test
whether including nominal upcoming motion action as an estimator context input
improves prediction when the locomotion policy itself is allowed to change.
