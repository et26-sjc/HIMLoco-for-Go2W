# MC Learned Admittance v1

Branch: `mc-learned-admittance-v1`

This branch extends the trained `mc_100hz` HIMLoco baseline with a learned,
sensorless per-leg admittance controller. The baseline branch is intentionally
left unchanged.

## 1. Hard information-flow boundary

### Deployable signals

The deployed policy/controller may use only:

- original 6-frame HIM proprioceptive history: 342D (`57 x 6`);
- original HIM estimate: base velocity 3D + latent 16D;
- learned contact estimate 8D: axial compressive force (4) + positive axial
  loading rate (4), both normalized;
- complete normalized admittance state 16D:
  - compression/max compression (4),
  - compression velocity/max compression velocity (4),
  - compliance activation alpha (4),
  - slow support-force bias/contact-force scale (4);
- policy output: motion 16D + compliance 4D.

### Training-only signals

Isaac Gym ground-truth contact force may be used only for:

- ContactEstimator supervision;
- quiet-impact rewards;
- critic privileged information already present in the MC baseline;
- offline/evaluation metrics.

Ground-truth contact force MUST NOT enter actor input or the deployed admittance
controller.

## 2. Dimensions

```text
original one-step proprioception       57
original HIM history              57 x 6 = 342
original HIM estimate             3 + 16 = 19
controller state                         16
contact estimate                           8

physical/motion action                    16
compliance policy action                   4
policy action                             20
physical MC DOFs                          16
```

`env.num_actions` intentionally remains 16. `env.num_policy_actions` is 20.
This prevents the extra policy outputs from corrupting DOF-sized torque, PD,
motor-delay, action-history or noise buffers.

## 3. Policy flow

```text
history 342
  |
  +--> original HIM estimator --> v_hat(3), z(16)
  |
  +--> ContactEstimator(history, controller_state16)
                         --> F_axial_hat(4), dF_axial_hat(4)

current proprio57 + v_hat + z + contact_hat + controller_state16
  |
  +--> frozen original baseline actor ----------> motion16
  +--> compliance head --------------------------> compliance4
  +--> optional motion adapter (scale=0 in v1 Stage 1)
```

The ContactEstimator receives no force sensor signal at inference time.

## 4. Training target definition

At every 200 Hz physics substep training computes both:

1. full 3-D wheel force magnitude/loading rate for quiet rewards and metrics;
2. compressive force projected onto the current hip-to-wheel leg axis and its
   positive loading rate for ContactEstimator supervision.

The estimator target for each 100 Hz transition is the peak axial force and peak
positive axial loading rate over the two physics substeps, normalized and
clipped. This makes it a short-horizon impact predictor rather than a direct
instantaneous force-sensor substitute.

## 5. Admittance law

For each leg:

```text
M*x_ddot + D(alpha)*x_dot + K(alpha)*x
    = alpha * gate(dF_hat) * F_transient_hat

K(alpha) = K_max - alpha*(K_max-K_min)
D(alpha) = 2*zeta*sqrt(M*K(alpha))
```

A slowly varying estimated support-force baseline is subtracted before force
enters the virtual system, so ordinary support load should not continuously
compress the leg. The policy sees this internal force-bias state explicitly.

The 100 Hz policy holds `alpha` and contact estimate over two 200 Hz physics
substeps. The virtual second-order dynamics are integrated at 200 Hz. Axial
compression is mapped to HIP/KNEE offsets by a damped least-squares Jacobian
using MC URDF lengths 0.20 m and 0.22 m, then added to the original HIM target
before the unchanged fixed-PD controller.

## 6. Staged training

### Stage 0: contact-estimator warm-up

Default: 500 policy transitions.

- load the trained `MC_100Hz` checkpoint;
- deterministic original baseline motion only;
- force all four compliance actions to exactly zero;
- no PPO update, no critic update, no HIM update;
- train only ContactEstimator from simulator-only axial force/loading labels.

With 4096 environments, 500 transitions provide roughly 2 million labeled
samples while the physical behavior remains the original baseline.

### Stage 1: compliance-only RL

- original 16D actor learning rate = 0;
- original HIM estimator frozen;
- motion-adapter scale = 0;
- policy std frozen;
- first 16 std values are migrated from the baseline checkpoint;
- four new compliance std values remain fixed at 0.05;
- ContactEstimator continues supervised learning at its own independent LR;
- PPO trains the 4D compliance head and critic;
- all original HIMLoco rewards remain active, with added impact/loading/base-acc
  and compliance-usage penalties.

This stage asks a deliberately clean question:

> Can learned admittance reduce stair impact/vibration while the original
> locomotion command itself is held fixed?

### Stage 2: optional co-adaptation

Only after Stage 1 succeeds should a separate experiment enable a small motion
adapter (e.g. scale 0.05) and/or a very small baseline-actor LR. This tests
whether compensating the nominal motion for controller-induced leg shortening
adds value without conflating the primary compliance result.

## 7. Training

A trained baseline must exist under `logs/MC_100Hz` unless the initializer
configuration is changed.

```bash
python legged_gym/scripts/train.py \
  --task=mc_learned_admittance_100hz \
  --headless
```

Recommended first smoke run:

```bash
python legged_gym/scripts/train.py \
  --task=mc_learned_admittance_100hz \
  --num_envs=64 \
  --max_iterations=20 \
  --headless
```

Before a long run verify:

1. all four semantic leg mappings resolve correctly;
2. physical action=16 and policy action=20;
3. controller state=16 and contact estimate=8;
4. baseline actor/HIM/critic and first 16 std values migrate successfully;
5. deterministic compliance begins at zero;
6. warm-up executes baseline motion with alpha forced to zero;
7. contact force/loading losses decrease during warm-up;
8. Stage-1 baseline motion output remains identical to the loaded checkpoint;
9. compliance remains sparse on flat terrain and increases around stair impacts;
10. velocity/yaw tracking remains near baseline while force/loading/base-acc fall.

## 8. Evaluation and deployment status

The existing `QuietMC` physics-rate metrics remain the authoritative final
silent-locomotion evaluator. The adaptive environment already computes training
force/loading/base-acc quantities, but a shared adaptive wrapper for the full
existing touchdown/impulse/quiet-score evaluator should still be added before
final experimental tables.

The adaptive JIT policy exports only deployable information and returns:

```text
input : history342 + controller_state16
output: policy_action20 + contact_estimate8
```

`mujoco/pdandrl.py` is still the old 16D deployment loop. Before sim-to-sim or
hardware claims it must be upgraded to maintain the same 16D internal admittance
state, execute the same second-order dynamics/Jacobian mapping, and consume the
JIT ContactEstimator output. MuJoCo/real contact force must never be used as a
controller input.

## 9. Research note on prediction horizon

The estimator predicts impact over the upcoming 10 ms policy transition. The
target is therefore partly action-dependent. Stage 1 intentionally fixes the
baseline motion mapping, making this dependence relatively clean. If Stage 2
allows locomotion co-adaptation, nominal upcoming motion action should be tested
as an additional deployable estimator context input.
