# MC ContactEstimator Stability Diagnostic

## 1. 研究问题

本实验用于解释 Impact-aware Learned Admittance 的 ContactEstimator 为何在
Stage-1 on-policy 训练中逐渐失去 impact 分离度。核心对照是在完全固定的
baseline contact validation set 上，比较：

- D1：Stage 0 warm-up 后继续用最新 on-policy rollout 做 supervised update；
- D2：Stage 0 warm-up 后冻结 ContactEstimator，只继续训练 compliance head 和 critic。

本实验不优化 classifier-to-control mapping，也不更改 PPO、reward、label、网络结构或
控制律。正式映射保持：

```text
pos_weight = 3.0
impact_logit_correction_scale = 0.10
p_control = sigmoid(raw_logit - 0.10 * log(3))
impact threshold = 5000 N/s
impact_gain = 2.0
```

## 2. Fixed validation 构建方法

- 初始化 checkpoint：`logs/MC_100Hz/Aug08_18-20-03_baseline/model_9000.pt`
- seed：1
- 环境：256
- Stage-0 ContactEstimator warm-up：500 policy steps
- validation collection：200 policy steps
- 动作：deterministic baseline `act_inference`
- compliance：强制为 0
- 样本数：51,200 env-transitions，即 204,800 个 per-leg classification samples
- 保存位置：`logs/validation/mc_contact_validation_seed_1_env_256_steps_200.pt`
- 内容：`obs_history`、`controller_state`、normalized axial force target、impact
  label、GT axial loading rate，以及 command metadata

D1 首次创建数据集。D2 为保持 Stage-1 起始物理进度一致，仍执行相同的 200-step
baseline collection trajectory，但评估时加载 D1 保存的同一份 tensor 数据。

验证在 iteration 0、20、40、60、80、100 执行。评估使用 `eval()` 和
`torch.inference_mode()`，分 batch 只做 forward；validation target 和 loading rate
不会进入任何 optimizer、rollout storage、reward、actor 或 controller。

PR AUC 和 ROC AUC 采用基于 PyTorch sort/cumulative sum 的轻量实现，没有引入
scikit-learn 等依赖。

## 3. 信息流安全检查

实验前后的架构保持：

```text
physical action       = 16
motion action         = 16
compliance action     = 4
policy action         = 20
controller state      = 16
contact estimate      = 8 = force(4) + raw impact logits(4)
motion_adapter_scale  = 0
base_actor_lr_scale   = 0
update_him_estimator  = False
```

新增 `transition_gt_axial_loading_rate` 只保存与当前诊断 target 对齐的 GT loading
rate；唯一读取位置是 validation collection。它没有进入 estimator input、actor、critic、
admittance controller、reward 或 PPO storage。

freeze 开关只包围 Stage-1 的
`contact_estimator.update(obs, controller_state, contact_target)`。Stage 0 的 500-step
warm-up 不受影响；compliance head 和 critic 的 PPO update 仍正常执行。

## 4. 实验配置与日志

| Experiment | Stage-1 estimator | Envs | Iterations | Seed | Fresh baseline |
|---|---:|---:|---:|---:|---:|
| D1 Continual | update | 256 | 100 | 1 | yes |
| D2 Frozen | frozen after warm-up | 256 | 100 | 1 | yes |

TensorBoard runs：

- D1：`Sep08_12-55-36_contact_stability_continual_seed_1_env_256_iter_100`
- D2：`Sep08_13-00-26_contact_stability_frozen_seed_1_env_256_iter_100`

两次均正常完成 warm-up、validation collection 和 100 iterations，无 NaN、dimension
mismatch 或 contact-force information leakage。D2 的 Stage-1 contact losses 为 0，验证
freeze 开关没有执行 estimator optimizer update。

相同 seed 的 GPU PhysX 轨迹不是 bitwise deterministic，因此两个独立 warm-up snapshot
有很小差异：D1/D2 iteration-0 fixed raw F1 相差 0.0075、logit separation 相差
0.0084。所有退化判断均优先采用各 run 在同一固定数据上的纵向变化；D2 的评估输入
tensor 与 D1 完全相同。

## 5. Online vs Fixed Validation

下表的 online 值是对应 20-iteration 窗口均值；validation 值是窗口结束点对固定数据集
的测量。

| Window | Continual Online F1 | Continual Val F1 | Frozen Online F1 | Frozen Val F1 |
|---|---:|---:|---:|---:|
| 0–19 | 0.4394 | 0.3224 @20 | 0.2568 | 0.3223 @20 |
| 20–39 | 0.3744 | 0.2947 @40 | 0.2446 | 0.3223 @40 |
| 40–59 | 0.2852 | 0.2892 @60 | 0.2577 | 0.3223 @60 |
| 60–79 | 0.2850 | 0.2830 @80 | 0.2592 | 0.3223 @80 |
| 80–99 | 0.2766 | 0.2847 @100 | 0.2743 | 0.3223 @100 |

### 5.1 Online classifier 五窗口

| Exp / Window | Precision | Recall | F1 | Raw logit sep. | GT ratio | Pred ratio |
|---|---:|---:|---:|---:|---:|---:|
| D1 0–19 | 0.3419 | 0.6356 | 0.4394 | 0.6939 | 0.2474 | 0.4619 |
| D1 20–39 | 0.2924 | 0.5227 | 0.3744 | 0.7255 | 0.2415 | 0.4299 |
| D1 40–59 | 0.2322 | 0.3708 | 0.2852 | 0.5152 | 0.2388 | 0.3814 |
| D1 60–79 | 0.2325 | 0.3693 | 0.2850 | 0.5287 | 0.2402 | 0.3815 |
| D1 80–99 | 0.2271 | 0.3548 | 0.2766 | 0.5335 | 0.2358 | 0.3685 |
| D2 0–19 | 0.3829 | 0.1945 | 0.2568 | 0.5735 | 0.2470 | 0.1256 |
| D2 20–39 | 0.3759 | 0.1821 | 0.2446 | 0.5752 | 0.2417 | 0.1171 |
| D2 40–59 | 0.3762 | 0.1968 | 0.2577 | 0.6008 | 0.2396 | 0.1254 |
| D2 60–79 | 0.3794 | 0.1975 | 0.2592 | 0.5850 | 0.2431 | 0.1265 |
| D2 80–99 | 0.3812 | 0.2149 | 0.2743 | 0.6033 | 0.2471 | 0.1393 |

D1 的 online F1 从 0.4394 降至 0.2766，raw logit separation 从 0.6939 降至
0.5335。D2 的 estimator 固定后，online F1 没有继续下降，反而从 0.2568 小幅升至
0.2743；raw separation 也保持在约 0.57–0.60。D2 的低 recall 说明仅靠 Stage-0
snapshot 不足以适配 adaptive contact distribution，但不支持“闭环分布必然让任意固定
classifier 随 iteration 持续退化”。

### 5.2 Fixed validation 详细趋势

| Exp | Iter | Raw P | Raw R | Raw F1 | Logit sep. | Prob. sep. | PR AUC | ROC AUC | Force MAE (N) | Force corr. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D1 | 0 | 0.3149 | 0.3460 | 0.3298 | 0.7695 | 0.1387 | 0.2813 | 0.7278 | 27.13 | 0.6066 |
| D1 | 20 | 0.2062 | 0.7381 | 0.3224 | 0.6908 | 0.1202 | 0.2409 | 0.6816 | 26.37 | 0.6089 |
| D1 | 40 | 0.1922 | 0.6317 | 0.2947 | 0.6920 | 0.1005 | 0.1965 | 0.6263 | 26.39 | 0.5996 |
| D1 | 60 | 0.1913 | 0.5922 | 0.2892 | 0.7162 | 0.0969 | 0.1869 | 0.6161 | 26.09 | 0.6160 |
| D1 | 80 | 0.1882 | 0.5702 | 0.2830 | 0.7365 | 0.0948 | 0.1833 | 0.6111 | 25.54 | 0.6250 |
| D1 | 100 | 0.1911 | 0.5576 | 0.2847 | 0.7269 | 0.0966 | 0.1856 | 0.6146 | 25.65 | 0.6308 |
| D2 | 0 | 0.3187 | 0.3260 | 0.3223 | 0.7612 | 0.1361 | 0.2805 | 0.7261 | 27.93 | 0.6008 |
| D2 | 20–100 | 0.3187 | 0.3260 | 0.3223 | 0.7612 | 0.1361 | 0.2805 | 0.7261 | 27.93 | 0.6008 |

D1 在固定分布上的 raw F1 下降 13.7%，PR AUC 下降 34.0%，ROC AUC 从
0.7278 降到 0.6146，raw probability separation 下降 30.3%。这不能由 on-policy
evaluation set 变难解释，因为 evaluation tensor 没有变化。它是 estimator parameter
drift/forgetting 的直接证据。

仅看 positive mean 与 negative mean 的差会低估退化：D1 final logit mean separation
仍有 0.7269，但 ROC/PR AUC 和 probability separation 明显下降，说明类内分布变宽、
重叠增加，同时整体 logit bias 发生变化。另一方面 fixed force MAE 从 27.13 N 改善到
25.65 N，表明 continual update 并非整体失效，而是在共同网络参数上改善 force task 的
同时损害了 held-out impact ranking。

D2 的所有 fixed metrics 在每个 evaluation point 完全不变，验证 fixed evaluation
本身没有状态更新、副作用或数据漂移。

## 6. GT loading-rate label histogram

### 6.1 Fixed baseline validation

| Axial loading-rate bin (N/s) | Ratio |
|---|---:|
| 0–3000 | 80.032% |
| 3000–4000 | 3.077% |
| 4000–4500 | 1.192% |
| 4500–4750 | 0.514% |
| 4750–5000 | 0.465% |
| 5000–5250 | 0.424% |
| 5250–5500 | 0.400% |
| 5500–6000 | 0.743% |
| 6000–8000 | 2.328% |
| >8000 | 10.823% |

```text
Fixed impact GT ratio                         = 14.719%
Fixed near threshold [4500, 5500] ratio      = 1.804%
Fixed very-near [4750, 5250] ratio           = 0.890%
```

### 6.2 D1 online 全 100 iterations 均值

| Axial loading-rate bin (N/s) | Ratio |
|---|---:|
| 0–3000 | 69.886% |
| 3000–4000 | 3.191% |
| 4000–4500 | 1.451% |
| 4500–4750 | 0.703% |
| 4750–5000 | 0.685% |
| 5000–5250 | 0.663% |
| 5250–5500 | 0.649% |
| 5500–6000 | 1.244% |
| 6000–8000 | 4.440% |
| >8000 | 17.088% |

```text
Online near threshold [4500, 5500] ratio     = 2.700%
Online very-near [4750, 5250] ratio          = 1.348%
```

固定 baseline 的 positive ratio 为 14.7%，而 adaptive online 约为 23.6–24.7%，且
online >8000 N/s 样本明显更多。这证明 baseline 与 adaptive contact distribution
确实不匹配。但边界附近样本只占全部样本约 1.8%（fixed）或 2.7%（online）；即使它们
全部有较高 label sensitivity，也不足以解释约 0.16 的 online F1 下跌和 fixed AUC 的
同步下降。因此 threshold 边界噪声不是本实验观察到的首要退化源。

`single_step_positive_ratio` 本轮未实现（SKIPPED）；在近阈值比例较低的结果下，它不
影响当前主结论。没有增加 hysteresis，也没有修改 label。

## 7. Compliance、quiet 与 locomotion

下面报告最后 20 iterations 均值。tracking 项是现有 episode reward component，并非
直接速度误差；两次使用相同 reward 配置。

| Experiment | Comp impact (mm) | Comp no-impact (mm) | R_comp | Comp p95 (mm) | Comp max (mm) | Saturation | dF peak (N/s) | F peak (N) | Base Acc (m/s²) | Lin track | Yaw track | Base height | Episode reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D1 Continual | 1.126 | 0.726 | 1.552 | 3.836 | 14.784 | 0.000001 | 4870.6 | 64.01 | 4.818 | 1.311 | 0.641 | -0.0491 | 25.76 |
| D2 Frozen | 1.037 | 0.705 | 1.471 | 3.895 | 18.602 | 0.000030 | 5032.3 | 64.68 | 4.874 | 1.355 | 0.667 | -0.0200 | 26.56 |

Freezing 保存了 baseline validation 能力，但没有产生更好的整体闭环工作点：D2 的
late online recall 更低，R_comp 更低，三个 quiet proxy 均略差，compression max 和
saturation 也更高；其 tracking/reward 略好。D1/D2 都没有明显持续撞击 20 mm limit。

这里没有配对的 compliance-off baseline physical evaluation，因此不能用该 100-iteration
对照宣称最终静音改善，只能比较 D1 与 D2 的相对趋势。

## 8. 原因判断

### Distribution shift

**存在，但不是单独解释。** Fixed GT ratio 14.7% 与 online 约 24% 的差异，以及高
loading-rate tail 的增加，证明 adaptive compliance policy 改变了接触分布。D2 的
online 性能显著低于其 baseline fixed performance，也说明 Stage-0 estimator 对 adaptive
distribution 的直接泛化有限。

但 D2 的 online F1/separation 没有随 iterations 持续下降；因此“on-policy 数据越来越难”
不能单独解释 D1 的长期退化。

### Forgetting / parameter drift

**明确存在，是持续退化的主因。** D1 在不变 fixed tensors 上的 PR AUC、ROC AUC、
probability separation 和 F1 同步下降；D2 冻结后全部完全稳定。这是 estimator 参数更新
损害旧 baseline decision boundary 的直接因果对照。

同时 D1 的 fixed force regression 改善，提示 drift 可能包含两部分：追逐最新 on-policy
distribution，以及 force/classification 共享网络参数下的 supervised task interference。
本实验没有把这两种 update-level 机制进一步拆开。

### Label noise

**不是当前主因。** near-threshold 比例较低。单帧 spike 尚未统计，所以不能断言 label
完全无噪声，但目前没有修改 threshold、soft label 或 hysteresis 的证据基础。

### Current-event observability limit

**不是退化的充分解释。** Warm-up snapshot 在 fixed set 上 ROC AUC 约 0.73、logit
separation 约 0.76，证明当前 6-frame proprioceptive history + controller state 有非随机的
impact 排序能力。绝对 AUC 并不高，observability 可能限制最终上限，但无法造成冻结模型
不变、continual 模型下降的差异。

### 总结

主结论是 **MIXED：parameter drift/forgetting 为主，baseline-to-adaptive distribution
shift 为次**。Continual adaptation 在早期明显提高 online recall/F1，说明完全冻结不是
最终解；但现有 `1e-3`、仅最新 rollout、每次 PPO update 多 minibatch/epoch 的持续监督
更新会逐步破坏 held-out impact ranking。

## 9. 后续决策

### 是否建议 Replay

**YES。** Fixed validation 的 ranking degradation 已满足 replay/mixed old-new batch 的
诊断前提。但不能把本报告的 held-out validation set 用作 replay；后续必须另行采集、与
validation 完全不重叠的 baseline training replay set。

### 是否建议降低 estimator LR

**YES。** 这是最小、最高信息量的下一步对照。建议先在相同 256-env / 100-iteration /
seed-1 设计中，仅将 Stage-1 estimator LR 从 `1e-3` 降到 `3e-4`，Stage-0 warm-up 配置和
其余参数保持不变，并检查 fixed PR/ROC AUC 是否停止下滑，同时 online recall 是否仍能
适配。

### 是否建议修改 label

**NO。** near-threshold ratio 不高，尚无足够证据支持 threshold、soft label、hysteresis
或 future target 修改。

### 是否建议扩大 num_envs

**暂不建议 1024/4096。** 更多环境会提高每次 on-policy supervised batch 的覆盖，但不会
自动阻止参数 drift；在当前更新策略下，它还会浪费大规模 PPO 计算预算。建议继续
256-env 诊断，先验证 lower LR；若仍出现 fixed degradation，再做一个使用独立 baseline
training replay 的 mixed old/new 对照。只有 fixed 与 online 两条曲线都稳定后再扩容。

## 10. 最终结论

```text
Stability diagnostic:                 PASS

Primary degradation source:           MIXED
                                       (FORGETTING / PARAMETER DRIFT primary,
                                        DISTRIBUTION_SHIFT secondary)

Fixed-validation degradation:         YES

Does freezing estimator help:         MIXED
                                       (preserves fixed performance, but hurts
                                        online adaptation and does not improve
                                        closed-loop physical metrics)

Near-threshold label ratio:            1.804% fixed / 2.700% online
Very-near-threshold label ratio:       0.890% fixed / 1.348% online

Is replay justified next:              YES
Is lower estimator LR justified next:  YES
Is label redesign justified now:       NO

Recommend further 256-env experiments: YES
Recommend 1024/4096 training:           NO
```

下一步只建议两个有信息价值的动作，按顺序执行：

1. 256 env / 100 iterations 配对测试 Stage-1 ContactEstimator LR `3e-4`；
2. 若 fixed PR/ROC AUC 仍下降，再用**独立训练集**测试少量 baseline replay 与最新
   on-policy supervised batch 的混合，不污染本固定 validation set。

不建议同时修改 label、network、history 或 classifier-to-control mapping。
