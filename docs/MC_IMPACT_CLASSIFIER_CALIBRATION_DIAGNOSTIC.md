# MC Impact Classifier Probability Calibration Diagnostic

## 1. 实验目的

Impact classifier 使用 `pos_weight=3` 的 weighted BCE 训练。理论最优 logits
相对未加权 posterior logits 会产生约 `+log(3)` 的偏移，因此当前控制器使用：

```text
p_full = sigmoid(raw_logit - log(3))
```

但该理论修正在有限训练阶段不一定与网络的实际 logit 偏移一致。本实验不改变训练、
reward 或控制律，只记录 GT 正负样本对应的 raw logits，并对同一批 logits 离线比较
不同 correction，判断低 recall 来自概率校准还是分类器缺乏区分能力。

## 2. 实验配置

| 配置 | 数值 |
|---|---:|
| 分支 | `mc-impact-classifier-admittance-v1` |
| baseline checkpoint | `logs/MC_100Hz/Aug08_18-20-03_baseline/model_9000.pt` |
| environments | 64 |
| Stage 0 warmup | 500 steps |
| Stage 1 iterations | 20（TensorBoard step 0–19） |
| impact BCE pos_weight | 3.0 |
| impact threshold | 5000 N/s |
| impact probability threshold | 0.5 |
| impact gain | 2.0 |
| compliance init std | 0.15 |
| compliance activation gain | 6.0 |

运行命令：

```bash
python legged_gym/scripts/train.py \
  --task=mc_learned_admittance_100hz \
  --num_envs=64 \
  --max_iterations=20 \
  --headless
```

运行结果为 PASS。实验从原始 `MC_100Hz` baseline fresh 初始化，没有 resume 之前的
impact-classifier checkpoint。Stage 0 明确将 compliance 强制为零，随后 Stage 1
完成 20 iterations；没有 NaN 或维度错误。

Warmup loss：

| Step | Force loss | Impact loss |
|---:|---:|---:|
| 50 | 0.86398 | 1.12805 |
| 100 | 0.31354 | 0.89880 |
| 250 | 0.22707 | 0.81836 |
| 500 | 0.24042 | 0.80818 |

本次 run：

```text
logs/MC_ImpactClassifier_Admittance_100Hz/
  Sep07_20-09-17_impact_classifier_admittance_v1
W&B run: nki148dl
```

以下主要结论使用最后 5 个 iterations（15–19）的均值；同时给出 10–19 的均值，
避免只读取最后一个 console 点。

## 3. Raw classifier 结果

`p_raw = sigmoid(raw_logit)`，分类阈值固定为 0.5。

| Metric | Iter 15–19 | Iter 10–19 |
|---|---:|---:|
| GT positive ratio | 0.245801 | 0.248910 |
| Raw predicted ratio | 0.544108 | 0.550187 |
| Raw precision | 0.341449 | 0.340482 |
| Raw recall | 0.755937 | 0.752790 |
| Raw F1 | 0.469204 | 0.467682 |
| Positive raw logit mean | 0.247243 | 0.236287 |
| Negative raw logit mean | -0.523774 | -0.491294 |
| Raw logit separation | 0.771017 | 0.727581 |
| Positive raw logit p50 | 0.351807 | 0.337056 |
| Negative raw logit p50 | -0.065862 | -0.043996 |
| Positive raw logit p95 | 0.946567 | 0.937591 |
| Negative raw logit p95 | 0.819120 | 0.824092 |
| Positive raw probability mean | 0.561400 | 0.560282 |
| Negative raw probability mean | 0.420711 | 0.428276 |
| Raw probability separation | 0.140689 | 0.132006 |
| Overall raw probability mean | 0.455294 | 0.461111 |
| Axial force MAE | 34.4296 N | 34.6811 N |

## 4. Full correction 结果

`p_full = sigmoid(raw_logit - log(3))`，分类阈值仍为 0.5。

| Metric | Iter 15–19 | Iter 10–19 |
|---|---:|---:|
| Positive probability mean | 0.313034 | 0.311792 |
| Negative probability mean | 0.222134 | 0.226047 |
| Predicted ratio | 0.012549 | 0.011947 |
| Precision | 0.369568 | 0.349713 |
| Recall | 0.019779 | 0.018178 |
| F1 | 0.037174 | 0.034120 |

正式的 `Estimator/impact_*` 指标与 `fullcorr_*` 使用相同 probability conversion，
二者一致。完整 correction 将预测正类比例从 raw 的 54.4% 压低到 1.25%，远低于
24.6% 的 GT ratio，几乎消除了 recall。

## 5. Correction sweep

同一批 raw logits 使用 `p_c = sigmoid(raw_logit - fraction * log(3))`；threshold
固定为 0.5。下表为最后 5 iterations 均值。

| Correction | Pred ratio | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| 0.00 log(3) | 0.544108 | 0.341449 | 0.755937 | **0.469204** |
| 0.25 log(3) | 0.385075 | 0.362683 | 0.568557 | 0.441334 |
| 0.50 log(3) | 0.200472 | 0.386428 | 0.315029 | 0.344680 |
| 0.75 log(3) | 0.066992 | 0.417500 | 0.113628 | 0.177121 |
| 1.00 log(3) | 0.012549 | 0.369568 | 0.019779 | 0.037174 |

Iter 10–19 得到相同排序：对应 F1 分别为 0.467682、0.436727、0.342100、
0.171583、0.034120。指定 sweep 中最佳候选为 `0.00 * log(3)`；没有证据支持当前
网络使用正的 correction，且 correction 越强，F1 越低。

作为额外诊断，对 `p_raw` 做了 threshold sweep：

| Raw threshold | Pred ratio | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|
| 0.3 | 0.754639 | 0.308330 | 0.946621 | 0.464178 |
| 0.4 | 0.673584 | 0.323398 | 0.886052 | **0.472768** |
| 0.5 | 0.544108 | 0.341449 | 0.755937 | 0.469204 |
| 0.6 | 0.296143 | 0.373668 | 0.450140 | 0.406327 |
| 0.7 | 0.059212 | 0.422409 | 0.101414 | 0.162187 |

该 threshold sweep 只证明 raw scores 包含排序信息；本实验没有修改正式 threshold。

## 6. Logit 分布分析

GT positive 的 mean logit 比 GT negative 高 0.771，raw probability separation 为
0.141，并且 raw classifier 在固定 0.5 threshold 下达到 0.756 recall。这说明网络
已经学习到有用的 impact 排序，问题不是“完全不可分”。

区分度仍非完美：negative p95 为 0.819，与 positive 分布明显重叠；raw predicted
ratio 54.4% 也高于 GT ratio 24.6%，导致 precision 只有 0.341。因此当前分类器是
有用但尚未达到高精度的中等区分度模型。

## 7. Calibration 判断

- **当前 classifier 是否有学习能力：**有。正负 mean logits 和 raw F1 均显示明确区分度。
- **`-log(3)` 是否过强：**是。它把 recall 从 0.756 降到 0.020、F1 从 0.469 降到 0.037。
- **哪个 correction 候选最好：**本次预定义 sweep 中为 `0.00 * log(3)`。如果下一轮
  需要更精细选择，可只在 `0–0.25 * log(3)` 间增加密集点，但本次数据没有显示正
  correction 能优于零 correction。
- **是否值得立即长训当前配置：**否。当前 full correction 使有效 impact activation
  几乎消失；先做 correction ablation 更有信息价值。
- **下一步方向：**优先验证零 correction（及可选的很小 correction）短实验，而不是
  重新设计 estimator。分类器仍有较大分布重叠，待 calibration 确定后再判断是否增加
  warmup、容量或调整 target。

该结果属于 **Case A：classifier 有明显区分度，但 full correction 过强**。Raw 模式
同时存在一定过报，但 correction sweep 没有出现“中间 correction 提升 F1”，因此不属于
典型 Case C。

## 8. 物理状态

| Metric | Iter 15–19 | Iter 10–19 |
|---|---:|---:|
| compression after impact | 0.816700 mm | 0.743720 mm |
| compression after no impact | 0.528880 mm | 0.496454 mm |
| compression p95 | 2.800203 mm | 2.643393 mm |
| compression saturation ratio | 0.000000 | 0.000000 |
| compression over 10 mm ratio | 0.000033 | 0.000049 |
| GT 3D force peak mean | 64.4042 N | 64.1811 N |
| GT 3D loading peak mean | 5091.8412 N/s | 5106.5735 N/s |
| GT base acceleration peak mean | 4.9561 m/s² | 4.9085 m/s² |

因果诊断保持：`y_(t-1) -> obs_t -> logits_t -> probability_t -> transition t ->
compression_t -> y_t`。`compression_after_*` 仍以 `y_(t-1)` 为 mask；impact label 是
current-event classification，不是 future prediction。

本次只有 20 iterations，目的为 classifier calibration diagnosis。上述物理指标仅用于
安全性和信息流检查，不能据此宣称最终静音性能。

## 9. 信息边界与结论

新增 raw/full/sweep 统计只在 diagnostics cache 中计算。GT contact/impact 没有进入 actor、
critic、ContactEstimator input、controller state 或 admittance 决策；训练 loss、reward、
PPO/GAE、action/observation/controller-state 维度均未改变。

最终诊断：

```text
Experiment: PASS
Classifier separability: GOOD（有明确但非完美的区分度）
Full log(3) correction: TOO STRONG
Best diagnostic correction candidate: 0.00 * log(3)
Recommend long training now: NO
Next recommended action: 对 zero/small correction 做短 ablation，再决定长训或 estimator 调整
```
