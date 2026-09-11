# MC Impact Classifier-to-Control Mapping Ablation

## 1. 实验目的

本实验寻找适合 Impact-aware Admittance 的 classifier-to-control 工作点。目标不是单独
最大化分类 F1，而是在保持 recall 和 locomotion 的同时，让真实 impact 后的 compression
高于 no-impact compression，并避免长期软化、压缩饱和和物理冲击指标恶化。

实验严格保持以下架构：16D physical action、16D frozen HIMLoco locomotion action、4D
compliance action、20D policy action、8D contact estimate 和 16D controller state。GT
contact/impact 只用于监督、reward 和 diagnostics，没有进入 actor、estimator input 或
controller。

## 2. 当前方法与实验参数化

ContactEstimator 仍输出：

```text
[normalized axial force(4), raw impact logits(4)]
```

Impact BCE 保持 `pos_weight=3.0`。实验只改变推理/控制映射：

```text
p_control = sigmoid(raw_logit - correction_scale * log(3))
drive = beta * (1 + impact_gain * p_control) * transient_force
```

新增 `impact_logit_correction_scale` 仅为控制映射参数，不进入 estimator loss。离线
correction sweep 始终以完整 `log(3)` 为基准，不受 active scale 影响。

所有 run 都满足：

- fresh 初始化自 `logs/MC_100Hz/Aug08_18-20-03_baseline/model_9000.pt`；
- `resume=False`，不同参数之间不继承 adaptive checkpoint；
- Stage 0 为 500-step deterministic baseline warmup，compliance 强制为零；
- baseline actor、HIM estimator冻结，motion adapter scale为0；
- `impact threshold=5000 N/s`、`compliance std=0.15`、beta gain 6、M/D/K、reward、PPO
  时间结构均保持不变。

本次共完成 13 个有效 GPU run：8 个 correction Phase-A run、3 个新增 impact-gain
run（gain=2复用 correction run）、2 个 Phase-B run。另有一个256-env run因交互中断
停在 iter 12，已排除且没有 resume。

主表使用每个 run 最后 20% iterations 的均值：30-iteration run 对应 iter 24–29，
100-iteration run 对应 iter 80–99。最后5轮也单独解析；结论与最后20%一致。

Compliance Selectivity Ratio：

```text
R_comp = compression_after_impact_mm /
         (compression_after_noimpact_mm + epsilon)
```

## 3. Phase A：Correction粗筛与多seed复核

首先固定 `impact_gain=2.0`，使用64 env、30 iterations。0.10和0.20使用seed 1/2/3，
其余候选为seed 1粗筛。

| Exp | Correction | Gain | Precision | Recall | F1 | GT ratio | Pred ratio | Comp impact (mm) | Comp no-impact (mm) | R_comp | dF peak (N/s) | F peak (N) | Base Acc (m/s²) | Saturation | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| C00-s1 | 0.00 log(3) | 2.0 | 0.347 | 0.776 | **0.479** | 0.260 | 0.581 | 1.090 | 0.760 | 1.434 | 5221.0 | 65.171 | 4.852 | 0 | 淘汰：过度激活、loading最高 |
| C10-3seed | 0.10 log(3) | 2.0 | 0.340±0.011 | **0.683±0.032** | **0.452±0.017** | 0.254±0.006 | 0.511±0.020 | 0.929±0.211 | 0.650±0.144 | 1.427±0.009 | 4993±228 | 64.207±0.435 | 4.752±0.103 | 0 | **选择** |
| C20-3seed | 0.20 log(3) | 2.0 | 0.347±0.016 | 0.585±0.051 | 0.434±0.027 | 0.253±0.006 | 0.426±0.028 | 0.919±0.166 | 0.636±0.108 | **1.443±0.023** | 4993±244 | 64.202±0.645 | 4.738±0.095 | 0 | 次选：选择性略高但recall损失明显 |
| C25-s1 | 0.25 log(3) | 2.0 | 0.365 | 0.594 | 0.451 | 0.261 | 0.424 | 1.124 | 0.762 | 1.475 | 5145.2 | 64.857 | 4.781 | 0 | 淘汰：无额外quiet收益、尾部较大 |

最后5轮的三seed统计也保持相同排序：

| Correction | Precision | Recall | F1 | Raw logit separation | Raw probability separation | R_comp |
|---|---:|---:|---:|---:|---:|---:|
| 0.10 log(3) | 0.338±0.011 | **0.684±0.032** | **0.451±0.017** | 0.696±0.070 | 0.124±0.011 | 1.432±0.007 |
| 0.20 log(3) | 0.347±0.016 | 0.584±0.052 | 0.433±0.026 | 0.702±0.074 | 0.125±0.012 | **1.447±0.024** |

0.20相对0.10只提高约0.015的 `R_comp`，却损失约0.10 recall。两者force/loading/
base acceleration差异均小于seed波动，因此按约束式选择保留0.10。

### Admittance与locomotion

最后5轮三seed均值：

| Metric | Corr 0.10 | Corr 0.20 |
|---|---:|---:|
| impact probability mean | 0.447±0.006 | 0.420±0.008 |
| impact probability p95 | 0.692±0.013 | 0.666±0.013 |
| impact active ratio | 0.512±0.022 | 0.425±0.032 |
| alpha mean | 0.059±0.002 | 0.061±0.004 |
| effective alpha mean | 0.217±0.010 | 0.223±0.010 |
| compression p95 | 3.135±0.490 mm | 3.126±0.473 mm |
| compression max | 10.523±0.719 mm | 10.201±0.649 mm |
| compression saturation ratio | 0 | 0 |
| compression >10mm ratio | 0.000136±0.000106 | 0.000130±0.000091 |
| drive force mean | 5.907±1.150 N | 5.804±1.071 N |
| transient force mean | 13.351±0.711 N | 12.958±0.913 N |
| tracking linear velocity reward | **1.220±0.005** | 1.211±0.003 |
| tracking yaw reward | **0.603±0.006** | 0.599±0.010 |
| base-height reward | -0.0124±0.0028 | -0.0126±0.0026 |
| episode reward | 9.222±0.740 | 9.263±0.770 |

没有观察到correction 0.10的明显locomotion退化。

## 4. Phase A第二层：Impact-gain粗筛

固定 `correction=0.10 log(3)`，使用64 env、30 iterations、seed 1。Gain=2.0复用前述
run；其余从baseline fresh启动。

| Exp | Correction | Gain | Precision | Recall | F1 | Comp impact | Comp no-impact | R_comp | dF peak | F peak | Base Acc | Comp p95 | Saturation | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| G10 | 0.10 | 1.0 | 0.354 | 0.722 | 0.473 | 0.832 | 0.575 | 1.447 | 5184.3 | 64.993 | 4.811 | 2.861 | 0 | 淘汰：响应与quiet改善不足 |
| G15 | 0.10 | 1.5 | 0.350 | 0.694 | 0.464 | 1.052 | 0.709 | **1.484** | 5138.8 | 64.926 | 4.819 | 3.480 | 0 | 次选：选择性高但quiet不优 |
| G20 | 0.10 | **2.0** | 0.351 | 0.719 | 0.470 | 1.138 | 0.794 | 1.433 | **5132.1** | **64.671** | 4.779 | 3.703 | 0 | **选择：三项quiet整体最好** |
| G25 | 0.10 | 2.5 | 0.346 | 0.701 | 0.462 | 1.286 | 0.904 | 1.422 | 5167.1 | 64.773 | **4.772** | 4.346 | 0 | 淘汰：no-impact/p95/尾部增加 |

Gain=2.5的最后5轮 `compression_over_10mm_ratio=0.00155`，显著高于gain=2.0的
0.00024；gain=1.0/1.5没有降低三项quiet指标。因此保留原基准 `impact_gain=2.0`，
没有继续做二维搜索。

## 5. Phase B：256-env / 100-iteration确认

为检查active mapping是否影响长期数据分布，运行两个fresh seed-1对照：zero correction
和0.10 correction，均使用gain 2.0。表中为最后20轮均值。

| Exp | Correction | Gain | Precision | Recall | F1 | GT ratio | Pred ratio | Comp impact | Comp no-impact | R_comp | dF peak | F peak | Base Acc | Saturation | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| B00 | 0.00 | 2.0 | 0.236 | 0.403 | **0.297** | 0.239 | 0.409 | 1.132 | 0.760 | 1.490 | **4916.9** | **64.125** | 4.899 | 0.000038 | 淘汰：长期柔顺尾部过大 |
| B10 | **0.10** | **2.0** | 0.232 | 0.363 | 0.282 | 0.241 | 0.378 | 0.842 | **0.556** | **1.514** | 4955.3 | 64.132 | **4.866** | **0.000011** | 保留，但不进入大规模长训 |

安全与locomotion对比：

| Metric | B00 zero | B10 corr 0.10 |
|---|---:|---:|
| compression p95 | 4.010 mm | **3.021 mm** |
| compression max | 18.657 mm | **16.875 mm** |
| compression >10mm ratio | 0.004877 | **0.002189** |
| alpha mean | 0.0695 | 0.0712 |
| effective alpha mean | 0.2405 | 0.2344 |
| linear tracking reward | 1.270 | **1.314** |
| yaw tracking reward | 0.631 | **0.652** |
| base-height reward | -0.0677 | **-0.0426** |
| episode reward | 24.136 | **25.483** |

0.10不是F1最优点，但相比zero明显减少no-impact compression和长尾，同时提高
`R_comp`、locomotion tracking和episode reward。因此F1最优参数与控制最优参数不一致。

## 6. Classifier长期稳定性

B10在100 iterations中的20轮窗口趋势：

| Iterations | Precision | Recall | F1 | Raw logit separation | Raw probability separation | Comp impact | Comp no-impact |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–19 | 0.341 | 0.643 | 0.439 | 0.719 | 0.128 | 1.012 | 0.702 |
| 20–39 | 0.287 | 0.514 | 0.368 | 0.692 | 0.100 | 1.015 | 0.667 |
| 40–59 | 0.227 | 0.362 | 0.279 | 0.594 | 0.047 | 0.847 | 0.548 |
| 60–79 | 0.235 | 0.373 | 0.288 | 0.532 | 0.032 | 0.754 | 0.502 |
| 80–99 | 0.232 | 0.363 | 0.282 | 0.520 | 0.032 | 0.842 | 0.556 |

同一末20轮数据做offline zero correction，F1也只有0.298；B00 active-zero run末20轮
F1为0.297。因此退化不是0.10 correction单独造成，也不能再靠减小correction修复。

256-env warmup impact loss从约1.07降到0.72，force MAE末期约32 N，说明增加env确实
改善了warmup监督统计和force estimator。但进入持续on-policy训练后，impact separation
仍随时间下降。最可能的下一瓶颈是变化中的contact/compliance/terrain/command分布下的
estimator稳定性或current-event label可分性，而不再是impact_gain。

在进一步扩大规模前，应增加固定验证分布诊断，区分：

1. on-policy任务变难导致的分布转移；
2. ContactEstimator连续更新造成的遗忘；
3. 5000 N/s边界附近的label噪声；
4. proprioceptive current-event输入本身的Bayes可分性上限。

这些检查不需要把GT加入部署输入，也不要求修改HIMLoco架构。

## 7. Physical quiet与locomotion判断

短期64-env seed-1中，corr 0.10相对zero同时降低force/loading/base acceleration；但
三seed的seed波动较大。256-env长期对照中，corr 0.10的base acceleration更低，而zero
的force/loading略低。当前没有独立、固定命令/地形的baseline评估，因此不能宣称稳定
静音提升。

可以确认的是：

- 0.10相对zero显著降低no-impact compression、p95和>10mm尾部；
- compression-after-impact始终高于compression-after-noimpact；
- saturation仍接近0；
- locomotion tracking未出现明显退化，B10反而优于B00；
- quiet三项尚未形成跨规模、跨seed一致下降。

因此 `Quiet improvement = INCONCLUSIVE`，需要固定评估集上的baseline与候选直接比较。

## 8. 候选淘汰与最终选择

- Full correction `1.0 log(3)`：已有诊断显示recall/F1近乎消失，淘汰。
- Zero correction：短期F1最高，但持续激活、loading偏高，长期柔顺尾部和接近20mm上限
  的风险更大，淘汰为主配置。
- `0.20–0.25 log(3)`：选择性略高，但recall损失没有换来稳定quiet优势，淘汰。
- `0.10 log(3)`：保留足够recall，减少no-impact柔顺与长尾，locomotion稳定，选为当前
  classifier-to-control候选。
- Gain 1.0/1.5：quiet不优；gain 2.5：柔顺尾部过大；保留gain 2.0。

推荐的当前Stage-1配置：

```text
impact BCE pos_weight = 3.0
impact_logit_correction_scale = 0.10
p_control = sigmoid(raw_logit - 0.10 * log(3))
impact threshold = 5000 N/s
impact_gain = 2.0
compliance_init_std = 0.15
compliance_activation_gain = 6.0
max_compression = 0.020 m
其他M/D/K、reward和PPO参数保持不变
```

这是下一轮诊断配置，不是4096-env最终生产参数。

## 9. 是否扩大训练

- **继续256-env定向实验：**只有在加入固定验证分布/漂移诊断后才建议。
- **1024-env / 500 iterations：**当前不建议直接开始；更多数据未阻止100轮内的
  separation退化。
- **4096-env正式训练：**不建议。classifier + compliance + quiet三层链路尚未在中规模
  下同时稳定。

下一步最有价值的实验不是继续correction或gain搜索，而是用固定baseline/contact验证集
持续评估ContactEstimator，定位100轮内separation下降来自分布转移还是遗忘。确认后再用
`corr=0.10, gain=2.0` 做256-env多seed复测。

## 10. 最终研究判断

1. **Impact classifier是否足够支持learned admittance：**足够支持短期探索性选择柔顺，
   但长期稳定性不足，尚不支持4096-env正式训练。
2. **最佳 correction 是否为0：**不是。Zero的F1最高，但控制安全性和no-impact柔顺较差。
3. **是否存在小正 correction 优于zero：**存在。0.10在中规模显著降低柔顺尾部并保持
   locomotion，尽管quiet指标仍为混合结果。
4. **F1最优与静音/控制最优是否一致：**不一致；F1偏向zero，控制约束选择0.10。
5. **当前主要瓶颈：**短期mapping已基本定位；主要瓶颈转为ContactEstimator在长期
   on-policy分布上的separation退化，不是impact_gain。
6. **是否值得扩大num_envs：**不值得直接扩大到1024/4096；先解决中规模稳定性诊断。
7. **是否开始500 iterations：**当前不建议。100 iterations已显示明确退化趋势。

```text
Experiment campaign: PASS
Best classifier-to-control mapping: sigmoid(logit - 0.10 * log(3))
Best impact_gain: 2.0
Classifier status: short-term learnable, long-term on-policy stability insufficient
Compliance selectivity: PASS, R_comp about 1.43 (64 env) to 1.51 (256 env)
Quiet improvement: INCONCLUSIVE
Locomotion degradation: no clear degradation; selected mapping is safer than zero
Recommend 256/1024-env training: NO for blind scale-up; targeted 256 only after drift diagnostics
Recommend 4096-env long training: NO
Recommended final Stage-1 configuration: correction_scale=0.10, impact_gain=2.0
```
