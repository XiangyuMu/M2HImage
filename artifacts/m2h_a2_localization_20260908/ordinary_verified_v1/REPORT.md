# A2补测与服装定位报告

本批无参数训练。新增A2 128张，复用B2/A4各128张；同一纯M/I输入协议。
区间为探索性来源组×参考文件bootstrap，不代表训练seed不确定性。

## 整体客观指标（128对）

| 模型 | ID↑ | garment DINO↑ | HF-LPIPS↓ | body↓ | head↓ |
|---|---:|---:|---:|---:|---:|
| B2 | 0.508047 | 0.840281 | 0.339890 | 0.038423 | 0.030002 |
| A2 | 0.527868 | 0.854421 | 0.322349 | 0.039070 | 0.030056 |
| A4 | 0.605978 | 0.798463 | 0.399819 | 0.035953 | 0.029677 |

### a2_minus_b2

- id_penalized: +0.019821, 95%CI [+0.007734, +0.032554]
- garment_dino: +0.014140, 95%CI [+0.003117, +0.026273]
- garment_hf_lpips: -0.017541, 95%CI [-0.029450, -0.006673]
- body_penalized: +0.000647, 95%CI [-0.004246, +0.005929]
- head_penalized: +0.000053, 95%CI [-0.000653, +0.000952]

### a4_minus_a2

- id_penalized: +0.078110, 95%CI [+0.062950, +0.093580]
- garment_dino: -0.055959, 95%CI [-0.075030, -0.037828]
- garment_hf_lpips: +0.077470, 95%CI [+0.059948, +0.096216]
- body_penalized: -0.003118, 95%CI [-0.011291, +0.003497]
- head_penalized: -0.000379, 95%CI [-0.001949, +0.000657]

### a4_minus_b2

- id_penalized: +0.097931, 95%CI [+0.081216, +0.115573]
- garment_dino: -0.041818, 95%CI [-0.060840, -0.024133]
- garment_hf_lpips: +0.059929, 95%CI [+0.044724, +0.075734]
- body_penalized: -0.002470, 95%CI [-0.011246, +0.004279]
- head_penalized: -0.000325, 95%CI [-0.001825, +0.000745]

## 解释边界

- A2/A4 differ in identity loss and sampling/bank details; not an isolated identity-loss causal ablation.
- No G2/manual annotations. Hightexture is a source-gradient proxy, not text/print truth.
- Near-duplicate sensitivity101 is not semantic-clean certification; identity overlap alone is not answer leakage.
- Lower cross-reference clothing drift can arise from ignoring identity: inspect identity fidelity alongside it.
- 边界梯度不等于自然度；分区分数也不能证明梯度冲突或定位具体训练原因。
- 本批不据分数自动启动训练；原始数据划分和历史结果不变。
