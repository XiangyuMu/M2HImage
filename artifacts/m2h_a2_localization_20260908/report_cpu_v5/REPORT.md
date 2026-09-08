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

## 分区：A4−A2（同一区域内配对，不比较不同区域绝对分数）

| 区域 | n | DINO差 | HF-LPIPS差 [95%CI] | 原分辨率梯度MAE差 |
|---|---:|---:|---:|---:|
| garment | 128 | -0.055958 | +0.077471 [+0.059948, +0.096219] | +0.006699 |
| interior | 128 | -0.057819 | +0.103460 [+0.081956, +0.125863] | +0.007639 |
| boundary | 128 | -0.033511 | +0.038810 [+0.028072, +0.051048] | +0.001849 |
| hightexture | 128 | -0.028635 | +0.077907 [+0.053032, +0.102460] | +0.001918 |
| lowtexture | 128 | -0.048562 | +0.110184 [+0.085348, +0.134840] | +0.008293 |

## 换参考图时的服装变化（64个人台，每M两参考）

| 区域 | B2 HF-LPIPS↓ | A2↓ | A4↓ |
|---|---:|---:|---:|
| garment | 0.133890 | 0.121157 | 0.142681 |
| interior | 0.128819 | 0.112107 | 0.142969 |
| boundary | 0.050634 | 0.045889 | 0.063319 |
| hightexture | 0.078292 | 0.066657 | 0.104678 |
| lowtexture | 0.103155 | 0.087010 | 0.130115 |

全量和101对敏感性分区、三组完整差值及区间见summary.json。
分区统一使用CPU；完整衣服DINO与旧GPU差≤1e-5、HF-LPIPS差≤1e-3，实际差值见summary.json，不宣称逐值相等。

## 解释边界

- A2/A4 differ in identity loss and sampling/bank details; not an isolated identity-loss causal ablation.
- No G2/manual annotations. Hightexture is a source-gradient proxy, not text/print truth.
- Near-duplicate sensitivity101 is not semantic-clean certification; identity overlap alone is not answer leakage.
- Lower cross-reference clothing drift can arise from ignoring identity: inspect identity fidelity alongside it.
- 边界梯度不等于自然度；分区分数也不能证明梯度冲突或定位具体训练原因。
- 本批不据分数自动启动训练；原始数据划分和历史结果不变。

## 重启恢复与计算消耗

384张图及A2三类指标完整复用；CPU重算分区，8个分片各4线程。13:28:19再次重启，但各分片已完整落盘，随后只合并报告，没有重跑评测或生成图片。
CPU分片最长墙钟503.4秒；CPU恢复新增GPUh为0。原GPU记录加丢失任务保守上界共1.2338 GPUh，并非精确总计。
CPU小样本与全量重算差≤1e-6；DINO与旧GPU差≤1e-5、HF-LPIPS差≤1e-3，具体最大差见summary.json。
