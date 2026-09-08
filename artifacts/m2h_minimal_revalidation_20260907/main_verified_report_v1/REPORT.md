# M2H 最小复验客观指标摘要

状态：READY 表示三类 per-image 指标已完整合并、4 个 arm 在同一 128 个 `(mid,jid,seed)` 键上配对、关键指标无 None、未发现基础设施 failed/reference_failed。

解释边界：本摘要是描述性复验，不自动晋级训练，不给出方法成功结论；区间为 source-group/ref-file 双向 pigeonhole bootstrap，10000 draws，seed=20260907，不估计训练 seed 不确定性。

## 样本与资格

- 主结果 pair 数：128；source group：64；reference file：32
- 身体关键点共同检出支持（不用于筛样本）：128/128
- 头部关键点共同检出支持（不用于筛样本）：128/128

## A4-B2 差值与 DID

| 指标 | legacyM A4-B2 | inputOnly A4-B2 | DID |
|---|---:|---:|---:|
| garment_dino | -0.0355151 [-0.0534925, -0.01999] | -0.041818 [-0.0608773, -0.0240121] | -0.00630298 [-0.0243563, 0.0122493] |
| garment_hf_lpips | 0.0499059 [0.0371877, 0.063377] | 0.0599288 [0.0446997, 0.0760496] | 0.0100229 [-0.00454659, 0.0257073] |
| id_penalized | 0.0948105 [0.076373, 0.113705] | 0.0979314 [0.0808485, 0.116104] | 0.00312086 [-0.0116335, 0.017405] |
| body_penalized | 0.00170224 [-0.00676883, 0.00929388] | -0.00247043 [-0.0112949, 0.00419825] | -0.00417267 [-0.0130075, 0.00206186] |
| head_penalized | -7.02841e-05 [-0.00100002, 0.000922703] | -0.000325395 [-0.00184419, 0.000744346] | -0.000255111 [-0.00179707, 0.000602533] |
| body_coverage | -0.000710227 [-0.00815146, 0.0079061] | 0.00362216 [-0.0030303, 0.0126388] | 0.00433239 [-0.0020979, 0.0132869] |
| head_coverage | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |

## 敏感性子集

- 模式：sensitivity_pair_indices；移除 pair：27；保留 pair：101；source group：58；reference file：28
- 这是 strict reference-sensitivity：GlobalSSIM candidate 过滤后的 candidate-stripped 子集，不是 certified clean。
- I 侧重复风险按 reference-sensitivity 报告，不自动等同于答案泄露。
- 同一敏感性 pair 规则已同时作用于 4 个 arm。

## 字段策略

- 只汇总 garment_dino、garment_hf_lpips、id_penalized、body_penalized、head_penalized、body_coverage、head_coverage。
- 所有 carryin 字段按协议视为 adapter self image 衍生，不进入均值、差值或区间。
- 协议 GlobalSSIM 是全局统计 proxy，不等同于 skimage 局部 SSIM。
- joint_detection_support 只是共同检出支持，不是输入资格；真实资格来自 M scores>=.3，本统计不筛样本。
