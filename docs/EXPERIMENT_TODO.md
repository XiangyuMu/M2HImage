# MCIC Experiment TODO And Ablation Ledger

状态约定：`[ ]` planned，`[~]` running，`[x]` completed，`[!]` blocked or invalid。

## Core Validation

| Status | ID | Experiment | Acceptance Criterion | Record |
|---|---|---|---|---|
| [ ] | Exp0 | 数据接入、审计、200 张 mask/crop 人工检查 | 配对有效且 masks 可用于监督 | `docs/experiments/exp0.md` |
| [ ] | Exp1 | Paired baseline，512x512，5K steps | 人物真人化可辨认，服装基本保留 | `docs/experiments/exp1.md` |
| [ ] | Exp2 | Exp1 + CF garment，512x512 | CF cloth 指标较 Exp1 改善或视觉漂移下降 | `docs/experiments/exp2.md` |
| [ ] | Exp3 | Exp2 + identity/triplet，512x512 | Delta ID 较 Exp2 提升且服装未显著退化 | `docs/experiments/exp3.md` |
| [ ] | Exp3-R | Exp3，768x512 分辨率复验 | 记录质量、显存与耗时变化 | `docs/experiments/exp3_r.md` |

## Engineering Prerequisites

| Status | Item | Completion Evidence |
|---|---|---|
| [x] | 配置化分辨率与 64 倍数验证 | `mcic/utils/config.py`、tests |
| [x] | 同名配对扫描和 metadata 缓存接口 | `mcic/data/`、`mcic/preprocess/` |
| [x] | Paired/CF 基础训练路径 | `mcic/training/train.py` |
| [x] | 推理与基础 CF 指标输出 | `mcic/inference/`、`mcic/evaluation/` |
| [ ] | 获取真实 paired 数据目录 | 配置更新与审计报告 |
| [ ] | 生成/导入 SCHP-ATR 等真实 parsing maps | `parsing_backend: label_maps` 可运行 |
| [ ] | 在正式数据上确认 Facenet 人脸成功率 | audit/人工抽查记录 |
| [ ] | 添加独立 ArcFace/InsightFace 评测 embedding | 避免训练指标偏差 |

## Ablations After Exp3

每项只与固定 Exp3 基线比较；一次不混合两个方法变量。

| Status | ID | Variable | Comparison / Decision |
|---|---|---|---|
| [ ] | A1 | Random negative -> semi-hard negative | Delta ID 与脸质量 |
| [ ] | A2 | 添加 multi-step fallback | face failure rate 与训练耗时 |
| [ ] | A3 | 添加 boundary edge loss | 领口/发际线伪影与 cloth 指标 |
| [ ] | A4 | 添加 neck/hand weak skin loss | 肤色一致性与服装污染 |
| [ ] | A5 | 添加 condition-sensitivity branch | 对身份条件响应与稳定性 |
| [ ] | A6 | 调整 gated attention token 数或共享策略 | 收敛和身份控制能力 |

## Resolution Studies

分辨率实验不得作为损失/模块消融结论，应独立记录显存、吞吐和视觉收益。

| Status | ID | Resolution | Based On | Notes |
|---|---|---:|---|---|
| [ ] | R1 | 768x512 | Exp3 | 竖幅服装常用比例 |
| [ ] | R2 | 768x768 | Exp3 | 细节质量/显存权衡 |
| [ ] | R3 | 1024x1024 | 最佳低分辨率配置 | 先做短程显存验证 |

## Reporting Rules

- `heuristic` parsing 或 `mock` identity 的运行标记为 smoke test，不填入性能结论。
- 每个实验保存 resolved config、checkpoint、metrics、典型输出与失败图集。
- 数据、解析标签、训练身份编码器或分辨率变化均必须在实验日志中显式记录。
- 若 Exp3 未达到 Delta ID 提升且服装稳定，不开始增强消融，先排查数据、mask 和身份后端。
