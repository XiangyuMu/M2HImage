# M2H 对比实验：任务重训 smoke 记录（2026-08-23）

这份记录对应服务器 `muxiangyu@10.249.190.76` 上的全新
`mannequin_to_human` 任务适配 smoke。验收目标是每个方法都能完成真实训练、
反事实推理并产出统一格式结果；生成质量不作为阻塞条件。

## 验收结论

| 方法 | 重训步数 | 反事实输出 | canonical 验证 | Metrics-v2 结构 | 质量状态 |
|---|---:|---:|---|---|---|
| RefTon | 100 optimizer steps | 4/4 | pass | `ok` | `ok` |
| ITA-MDT | 100 optimizer steps | 4/4 | pass | `ok` | warning（hair 2/4，identity 0/4） |
| IDM-VTON | 100 optimizer steps | 4/4 | pass | `ok` | `ok` |
| MCLD | 100 optimizer steps | 4/4 | pass | `ok` | warning（identity 2/4） |
| OminiControl | 100 optimizer steps | 4/4 | pass | `ok` | `ok` |

因此当前已经得到 **5 个可运行、经过任务重训的 M2H baseline**。这里的
`smoke` 是链路验收结果，不应解读为论文级最终精度。

## 统一诊断指标

指标来自同一 Metrics-v2 配置；Garment-DINO、Hair-DINO 越高越好，
HF-LPIPS 和 head-5 distance 越低越好。

| 方法 | Garment-DINO ↑ | Garment-HF-LPIPS ↓ | Hair-DINO ↑ | head-5 distance ↓ |
|---|---:|---:|---:|---:|
| RefTon | 0.7869 | 0.3202 | 0.3457 | 0.0198 |
| ITA-MDT | 0.9053 | 0.2940 | 0.3870 | 0.0520 |
| IDM-VTON | 0.9618 | 0.1348 | 0.4465 | 0.0069 |
| MCLD | 0.2922 | 0.5133 | 0.3525 | 0.0500 |
| OminiControl | 0.8418 | 0.4148 | 0.3309 | 0.0224 |

由于每个方法当前只有 4 张反事实图，这些数值只用于确认评测脚本完整运行，
不用于宣称方法优劣。

## M2H 条件接线

统一协议是：`mid` 提供人台、服装、姿势和场景，`jid` 只提供目标身份；
`human(mid)` 只作为 paired training 的监督目标，反事实推理不会读取它。

| 方法 | 空间/源条件（mid） | 身份条件（jid） | 监督目标 |
|---|---|---|---|
| RefTon | `person=mannequin(mid)`，并保留 `agnostic/pose` | `face(jid)` + `identity_card(jid)` | `human(mid)` |
| ITA-MDT | `agnostic(mid)` + `M_replace(mid)` + mannequin 的 dense pose | `identity_person(jid)` + `face(jid)` | `human(mid)` |
| IDM-VTON | mannequin source + replace mask + pose | `identity_card(jid)` | `human(mid)` |
| MCLD | `garment/pose/face_region(mid)` | jid 的 Antelopev2 face embedding | `human(mid)` |
| OminiControl | `agnostic/mannequin(mid)` + pose | `identity_card(jid)` | `human(mid)` |

冻结协议 SHA256：
`7fa287b79ffe32e6bbf8e8d20b0082a74e20c696997abf669f7b80d29a753985`。

反事实 manifest SHA256：
`da54b106e236332580c97da1b8c9a1644e6fc94cab8606b908109dea29b0a455`。

## 远端产物

服务器项目根目录：
`/data/muxiangyu/programs/M2H_Baselines_2026`

- checkpoint：`runs_task_retrain_20260822/<method>/smoke/`
- 原始/统一输出：`outputs_task_retrain_20260822/<method>/smoke/`
- Metrics-v2 报告：`reports_task_retrain_20260822/generated/<method>/smoke/`
- 总结：`reports_task_retrain_20260822/comparison_summary.md`

主要 checkpoint：

```text
refton/runs_task_retrain_20260822/refton/smoke/checkpoint-100
ita_mdt/runs_task_retrain_20260822/ita_mdt/smoke/ema_0.9999_000100.pt
idm_vton/runs_task_retrain_20260822/idm_vton/smoke/checkpoint-100
mcld/runs_task_retrain_20260822/mcld/smoke/mcld_m2h_smoke
ominicontrol/runs_task_retrain_20260822/ominicontrol/smoke/20260823-104323/ckpt/800
```

每个方法的 canonical 目录都包含以下 4 个固定键：

```text
01968__id18372__seed0.png
01968__id18372__seed1.png
01968__id35295__seed0.png
01968__id35295__seed1.png
```

## 复现实验与下一步

当前没有启动 full training，也没有触碰服务器已有的
`dresscode_milestone_eval`、`dresscode_upper_full` tmux 会话。若需要论文级
对比，只需在远端先准备 full 数据，再按方法运行 `train_* <full>`、对应的
`infer_* <full>` 和 `evaluate_metrics_v2.sh <method> full`；建议继续使用独立
的 `M2H_RUN_ROOT/M2H_OUTPUT_ROOT/M2H_REPORT_ROOT` 命名空间。Scone 因官方训练
需要 8×A800，仍保持 deferred，不纳入这 5 个 3090 baseline 的公平排名。
