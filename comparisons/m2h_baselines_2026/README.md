# M2H Baselines 2026

这是人台模特真人化（Mannequin-to-Human, M2H）的可复现对比实验工程。默认部署到 `muxiangyu@10.249.190.76:/home/muxiangyu/programs/M2H_Baselines_2026`，数据只读使用 `/home/muxiangyu/datasets/M2HImage/M2H_Final_v2`；环境、模型下载缓存和临时文件全部落在 `/data/muxiangyu`，避免占用只剩少量空间的系统盘。

## 方法范围

| 方法 | 论文 | 年份/会场 | 当前状态 |
|---|---|---|---|
| RefTon | Reference-based Virtual Try-On | CVPR 2026 | active |
| ITA-MDT | Improving VTON via Masked Diffusion Transformer | CVPR 2025 | active |
| IDM-VTON | Improving Diffusion Models for Authentic Virtual Try-on | ECCV 2024 | active |
| MCLD | Multi-focal Conditioned Latent Diffusion | CVPR 2025 | active |
| OminiControl | Minimal and Universal Control for Diffusion Transformer | ICCV 2025 Highlight | active |
| Scone | Scalable Contextualized Image Generation | CVPR 2026 Highlight | deferred |

Scone 保留为第二个 2026 方法，但官方训练是四阶段、8×A800；当前 2×RTX 3090 不能做公平复现，因此不伪造一个缩水训练结果。所有上游仓库及固定 commit 见 `manifests/repos.json`。

## M2H 条件接线

统一定义：`mid` 提供人台、服装、身体姿势和背景，`jid` 只提供目标身份。反事实推理绝不把 `human(mid)` target 当条件；该图只在 paired training 中作为监督。

| 方法 | 空间/source 条件（mid） | 身份条件（jid） | paired target | 反事实推理是否读取 target |
|---|---|---|---|---|
| RefTon | `person=mannequin`，另有 `agnostic`、`pose` | `cloth=face`，`image_ref=identity_card` | `human(mid)` | 否；pipeline 只打包显式 condition keys |
| ITA-MDT | `agnostic(mid)`、`M_replace(mid)`、`pose_dense=mannequin DWPose(mid)` | `cloth=identity_person(id_strong ∪ id_weak)`、`cloth_sr=face` | `human(mid)` | 否；生成器的 `cond_keys` 不含 image |
| IDM-VTON | `source=mannequin`、replace mask、pose | `cloth=identity_card` | `human(mid)` | 否；inpaint canvas 显式使用 `source` |
| MCLD | `texture=garment(mid)`、`pose(mid)`、`face_region(mid)` | Antelopev2 embedding of `face(jid)` | `human(mid)` | 否；inference function 不读取 `img` |
| OminiControl | `agnostic/mannequin(mid)`、`pose(mid)` | `identity_card(jid)` | `human(mid)` | 否；直接读取 manifest 的三个条件字段 |

ITA-MDT 的磁盘 mask 固定为 `M_replace`（255 表示需要生成），官方 loader 内部反转一次得到 keep mask。`preflight.sh` 会用真实样本数值验证 `loader_keep ≈ 1 - M_replace`。训练 loss 另行保留空间增广后的单通道二值 `replace_mask`，下采样到 latent 后按选区面积归一化 masked MSE；禁止把 VAE 编码后的 keep-mask latent 当作空间 loss mask。

## 固定实验协议

- 训练 seed：42；每种方法只训练一次。
- 第一阶段分辨率：512×512（原 768×1024 图像等比缩到 384×512，再左右反射 padding 64 px）。
- full counterfactual：200 pairs × 2 seeds = 400 张。
- 冻结协议 SHA256：`7fa287b79ffe32e6bbf8e8d20b0082a74e20c696997abf669f7b80d29a753985`。
- smoke：32 train、8 val、2 counterfactual pairs × 2 seeds。
- 预算：RefTon smoke 100 steps/full 64 epochs；ITA-MDT smoke 100/full 200k；其余 smoke 100/full 60k。

## 部署与准备

在当前开发机执行：

```bash
comparisons/m2h_baselines_2026/scripts/deploy.sh
comparisons/m2h_baselines_2026/scripts/deploy_metrics_v2.sh
```

在目标服务器执行：

```bash
cd /home/muxiangyu/programs/M2H_Baselines_2026
scripts/setup_envs.sh all
scripts/setup_models.sh all
scripts/prepare_data.sh smoke low
scripts/prepare_mcld_faces.sh all
scripts/preflight.sh all paired smoke
scripts/preflight.sh all counterfactual smoke
```

`setup_models.sh` 先复用 `/data` 中已有的 RefTon、IDM-VTON 和 FLUX 权重；只有缺失内容才下载到 `/data/muxiangyu/cache/m2h_baselines_2026/models`。ITA-MDT 从论文作者发布且 SHA256 固定的 2M-step EMA checkpoint 初始化 M2H 微调，但将 optimizer、EMA 和训练 step 重新置零；不会把 100-step smoke 从随机网络开始训练。项目目录中的 `models/` 只有符号链接。

`deploy_metrics_v2.sh` 只选择性同步主工程中的 Metrics-v2 入口、配置和指标模块，并部署 FASHN parser、固定哈希的 AdaFace IR-101 与 UniFace 3.7.1。它不会删除或覆盖主工程中的其它旧文件。对比实验使用 `configs/metrics_v2.yaml` 将 DWPose、DINOv2、InsightFace 和评测 Python 接到目标服务器已有的 `/data` 资源。

## Runnable 训练门禁

每个方法先做一个真实 forward/backward optimizer step，再做 100-step smoke。以下以 RefTon 为例，其余方法名替换为 `ita_mdt`、`idm_vton`、`mcld`、`ominicontrol`：

```bash
M2H_STEPS=1 scripts/train_refton_smoke.sh
scripts/train_refton_smoke.sh
scripts/infer_refton_smoke.sh
scripts/evaluate_metrics_v2.sh refton smoke
```

smoke 的验收只判断链路是否可运行：data-only probe、真实 optimizer
step、任务适配 checkpoint、反事实推理、canonical 输出完整性和
Metrics-v2 无异常结束。视觉质量和指标数值照常记录，但不阻塞后续方法或
full 训练。以上 runnable 检查都通过后，才启动对应 full：

IDM-VTON、MCLD 和 OminiControl 的新 checkpoint 会写入
`m2h_checkpoint.json`。推理前强制核验 `task=mannequin_to_human`、冻结
协议哈希、至少一个 optimizer step、条件接线和全部权重文件；缺少这些
provenance 的官方原始权重或早期历史 checkpoint 会被拒绝。

```bash
scripts/prepare_data.sh full low
scripts/prepare_mcld_faces.sh all
scripts/train_refton_full.sh
```

训练可放入独立 tmux session，例如：

```bash
tmux new-session -d -s m2h_refton_smoke 'cd /home/muxiangyu/programs/M2H_Baselines_2026 && scripts/train_refton_smoke.sh'
scripts/monitor.sh refton smoke once
```

## 推理与评测产物

每种上游方法先写到：

```text
outputs/<method>/<profile>/raw/
```

收集器验证图像可读性、尺寸、记录数和唯一性，再创建 Metrics-v2 标准命名：

```text
outputs/<method>/<profile>/canonical/<mid>__id<jid>__seed<seed>.png
```

同时保存 checkpoint 路径、每张图 SHA256、推理 manifest、子协议和 summary。`evaluate_metrics_v2.sh` 复用主项目的 Metrics-v2（garment、hair、pose、held-out identity、distribution、panels、report），并把报告写入 `reports/generated/<method>/<profile>/`。

## 常用覆盖

```bash
M2H_STEPS=1                    # 单优化步
M2H_INFER_LIMIT=1              # 单图推理与同步收集
M2H_CHECKPOINT=/absolute/path  # 显式 checkpoint
M2H_INFER_GPU=0                # 单卡推理
M2H_INFER_MEMORY_MODE=group_offload  # 24 GiB 卡上的 FLUX 推理（默认）
M2H_INFER_GROUP_BLOCKS=2       # 每次 onload 的 FLUX blocks
M2H_DATALOADER_WORKERS=2
M2H_NUM_PROCESSES=2
M2H_MODEL_SAVE_EPOCH_INTERVAL=10     # MCLD component 周期；最终步总会保存
```

2026-08-22 发起的全新任务适配 smoke 使用独立命名空间，避免自动选择早期
checkpoint。训练、推理和评测前均可执行：

```bash
source configs/task_retrain_20260822.env
```

脚本均为可重复运行设计；不会自动删除远端文件，也不会在 smoke 未通过时自动启动 full training。

## Full native-protocol 队列（2026-08-23）

全量实验使用隔离的 `prepared_task_full_native_20260823` 和
`layouts_task_full_native_20260823`，不会覆盖 smoke staging。先从全量数据
生成 native 768x1024 条件，再生成模型实际读取的 512x512 letterbox staging；
模型输出去掉左右各 64px padding 后恢复为 768x1024 canonical，再交给
Metrics-v2。

```bash
source configs/task_full_native_20260823.env
scripts/prepare_full_native.sh
scripts/prepare_mcld_faces.sh all
scripts/preflight.sh all paired full
scripts/preflight.sh all counterfactual full
scripts/run_full_native.sh --skip-prepare
```

`run_full_native.sh` 按 RefTon、ITA-MDT、IDM-VTON、MCLD、OminiControl
顺序执行 full 训练、400 张反事实推理、native 转换和指标计算。每个 phase
写入 `runs_task_full_native_20260823/orchestrator/status.tsv`；检测到 CUDA/系统
OOM 时将当前方法标为 `oom`、保留日志和已有 checkpoint，并继续下一方法。
