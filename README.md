# MCIC: Masked Counterfactual Identity Consistency Training

MCIC 是一个用于人台图像真人化的研究原型：输入一张穿着目标服装的人台图像和一张参考身份图像，生成服装与姿势保持、头脸与真人外观符合目标身份的图像。

本仓库实现了首轮端到端实验闭环：

- `Exp0`：配对数据审计、mask 与身份 crop 可视化。
- `Exp1`：SDXL Inpainting + LoRA 的 paired 真人化基线。
- `Exp2`：加入 masked counterfactual branch 和安全服装保持损失。
- `Exp3`：加入目标身份 cosine loss 与 triplet loss。
- 配置化分辨率、推理、counterfactual 评测、实验 TODO 与结果记录模板。

## 方法概览

训练数据是一一对应的人台图 `m_i` 与真人图 `h_i`。模型接受人台图和身份条件：

```text
(mannequin image, reference identity) -> humanized image
```

Paired branch 使用对应身份学习基本转换：

```text
(m_i, identity_i) -> h_i
```

Counterfactual branch 将相同人台搭配另一个身份，但没有完整真值图：

```text
(m_i, identity_j) -> h_(i -> j), j != i
```

因此反事实分支仅施加可成立的局部监督：

- `cloth_safe` 区域应保持原服装视觉内容。
- 生成脸的 embedding 应靠近目标身份 `identity_j`。
- 生成脸应远离原身份 `identity_i`。

首版保留输入图中的安全服装像素，不要求扩散模型重新发明已有的图案和纹理。该选择直接服务于“换人不换衣”的验证目标。

## 能力边界

本实现包含两种预处理/身份后端，请严格区分用途：

| 后端 | 用途 | 可作为论文实验结果 |
|---|---|---|
| `preprocess.parsing_backend: heuristic` + `identity.backend: mock` | 没有真实数据/权重时进行工程冒烟测试 | 否 |
| `preprocess.parsing_backend: label_maps` + `identity.backend: facenet` | 使用真实 human parsing 标签和可微人脸 embedding 训练 | 是，人工质检通过后 |

`label_maps` 接受 SCHP-ATR 或等效解析器预生成的离散标签 PNG。仓库不内置第三方解析模型权重，以便解析器版本、标签映射和许可由实验记录明确管理。

## 仓库结构

```text
configs/                         实验、预处理与评测配置
docs/
  DATASET_SPEC.md                数据、缓存和 mask 约定
  MODEL_DESIGN.md                模型与损失的实现说明
  EXPERIMENT_TODO.md             实验和消融执行清单
  EXPERIMENT_LOG_TEMPLATE.md     每次运行的人工记录模板
mcic/
  data/                          配对扫描与训练 dataset
  preprocess/                    数据审计、解析 mask 接入和身份缓存
  models/                        SDXL/LoRA 条件模块与身份编码器
  losses/                        paired、garment、identity、triplet 损失
  training/                      训练分支与 accelerate 入口
  inference/                     单图身份转换
  evaluation/                    paired 与 CF 评测
scripts/                         常用命令封装
tests/                           不加载大模型的基础测试
```

## 环境安装

推荐使用 Python `3.11` 与 CUDA 12.1 PyTorch wheel。当前机器驱动 `535.183.01` 报告支持 CUDA `12.2`，且有两张 RTX 3090（各 24 GB）；默认训练配置据此选择 `512x512`、每卡 batch size 1 和梯度累积 8。由于正式身份后端 `facenet-pytorch 2.6.0` 的依赖范围，环境固定为 `torch 2.2.2`、`torchvision 0.17.2`、`numpy 1.x` 与 `Pillow 10.2.x`；与此 PyTorch 版本兼容的生成训练栈固定为 `diffusers 0.30.3`、`transformers 4.44.2`、`accelerate 0.33.0` 与 `peft 0.12.0`。

```bash
conda create -n mcic python=3.11 pip -y
conda activate mcic
python -m pip install --upgrade pip setuptools wheel

# CUDA 12.1 wheel 与本机驱动、facenet-pytorch 约束匹配：
pip install torch==2.2.2 torchvision==0.17.2 \
  --index-url https://download.pytorch.org/whl/cu121

pip install -e '.[dev,parsing]'
```

也可用仓库提供的 Conda 清单创建同等版本的环境：

```bash
conda env create -f environment.yml
conda activate mcic
```

不要在该环境中直接升级到 `transformers 5.x` 或 `diffusers 0.38+`：它们需要更新的 PyTorch 接口，而 `facenet-pytorch 2.6.0` 将身份监督路径限制在 PyTorch `<2.3`。如要升级生成训练栈，应同时替换或验证身份编码后端。

验证环境：

```bash
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
pytest
```

Hugging Face 模型下载需要可访问网络及足够磁盘空间：

```bash
export HF_HOME=/data/huggingface_cache
huggingface-cli login       # 权重许可要求登录时执行
```

主要权重由配置引用：

```yaml
model:
  base_model: diffusers/stable-diffusion-xl-1.0-inpainting-0.1
  mannequin_encoder: openai/clip-vit-large-patch14
identity:
  backend: facenet
```

若使用本地权重目录，直接将这些配置值改为本地路径。

## 数据格式

默认输入以相对同名配对，扩展名可不同：

```text
/data/project/dataset/
  mannequin/
    0001.jpg
    subfolder/0002.png
  human/
    0001.jpg
    subfolder/0002.webp
```

`mannequin/subfolder/0002.png` 会与 `human/subfolder/0002.webp` 配成 `sample_id=subfolder/0002`。每对图像应满足：

- 相同服装，姿势和构图近似空间对齐。
- 真人图中面部可见，能够成为身份参考。
- 首版不要求 `person_id`；每张真人图作为一个身份条件。

正式训练需要先从真人图生成解析标签。将每张标签图按 `sample_id` 保存为 PNG，并在 `configs/preprocess.yaml` 中配置：

```yaml
preprocess:
  parsing_backend: label_maps
  parsing_dir: /data/project/parsing_labels
  label_ids:
    face: [1]
    hair: [2]
    cloth: [3, 4, 5, 6]
    person: [1, 2, 3, 4, 5, 6, 7]
identity:
  backend: facenet
```

标签 ID 必须按照实际解析器的 label schema 修改。完整缓存 schema 见 [docs/DATASET_SPEC.md](docs/DATASET_SPEC.md)。

## 分辨率设计

训练、推理与评测都使用同一个公共配置段：

```yaml
image:
  height: 512
  width: 512
  interpolation_image: bicubic
  interpolation_mask: nearest
  center_crop: false
  pad_to_aspect_ratio: true
  pad_value: 255
```

`height` 和 `width` 都必须为 64 的正整数倍。加载器按比例 resize 并 padding 到目标画布，不拉伸人物；离散 mask 用 nearest resize。原始 mask 不随分辨率变化，因而更改训练尺寸无需重新预处理。

推荐阶梯：

| 分辨率 | 用途 | 双 3090 起始建议 |
|---|---|---|
| `512x512` | Exp1-3 首次验证 | 每卡 batch 1，accumulation 8 |
| `768x512` | 竖幅服装对照 | 每卡 batch 1，开启 checkpointing |
| `768x768` | 细节质量验证 | 视显存调高 accumulation |
| `1024x1024` | 后续高质量实验 | 单卡 micro batch 1，并先短程测显存 |

用命令行覆盖即可运行矩形实验：

```bash
./scripts/train_exp03.sh image.height=768 image.width=512 training.batch_size_per_gpu=1
```

不同输出尺寸的指标应单独报告，不与损失消融表混合比较。

## 快速工程冒烟

此流程验证目录、metadata、mask、代码导入和配置覆盖，不产生有效科研结论。

1. 准备少量同名图像，并将 `configs/preprocess.yaml` 保持为：

   ```yaml
   preprocess:
     parsing_backend: heuristic
   identity:
     backend: mock
   ```

2. 运行审计和预处理：

   ```bash
   ./scripts/audit_data.sh data.root=/path/to/smoke_dataset
   ./scripts/preprocess.sh data.root=/path/to/smoke_dataset
   ```

3. 查看输出：

   ```text
   /path/to/smoke_dataset/cache_mcic/audit_report.json
   /path/to/smoke_dataset/cache_mcic/metadata.jsonl
   /path/to/smoke_dataset/cache_mcic/visual_checks/
   ```

4. 运行基础单元测试：

   ```bash
   pytest
   ```

## 正式数据预处理

1. 修改 `configs/preprocess.yaml` 使用真实数据路径、真实解析标签目录与 `facenet`。

2. 数据审计：

   ```bash
   python -m mcic.preprocess.audit --config configs/preprocess.yaml
   ```

   检查：

   - `paired_count` 是否符合预期。
   - `missing_human`、`missing_mannequin` 是否为空。
   - `size_mismatches` 是否可解释；像素损失要求配对基本对齐。
   - `visual_checks/pair_*.jpg` 中服装与姿势是否对应。

3. 缓存 parsing mask、face crops 和 identity embeddings：

   ```bash
   python -m mcic.preprocess.run --config configs/preprocess.yaml
   ```

4. 人工查看至少 200 张 `visual_checks/mask_*.jpg`，确认脸/头发编辑区与安全服装区合理，再在 `docs/EXPERIMENT_TODO.md` 勾选 Exp0。

## 训练

所有训练结果写入 `outputs/<experiment_id>/`：

```text
resolved_config.yaml
checkpoints/step-*/
checkpoints/final/
logs/train.jsonl
```

### Exp1: Paired Baseline

正式训练前请确认预处理缓存由 `identity.backend: facenet` 生成；如只做小数据工程冒烟，可显式覆盖 `identity.backend=mock`。

```bash
./scripts/train_exp01.sh data.root=/data/project/dataset
```

目标：验证 SDXL-inpainting、mannequin tokens 与 identity tokens 可以产生基本合理的真人化结果。运行 5K steps 后评估是否继续。

### Exp2: Counterfactual Cloth Consistency

将 Exp1 最佳 checkpoint 写入 `experiment.init_checkpoint` 或覆盖：

```bash
./scripts/train_exp02.sh \
  data.root=/data/project/dataset \
  experiment.init_checkpoint=outputs/exp01_paired/checkpoints/final
```

目标：加入 `cloth_safe` L1 后，给同一人台换身份不应使服装漂移。

### Exp3: Counterfactual Identity Consistency

Exp3 的预处理和训练配置必须都使用一致的 `identity.backend: facenet`。

```bash
./scripts/train_exp03.sh \
  data.root=/data/project/dataset \
  experiment.init_checkpoint=outputs/exp02_cf_cloth/checkpoints/final
```

目标：相比 Exp2，`Delta ID = sim(output,target)-sim(output,source)` 上升，同时安全服装误差不显著上升。

Exp3 在训练中使用生成脸检测质量门控；检测失败的预测仍参与服装损失，但不会被强制施加身份/triplet loss。日志中的 `identity_face_pass_rate` 用于识别脸部退化问题。

### 断点与配置记录

训练会自动保存配置快照与模型条件器/LoRA checkpoint。每次正式训练均应复制 [docs/EXPERIMENT_LOG_TEMPLATE.md](docs/EXPERIMENT_LOG_TEMPLATE.md) 记录数据版本、checkpoint、硬件、分辨率与视觉结论。

## 推理

单图推理支持自定义 mask。为了正式结果可解释，建议提供由解析器或人工修订得到的 mask；未提供时命令使用粗略自动 mask，仅用于预览。

```bash
python -m mcic.inference.generate \
  --config configs/eval.yaml \
  --checkpoint outputs/exp03_cf_identity/checkpoints/final \
  --mannequin /data/project/dataset/mannequin/0001.jpg \
  --reference /data/project/dataset/human/0042.jpg \
  --mask /data/project/dataset/cache_mcic/masks/0001_paired_mask.png \
  --output outputs/demo/0001_to_0042.png
```

输出包括生成图、四联对照 panel、resolved config 和 metadata JSON。

## 评测

评测使用预处理 metadata 的固定 `test` split：

```bash
./scripts/evaluate.sh \
  --checkpoint outputs/exp03_cf_identity/checkpoints/final \
  data.root=/data/project/dataset
```

输出位置：

```text
outputs/evaluation/
  metrics.json
  samples/paired_*.png
  samples/cf_*.png
```

当前自动汇总指标：

| 指标 | 含义 |
|---|---|
| `paired.cloth_l1_mean` | 对应身份生成时安全服装区域误差 |
| `counterfactual.cloth_l1_mean` | 换身份时安全服装区域误差 |
| `ssim_mean` / `cloth_ssim_mean` | 全图与安全服装区域结构相似度 |
| `lpips_mean` / `cloth_lpips_mean` | 全图与安全服装区域感知距离 |
| `face_detect_rate` | 生成脸可被身份评测后端检测的比例 |
| `sim_target_mean` | 输出脸与目标身份相似度 |
| `sim_source_mean` | 输出脸与原身份相似度 |
| `delta_id_mean` | 身份切换强度，期望大于零并相对 Exp2 增长 |

姿势/keypoint 指标留待真实数据与解析器确认后加入，避免先固定不合适的姿势检测口径。

## 实验与消融管理

执行顺序和待办见 [docs/EXPERIMENT_TODO.md](docs/EXPERIMENT_TODO.md)。规则如下：

- Exp0 人工 mask 质检未完成前，不启动长程训练。
- Exp3 是消融固定比较基线。
- 消融一次只改变一个变量，例如 negative sampling 或 edge loss。
- 分辨率变化标记为 resolution study，不解释为损失函数贡献。
- `heuristic` 或 `mock` 后端的输出只能标记为 pipeline smoke test。

## 常见问题

**显存不足**

首先降低 `training.batch_size_per_gpu`、使用 `512x512`、确保 `gradient_checkpointing: true`，再提高 gradient accumulation。更高分辨率必须单独记录显存与运行时间。

**mask 错位或领口伪影**

检查标签 ID 和配对对齐；扩大 `boundary_dilate_radius` 或人工修订失败 mask。错误解析会直接污染 garment loss。

**身份损失不下降**

确认预处理与训练都使用 `identity.backend: facenet`，目标身份不是原样本，人脸编辑区覆盖完整 face/hair，并人工抽查生成脸可检测性。

**服装漂移**

首先检查 `cloth_safe_mask` 是否包含身份边界或背景，随后比较 Exp1/Exp2 的 counterfactual garment 指标，再考虑提高 `lambda_cloth`。

**权重下载失败**

设置可写的 `HF_HOME` 并检查模型访问许可；亦可在 YAML 中将模型标识换为已经下载的绝对路径。

## 复现清单

正式结果至少保存：

- 代码 commit 或源码快照。
- `resolved_config.yaml` 与命令行覆盖项。
- 数据目录版本、解析器及 label ID 映射、split seed。
- SDXL/CLIP/人脸模型版本与权重来源。
- GPU 型号、数量、CUDA/PyTorch 版本。
- 实际分辨率、batch size、累积步数、总 step 和运行时间。
- `metrics.json`、视觉样本和失败样本分析。

更完整的方法实现说明见 [docs/MODEL_DESIGN.md](docs/MODEL_DESIGN.md)。
