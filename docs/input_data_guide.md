# M2HImage 输入数据目录与语义说明

本文档记录当前 M2HImage Phase 1、B2-cont、A2 和 A4 使用的数据输入目录、图像特点、条件语义及已知风险。
当前数据根目录为：

```text
/data/muxiangyu/datasets/M2HImage/M2H_Final_v2
```

当前有效分辨率为 `768 x 1024`（宽 x 高）。split 共包含 40,014 个唯一 ID：

| split | ID 数量 | 说明 |
|---|---:|---|
| `splits/train.txt` | 36,034 | paired warmup、A2、A4 的训练来源 |
| `splits/val.txt` | 1,996 | watcher 固定样本和 paired 验证 |
| `splits/test.txt` | 1,984 | B2/A2/A4 冻结反事实评测来源 |

A2、A4 和 B2-cont 为公平性共同排除了训练 ID `47160`，因此这些 run 的实际训练样本数为 36,033。

## 1. ID 语义

同一个 ID 在配对数据中连接一组空间对齐的数据：

```text
images/mannequin/{id}                 人台输入
images/human/{id}                     配对真人目标
dwpose/without_head/mannequin/{id}    人台姿势
clothes_bySAM/masks/human/{id}        服装区域
derived/face_crops/human/{id}         真实脸裁剪
derived/head_pose_6drepnet/human/{id} 头朝向
```

反事实评测使用两个不同 ID：

- `mid`：mannequin ID，决定服装、身体姿势和头朝向。
- `jid`：identity ID，决定 PuLID 身份和头发/外观。

因此反事实生成的目标不是复原某一张现有真人图，而是组合：

```text
生成结果 = mid 的服装与姿势 + jid 的身份与头发外观
```

## 2. 原始图像输入目录

### 2.1 `images/mannequin/`

示例：

```text
images/mannequin/21603.png
```

| 属性 | 内容 |
|---|---|
| 数量 | 40,014 张，覆盖全部 split ID |
| 分辨率 | 主导且当前使用的分辨率为 `768 x 1024` |
| 图像特点 | 竖幅 RGB 图；人台穿着目标服装；保留完整服装、身体轮廓与姿势；脸部不是真实身份 |
| 模型含义 | 服装和人台侧空间条件的来源 |
| 实际用法 | 与服装 mask 相乘并裁剪后，由 CLIP-L 提取 `garment_grid` |

人台原图不会直接作为 FLUX 图像条件送入 transformer。模型使用的是由它提取出的服装 patch tokens，姿势则由独立 DWPose 控制图提供。

### 2.2 `images/human/`

示例：

```text
images/human/21603.jpg
images/human/21053.jpg
```

| 属性 | 内容 |
|---|---|
| 数量 | 40,014 张，覆盖全部 split ID |
| 分辨率 | `768 x 1024` 竖幅 RGB 真人图 |
| 图像特点 | 真人穿着与同 ID 人台一致的服装，通常具有对应的身体姿势和空间布局 |
| paired 语义 | `images/human/{mid}` 是训练监督目标 `h_i`，VAE 编码后成为 `target_latents` |
| identity 语义 | `images/human/{jid}` 用于检测并裁出 1.8 倍外扩头部，提取发型/外观 `appearance` |

必须区分以下两种用法：

- `human/{mid}`：配对真人真值，用于训练 `L_pair` 和观察服装/姿势是否复原。
- `human/{jid}`：身份参考原图，用于离线提取身份侧 appearance；它不是反事实结果的像素级真值。

在 B2/A2/A4 反事实推理时，完整的 `human/{mid}` 和 `human/{jid}` 都不会直接送入生成模型，推理读取的是它们的离线缓存特征。

### 2.3 `dwpose/without_head/mannequin/`

示例：

```text
dwpose/without_head/mannequin/21603.png
```

| 属性 | 内容 |
|---|---|
| 数量 | 40,014 张，覆盖全部 split ID |
| 分辨率 | `768 x 1024` |
| 图像特点 | RGB 骨架/关键点控制图；保留身体姿势；头部关键点被去除 |
| 模型含义 | 人台身体姿势条件，不承载身份和脸部结构 |
| 实际用法 | VAE 编码为 `(3072, 64)` 的 `pose_latents`，输入 InstantX Union ControlNet，`control_mode=4` |

移除头部姿势是为了避免 DWPose 头部结构与独立的 `head_pose`、PuLID 身份条件冲突。

### 2.4 `clothes_bySAM/masks/human/`

示例：

```text
clothes_bySAM/masks/human/21603.png
```

| 属性 | 内容 |
|---|---|
| 数量 | 40,014 张，覆盖全部 split ID |
| 分辨率 | `768 x 1024` |
| 图像特点 | 单通道/灰度服装二值 mask；白色表示服装，黑色表示非服装 |
| 模型含义 | 定义服装 crop 的有效区域 |
| 实际用法 | mask 应用到同 ID 的 `images/mannequin`，mask 外置白，再按外接框裁剪 |

目录名包含 `human`，但当前实现把该 mask 投影到配对人台图上。这依赖同 ID 的 human/mannequin 空间对齐。样本 `21603` 中该 mask 与 `region_masks` 的 cloth 区 IoU 为 `0.988`，接线正确；缓存构建仍缺少覆盖全部样本的自动配准质量检查。

### 2.5 `derived/face_crops/human/`

示例：

```text
derived/face_crops/human/21053.png
```

| 属性 | 内容 |
|---|---|
| 数量 | 40,014 张，覆盖全部 split ID |
| 分辨率 | 随人脸 bbox 变化，不是固定 768 x 1024 |
| 图像特点 | RetinaFace/InsightFace 得到的紧脸裁剪，主要包含脸部，尽量排除衣服和背景 |
| 模型含义 | PuLID 的身份信息来源 |
| 实际用法 | 由 PuLID 官方 EVA-CLIP + ArcFace 混合编码流程得到 `(32, 2048)` 的 `pulid_id_embed` |

PuLID 身份条件与 held-out AdaFace 评测严格分离。AdaFace 只在指标 runner 中使用，不参与训练、采样或条件构建。

### 2.6 `phase1/cache_768x1024/debug_head_crops/`

| 属性 | 内容 |
|---|---|
| 数量 | 20 张抽样 debug 图 |
| 图像特点 | 从原始 `images/human` 检测人脸后，按 1.8 倍外扩并向上偏移得到的头部 crop |
| 覆盖内容 | 脸、头发、部分颈部，可能包含肩部和领口 |
| 用途 | 仅供人工检查 appearance crop，不是训练时逐步读取的输入目录 |

这些 crop 经 CLIP-L pooled feature 编码为 1024 维 `appearance`。它们补充发型和头部外观，但也可能把身份来源图的领口或上衣信息带入条件。

### 2.7 `derived/region_masks_z/debug/`

| 属性 | 内容 |
|---|---|
| 数量 | 20 张 mask 叠加 debug 图 |
| 图像特点 | 将 48 x 64 token mask 上采样回 768 x 1024 后叠加到原图 |
| 用途 | 人工检查 `cloth_safe/body_bg/face` 与 packed latent token 是否对齐 |

该目录只用于检查，不参与训练和推理读取。

## 3. 非图像派生输入

### 3.1 `derived/head_pose_6drepnet/human/`

文件格式：

```text
derived/head_pose_6drepnet/human/{id}.json
```

每个 JSON 记录真人图的 `yaw/pitch/roll`、检测状态和置信度。角度被编码为：

```text
[sin(yaw), cos(yaw), sin(pitch), cos(pitch), sin(roll), cos(roll), valid]
```

得到 7 维 `head_pose` token。当前 40,014 个 JSON 均为 `status=ok`；训练时另有 10% 动态 null-dropout，评测时 dropout 为 0。

### 3.2 `derived/region_masks/`

文件格式：

```text
derived/region_masks/{id}.npz
```

每个 NPZ 的主要键及含义：

| 键 | 形状 | 含义 |
|---|---|---|
| `id_strong` | `(1024, 768)` | 强身份区，主要是脸和头发 |
| `id_weak` | `(1024, 768)` | 弱身份区，例如皮肤/肢体等区域 |
| `cloth` | `(1024, 768)` | 完整服装区域 |
| `cloth_safe` | `(1024, 768)` | 远离身份边界的安全服装区域 |
| `body_bg` | `(1024, 768)` | 身体非身份区与背景区域 |
| `edge` | `(1024, 768)` | 区域边缘/不稳定边界 |

用途包括官方 GarmentSim mask 投影、差分 mask 构建以及服装保护采样。

### 3.3 `derived/region_masks_z/`

文件格式：

```text
derived/region_masks_z/{id}.npz
```

每个 mask 从 768 x 1024 按 16 x 16 像素平均池化到 48 x 64 token 网格，并展平为 3072 个软权重：

| 键 | 形状 | A2/A4 用途 |
|---|---|---|
| `cloth_safe_z` | `(3072,)` | `L_teach` 的服装区域 |
| `body_bg_z` | `(3072,)` | `L_inv` 的身份不变区域 |
| `face_z` | `(3072,)` | `L_hinge` 的身份响应区域 |

当前该目录覆盖 36,034 个 train ID，没有为 val/test 的 3,980 个 ID 预先落盘。训练覆盖完整；需要 eval mask 的推理工具会从 `derived/region_masks` 在内存中做相同投影。

### 3.4 `derived/identity_bank_v2.npz`

该文件包含 train split 的：

```text
ids       样本 ID
embeds    512 维 F_train Glint360K ArcFace embedding
attrs     gender / age / skin_cluster 等属性
```

它只用于 A4 的 semi-hard `(j,k)` 采样、身份距离、hinge 标定和定向身份损失参考，不会作为 PuLID 条件直接送入生成模型。

### 3.5 `eval/cf_subset.json`

冻结的反事实评测协议，包含：

```text
mannequin_id / mid   服装、姿势、头朝向来源
identity_id / jid    身份与 appearance 来源
theta_source         头朝向来源，当前等于 mid
garment_type         top / dress / pants / skirt
seeds                [0, 1]
```

所有 B2-cont、A2、A4、权重插值和保护采样使用同一 pair 与 seed，保证逐图配对比较。

## 4. 实际训练/推理读取的缓存

训练循环不会重复运行 VAE、CLIP、PuLID 或头姿编码器，而是读取：

```text
phase1/cache_768x1024/samples/{id}.npz
phase1/cache_768x1024/text/prompt.npz
```

单样本缓存内容：

| 缓存键 | 形状 | 原始来源 | 含义 |
|---|---:|---|---|
| `target_latents` | `(3072, 64)` | `images/human/{id}` | paired flow 的真实终点 `z0` |
| `pose_latents` | `(3072, 64)` | `dwpose/.../mannequin/{id}` | pose ControlNet 条件 |
| `pulid_id_embed` | `(32, 2048)` | `face_crops/human/{id}` | 预训练 PuLID 身份条件 |
| `appearance` | `(1024,)` | `images/human/{id}` 的外扩头部 crop | 头发和头部外观条件 |
| `garment_grid` | `(64, 1024)` | `mannequin + garment mask` | 8 x 8 CLIP 服装 patch token |
| `head_pose` | `(7,)` | head-pose JSON | 头朝向 token |

固定 prompt 缓存包含：

| 缓存键 | 形状 | 含义 |
|---|---:|---|
| `prompt_embeds` | `(512, 4096)` | T5 文本条件 |
| `pooled_prompt_embeds` | `(768,)` | CLIP pooled 文本条件 |

当前 prompt 为：

```text
a photorealistic human wearing the same garment, same body pose, natural skin and hair
```

## 5. 不同阶段如何组合输入

### 5.1 Paired warmup

对于样本 `i`，全部条件和监督目标都来自相同 ID：

```text
target_latents(i)
pose_latents(i)
garment_grid(i)
head_pose(i)
pulid_id_embed(i)
appearance(i)
```

模型学习从噪声流向真实 `images/human/{i}` latent。

### 5.2 A2/A4 差分训练

paired 分支仍使用样本 `i`。反事实 `j/k` 分支只替换身份侧两路条件：

```text
保持 i：pose_latents / garment_grid / head_pose / ControlNet 输出
替换 j：pulid_id_embed(j) + appearance(j)
替换 k：pulid_id_embed(k) + appearance(k)
```

这保证差分比较只改变身份侧条件。

### 5.3 B2/A2/A4 反事实推理

推理接线位于 `eval_b2.make_cf_batch`：

```text
从 mid 缓存读取：pose_latents / garment_grid / head_pose
从 jid 缓存读取：pulid_id_embed / appearance
共同读取：固定 prompt embedding
不读取：target_latents
```

因此反事实推理没有像素级 ground truth。评测分别检查身份是否趋向 `jid`、服装是否保持 `mid`、姿势和头朝向是否保持 `mid`。

## 6. 完整样本示例

检查样本：

```text
mid  = 21603
jid  = 21053
seed = 0
run  = A4
```

对应文件：

```text
images/mannequin/21603.png
    人台输入；决定服装和身体姿势

images/human/21603.jpg
    同 ID 配对真人；训练真值和服装/姿势参考

images/human/21053.jpg
    目标身份原始真人；仅用于离线提取 identity appearance

derived/face_crops/human/21053.png
    目标身份紧脸；用于离线提取 PuLID embedding

eval/a4_gen/21603__id21053__seed0.png
    A4 已生成结果
```

正确的检查标准是：

- 生成图的脸、头发应接近 ID `21053`。
- 生成图的服装、版型、花色和身体姿势应接近 ID `21603` 的人台/配对真人。
- 不应要求生成图复现 `human/21053` 的衣服或姿势。

完整检查面板位于：

```text
artifacts/inspection_samples/sample_mid21603_id21053_seed0_a4.png
```

## 7. 当前数据检查结果

2026-08-17 的只读检查结果：

- `human/mannequin/DWPose/garment mask/face crop/head pose/region mask/cache` 对 40,014 个 split ID 均为 100% 覆盖。
- 未发现重复 ID stem。
- 前 100 个 human、mannequin、DWPose 和 garment mask 均为 `768 x 1024`。
- 40,014 个 head-pose JSON 全部为 `status=ok`，没有因低置信被缓存为 null 的样本。
- 数据集类对 train/val/test 的完整缓存 fail-fast 检查通过。
- 按 ID 范围均匀抽查 100 个缓存：所有键形状一致，无 NaN、Inf 或全零条件。
- A4 训练读取 36,033 个样本；差分 mask 对该训练集覆盖完整。

## 8. 已知问题和风险

### 8.1 PuLID fallback 缺少来源记录

缓存构建首先用 `derived/face_crops/human/{id}` 提取 PuLID embedding；若抛出 `RuntimeError`，当前代码会改用完整 `images/human/{id}`。缓存没有记录哪些 ID 触发过该路径，因此现有 embedding 虽然完整、非零，但来源不能逐样本追溯。

建议下次重建时记录 `pulid_source=tight_face/full_image`，并将非预期 fallback 改成 fail-fast 或显式错误清单。

### 8.2 Appearance 可能包含身份来源服装

1.8 倍外扩头部 crop 能保留头发，但也可能包含肩部和领口。该区域经过 appearance token 进入模型，可能把 `jid` 原图中的衣服信息带入生成结果。这是 A4 身份增强后服装稳定性回退的可能来源之一。

### 8.3 Human garment mask 投影依赖配准

`clothes_bySAM/masks/human/{id}` 被应用到 `images/mannequin/{id}`。现有样本尺寸和 ID 对齐，且抽查正确，但这一路径依赖 human/mannequin 像素空间足够一致。当前没有覆盖全部 40,014 对的自动配准分数。

### 8.4 Garment crop 会被拉伸到正方形

服装 crop 在进入 CLIP-L 前直接 resize 到 `224 x 224`，没有保持 crop 宽高比。长裙、长裤等纵向版型可能因此发生几何压缩。它不属于目录接错，但可能削弱 garment token 的版型表达。

### 8.5 Packed region mask 只预构建 train split

`derived/region_masks_z` 没有 val/test 的预构建文件。当前训练没有缺失，现有评测/保护采样也会从 image-level region mask 做确定性内存投影，因此不影响已完成结果；新工具若直接假定所有 eval ID 都有 packed mask，需要先补建或复用现有 fallback。

### 8.6 Cache manifest 不是四分片聚合报告

当前 `manifest.json` 最后记录的是 shard 3，不能单独证明四个 shard 全部成功。数据集对全部 NPZ 的实际逐文件覆盖检查已经通过，因此不存在当前缺失，但未来缓存构建应输出单独 shard manifest，再由 rank 0 汇总。

## 9. 代码位置

- 原始图像到缓存的映射：[`build_cache.py`](../build_cache.py)
- 服装 crop、head crop 和 token 构建：[`conditions.py`](../conditions.py)
- 训练 DataLoader 实际读取内容：[`dataset.py`](../dataset.py)
- 反事实 `mid/jid` 条件组合：[`eval_b2.py`](../eval_b2.py)
- 原生分辨率和当前路径配置：[`configs/warmup.yaml`](../configs/warmup.yaml)
- A4 identity bank 与差分设置：[`configs/a4_directed.yaml`](../configs/a4_directed.yaml)

