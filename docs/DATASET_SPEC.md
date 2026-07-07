# MCIC Dataset And Cache Specification

## Source Pairs

数据根目录包含 `mannequin/` 与 `human/`。配对键是去除扩展名后的相对路径；因此子目录也必须匹配。支持 `.jpg`、`.jpeg`、`.png`、`.webp`。

```text
root/mannequin/look_a/0001.jpg
root/human/look_a/0001.png
```

该样本的 `sample_id` 为 `look_a/0001`。

每对图像应为同一服装、同一或近似姿势、近似构图对齐。像素级服装约束不适用于跨姿态或跨构图配对；这些样本应在审计阶段排除或另行设计特征级监督。

## Parsing Labels

正式实验使用离线生成的离散 parsing label maps：

```text
/data/project/parsing_labels/look_a/0001.png
```

配置中的 `label_ids` 将解析器类别归并为：

| Mask | 需要覆盖的区域 |
|---|---|
| `face` | 面部 |
| `hair` | 头发 |
| `cloth` | 上装、裤/裙、连衣裙等待保留服装 |
| `person` | 人体全部前景区域 |

`heuristic` backend 会画出粗略几何区域，只用于测试文件流和代码接口；它不可替代 human parsing。

## Derived Masks

对 human 标签构造训练 mask：

```text
identity_paint = face union hair
expanded_identity = dilate(identity_paint, boundary_dilate_radius)
cloth_safe = erode(cloth - expanded_identity, cloth_erode_radius)
editable_human = person - cloth_safe
ambiguous_boundary = expanded_identity - erode(identity_paint)
```

对应缓存名称：

| 文件后缀 | metadata 字段 | 训练用途 |
|---|---|---|
| `_cf_mask.png` | `cf_mask_path` | CF identity 区域 inpainting |
| `_paired_mask.png` | `paired_mask_path` | Paired 真人化可编辑区域 |
| `_cloth_safe_mask.png` | `cloth_safe_mask_path` | garment L1 |
| `_ambiguous_mask.png` | 未送入首版训练 | 边界诊断和后续 edge loss |

## Identity Cache

首版没有 `person_id`，每条 human 图产生一个身份 embedding：

```text
cache_mcic/faces/<sample_id>.png
cache_mcic/identity_embeddings/<sample_id>.npy
```

正式 Exp3 使用 `facenet` backend，离线目标 embedding 与训练时可微生成脸 encoder 必须一致。`mock` embedding 仅保证测试时接口一致。

## Metadata Schema

`cache_mcic/metadata.jsonl` 一行一个 JSON object：

```json
{
  "sample_id": "0001",
  "mannequin_path": "/data/project/dataset/mannequin/0001.jpg",
  "human_path": "/data/project/dataset/human/0001.jpg",
  "original_mannequin_size": [768, 1024],
  "original_human_size": [768, 1024],
  "paired_mask_path": ".../0001_paired_mask.png",
  "cf_mask_path": ".../0001_cf_mask.png",
  "cloth_safe_mask_path": ".../0001_cloth_safe_mask.png",
  "face_box": [300, 100, 470, 270],
  "identity_embedding_path": ".../0001.npy",
  "face_quality_pass": true,
  "split": "train",
  "parsing_backend": "label_maps"
}
```

`face_quality_pass=false` 的样本保留用于排查，但 dataset 不送入训练。

## Resolution And Geometry

缓存 mask 始终维持原图尺寸。dataloader 对 mannequin、human 及其 masks 应用相同 letterbox transform：

- 按比例缩放至目标 `image.height`、`image.width` 内。
- 剩余画布用配置值 padding。
- RGB 使用 bicubic，mask 使用 nearest。

配置要求高度和宽度都为 64 的倍数。对不同原始尺寸的 paired 图像，审计报告会列出 mismatch；正式训练应确认这些样本几何仍可比较或将其排除。

## Quality Checklist

- 配对缺失数为零，非法图像数为零。
- 人工查看至少 200 对原图及 mask overlay。
- face/hair mask 不包含大面积服装。
- `cloth_safe` 不侵入脸、头发或领口模糊边界。
- test split 固定后不用于训练选择或负样本跨拆分采样。
