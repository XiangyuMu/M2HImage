# MCIC Model Design

## Objective

给定人台图 `m_i` 和身份参考 `c`，模型输出真人化图像。paired 数据提供 `h_i`，但反事实身份 `c_j` 没有完整目标图，因此 MCIC 只监督能够确定的区域和身份关系。

## Backbone And Conditioning

主干使用 SDXL Inpainting UNet 的原生九通道输入：

```text
[noisy target latent (4), paint mask (1), masked mannequin latent (4)]
```

实现冻结 VAE 和 text encoders，并向 UNet attention 层应用 LoRA。条件包括：

- SDXL 文本 prompt embeddings。
- CLIP vision 对 mannequin 图像提取的 patch tokens，经 projector 压缩到 `mannequin_tokens`。
- 缓存身份 embedding 经 projector 产生 `identity_tokens`。

每个 cross-attention block 使用独立 gated processor：文本 attention 输出保持基础路径，mannequin 与 identity 分别通过额外 K/V projection 形成残差项；两个 gate 初始化为零，因此训练开始时新增条件残差为零。UNet 的基础投影通过 LoRA 适配，额外 processor 和条件 projector 共同训练。

## Paired Branch

```text
condition: mannequin_i, identity_i
clean target: human_i
paint mask: editable_human
visible inpaint source: mannequin_i outside editable_human
```

用标准 diffusion noise prediction loss 训练，并按 latent mask 加权：

```text
L_pair = weighted_mse(epsilon_pred, epsilon,
                      weight(editable)=1.0,
                      weight(context)=0.1)
```

服装安全区域仍作为可见 inpainting source，减轻纹理被无谓重建的问题。

## Counterfactual Branch

```text
condition: mannequin_i, identity_j where j != i
paint mask: face union hair
visible source: mannequin_i outside identity region
```

CF branch 使用目标真人 `h_i` 的 latent 作为训练噪声载体，但仅在低/中噪声 timestep 范围解码单步 `x0` 预测，以计算下列监督：

```text
L_cloth = L1(output, h_i) on cloth_safe
L_id = 1 - cosine(F_face(output), embedding_j)
L_tri = relu(d(output, j) - d(output, i) + margin)
```

总损失：

```text
L_total = lambda_pair * L_pair
        + lambda_cf * (
              lambda_cloth * L_cloth
            + lambda_id * L_id
            + lambda_tri * L_tri)
```

首版默认：

```text
lambda_cf=0.2, lambda_cloth=1.0,
lambda_id=0.05, lambda_tri=0.03, margin=0.25
```

当正式身份后端为 `facenet` 时，Exp3 对单步生成图运行冻结的人脸检测质量门控；只对检测成功的脸执行可微裁剪与身份损失，检测失败样本仍保留服装损失并记录 pass rate。

## Train/Inference Consistency

Masked source latent 在 paired 和 CF 分支都来自 mannequin，而不是 human 目标图。推理端同样使用 mannequin masked source，并在最终输出中将 mask 外像素与输入 source 合成，以执行“保留服装”这一产品定义。

## Identity Backend

- `mock`：固定随机投影的低成本可微编码器，只验证梯度和 I/O，不用于身份指标。
- `facenet`：离线 embedding 和训练中可微 encoder 使用相同的冻结 InceptionResnetV1 权重；作为首版正式身份监督。

计划文档中提出的 InsightFace/ArcFace 可作为后续独立评测模型，防止训练和评测使用同一 embedding backbone 造成指标偏乐观。

## Deferred Ablations

以下模块未启用在核心代码路径中，应在 Exp3 成立后逐一实现并比较：

- Semi-hard counterfactual identity sampling。
- 检测失败时低频 multi-step identity fallback。
- 发际线、领口边界 edge loss。
- 手、脖子等弱肤色/身份区域。
- 独立 standard condition-sensitivity branch。
