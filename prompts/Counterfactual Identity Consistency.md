

数据	是否有	用途
人台图像 m_i	有	服装、姿势、人体布局、空间结构来源
对应真人图像 h_i	有	paired supervision / 服装 teacher / identity source
人体分割图	有	构造 face、hair、skin、cloth、body、background 等区域
无头部 DWPose	有	控制身体姿势，但不能控制头部/脸部
SAM 提取的服装 mask	有	提供更可靠的服装区域和服装保持 loss



最终方法名

建议命名为：

MA-RA-CDT

全称：

Mannequin-Anchored Region-Adaptive Counterfactual Differential Training

中文：

人台锚定的区域自适应反事实差分训练

相比之前的名字，我建议把 MCIC-Flow 改成 CDT / Counterfactual Differential Training，因为最终真正成立的反事实机制不是“给 c_j 伪造一个完整 GT”，而是：

同一张人台图像，在不同身份条件下生成结果；要求服装/姿势区域一致，身份区域随身份条件变化。

这个差分式反事实结构更严谨，也不需要伪造不存在的 h_{i\rightarrow j}。

⸻

1. 任务定义

给定 paired 数据：

\mathcal{D}=\{(m_i,h_i)\}_{i=1}^{N}

其中：

* m_i：人台图像；
* h_i：对应真人图像。

从真人图像中提取身份条件：

c_i = R(h_i)

其中 R(\cdot) 可以是 RetinaFace + ArcFace / InsightFace。

目标是训练模型：

G(m_i,c)\rightarrow \hat{h}

要求：

1. 服装来自人台图像；
2. 身体姿势来自人台图像；
3. 人脸、头发、身份相关区域来自身份条件；
4. 换身份时服装不变；
5. 换身份时姿势不变；
6. 生成结果是真人照片，不是贴脸或人台换脸。

⸻

2. Backbone 选择

推荐使用：

FLUX.1 Fill-dev + LoRA / Adapter

但注意，它不能被简单当成普通局部 inpainting 模型。

你的任务不是：

保留未 mask 区域，只补脸。

而是：

基于人台图像和身份条件，重渲染成真人。

所以 FLUX Fill 在这里被用作：

structure-guided image generation backbone。

也就是说：

* 人台图像提供结构 canvas；
* 服装 mask 提供服装区域；
* DWPose 提供身体姿势；
* 身份 adapter 提供目标身份；
* FLUX Fill 负责生成真人图像。

⸻

3. 现有数据如何映射到模型输入

3.1 人台图像 m_i

用于三件事：

1）构造 source canvas

不直接用原始人台 RGB，也不把整个人台完全清空，而是构造：

x_i^{src}
=
(1-M_{fg})\odot m_i
+
M_{fg}\odot \operatorname{LowPass}(m_i)

其中：

* M_{fg}：人台前景区域，可以由人体分割图或 SAM/简单前景 mask 得到；
* \operatorname{LowPass}(m_i)：模糊后的人台图。

这样做的作用是：

保留	去除
服装轮廓	人台塑料质感
人体姿势	假人材质
服装大致位置	错误高频纹理
空间布局	过拟合人台像素

⸻

2）提取 garment local condition

用 SAM 服装 mask：

M_{cloth}^{sam}

裁剪服装区域：

g_i = m_i \odot M_{cloth}^{sam}

提取服装局部 tokens：

T_{cloth}=E_{cloth}(g_i)

注意这里不要只用 16 个全局 token，建议保留空间结构：

T_{cloth}^{grid}\in \mathbb{R}^{B\times H_cW_c\times d}

例如：

16×12 或 24×18 patch tokens

这样更利于保留：

* logo；
* 印花；
* 领口；
* 袖口；
* 裙摆；
* 局部褶皱。

⸻

3）提取 mannequin global tokens

从整张人台图像提取全局语义：

T_m=E_m(m_i)

用于补充整体风格和布局。

⸻

3.2 对应真人图像 h_i

用于监督，不作为 CF branch 的 canvas。

它的作用包括：

1. paired flow target；
2. identity source c_i；
3. 服装区域 teacher；
4. 真实感监督；
5. 训练指标计算。

关键点：

h_i 不能作为 counterfactual branch 的输入 canvas，否则模型会学成真人图换脸。

⸻

3.3 人体分割图

人体分割图用于构造区域 mask。

建议构造：

强身份区域

M_{id}^{strong}=M_{face}\cup M_{hair}

弱身份区域

M_{id}^{weak}=M_{neck}\cup M_{skin}\cup M_{hand}

如果分割类别不够细，可以用：

* face；
* hair；
* arms；
* legs；
* exposed skin；

近似。

服装区域

优先使用 SAM 服装 mask：

M_{cloth}=M_{cloth}^{sam}

如果 SAM mask 有漏，可以与 parsing 中的 clothing 区域合并：

M_{cloth}=M_{cloth}^{sam}\cup M_{cloth}^{parsing}

安全服装区域

M_{cloth}^{safe}
=
M_{cloth}
\cdot
(1-M_{id}^{strong})
\cdot
(1-M_{id}^{weak})
\cdot
(1-M_{edge})

其中：

M_{edge}
=
\operatorname{Dilate}(M_{cloth},r)
-
\operatorname{Erode}(M_{cloth},r)

建议：

r = 5~15 pixels

⸻

3.4 无头部 DWPose

你目前的 DWPose 没有头部，这是一个重要限制。

它可以很好控制：

* 躯干；
* 手臂；
* 腿；
* 身体姿势；
* 人体大结构。

但它不能控制：

* 头部朝向；
* 脸部关键点；
* 发型轮廓；
* 表情；
* 头发体积。

因此最终方案中：

DWPose 只作为身体姿势条件，不承担头部身份控制。

头部由：

* identity adapter；
* face / hair region mask；
* source canvas 中的弱头部位置先验；

共同控制。

⸻

4. Source Canvas 的最终设计

由于你没有头部 DWPose，source canvas 需要区分身体区域和头部区域。

⸻

4.1 身体和服装区域

身体 / 服装区域保留人台低频结构：

x_{body}
=
\operatorname{LowPass}(m_i)

这样可以保留：

* 服装位置；
* 身体姿态；
* 裙摆/袖子/裤腿轮廓；
* 粗略服装纹理分布。

⸻

4.2 头部区域

头部区域不要保留人台假头轮廓。

因为人台通常是：

* 光头；
* 无五官；
* 头型简化；
* 和真人发型几何不同。

所以头部区域 source canvas 建议使用：

x_{head}=\eta_{head}

其中 \eta_{head} 可以是：

* very blurred head region；
* soft ellipse；
* gray / mean color；
* 只保留头部大致 bounding box。

即：

头部只给大致位置，不给人台假头形状。

⸻

4.3 最终 source canvas

x_i^{src}
=
M_{bg}\odot m_i
+
M_{body}\odot \operatorname{LowPass}(m_i)
+
M_{head}\odot \eta_{head}

其中：

* M_{bg}：背景；
* M_{body}：除头部外的人台前景；
* M_{head}：face/hair/head 区域。

这比“整个人台灰掉”更稳，也比“直接输入原始人台”更不容易复制塑料感。

⸻

5. Region-Adaptive Condition Corruption

区域自适应扰动只作用于 condition canvas，不作用于 flow endpoint。

构造：

M_{ra}
=
\rho_sM_{id}^{strong}
+
\rho_wM_{id}^{weak}
+
\rho_eM_{edge}
+
\rho_cM_{cloth}^{safe}
+
\rho_bM_{bg}

推荐：

区域	\rho
face + hair	1.0
neck / hand / skin	0.3
cloth edge	0.2
cloth safe	0.02
background	0.0

构造 CF canvas：

x_i^{ra}
=
(1-M_{ra})\odot x_i^{src}
+
M_{ra}\odot \eta_i

其中：

\eta_i=\operatorname{LowPass}(m_i)

或：

\eta_i=\operatorname{Blur}(m_i)

作用是：

* 身份区域更自由；
* 弱身份区域可变；
* 边界可微调；
* 服装区域基本稳定；
* 背景不变。

⸻

6. 条件输入设计

最终模型输入条件包括：

条件	来源	作用
source canvas x_i^{src}	人台图构造	稠密空间结构
SAM 服装 mask	已有	服装区域定位
garment grid tokens	人台服装 crop	局部服装细节
no-head DWPose	已有	身体姿势控制
parsing map	已有	人体/服装/身份区域结构
identity adapter	真人 face crop	身份控制
text prompt	固定 prompt	真实人像先验

⸻

6.1 Text prompt

主线使用固定 prompt：

a photorealistic human model wearing the same garment, same body pose, realistic face and hair

建议不要把 text identity 作为主线 claim。

⸻

6.2 Garment grid tokens

从人台服装区域提取：

T_{cloth}^{grid}
=
E_{cloth}(m_i\odot M_{cloth}^{sam})

必须保留空间位置编码。

否则 token 虽然知道“是什么衣服”，但不知道“印花在哪个位置”。

⸻

6.3 DWPose condition

由于没有头部 DWPose，它只作为 body pose：

P_i^{body}

用于约束：

* 躯干；
* 手臂；
* 腿；
* 身体轮廓。

⸻

6.4 Identity adapter

使用成熟 identity adapter：

PuLID / InstantID / InfiniteYou-like identity adapter

它是身份控制的主路径。

不要只依赖 ArcFace loss。

⸻

7. 训练分支

最终训练由三个部分组成：

1. Paired Flow Branch；
2. Differential Counterfactual Branch；
3. Teacher Cloth Invariance Regularizer。

⸻

8. Paired Flow Branch

这是基础 M2H 分支。

目标：

(m_i,c_i)\rightarrow h_i

编码真人图：

z_0^i=E(h_i)

采样噪声：

z_1\sim\mathcal{N}(0,I)

采样 timestep：

\tau\sim U(0,1)

构造：

z_\tau^i=(1-\tau)z_0^i+\tau z_1

target velocity：

v^*=z_1-z_0^i

预测：

\hat{v}_i
=
v_\theta
(
z_\tau^i,
\tau,
x_i^{src},
M_{edit},
T_{cloth}^{grid},
P_i^{body},
T_{id}(c_i)
)

paired loss：

\mathcal{L}_{pair}^{flow}
=
\|\hat{v}_i-v^*\|_2^2

这个分支负责学习：

* 人台到真人；
* 服装重渲染；
* 姿势保持；
* 真实感；
* 原 paired identity。

⸻

9. Differential Counterfactual Branch

这是最终反事实机制的核心。

它不再给 c_j 伪造 endpoint，也不再要求：

G(m_i,c_j)\rightarrow h_i

而是对同一人台，输入两个身份：

c_j,\quad c_k

分别生成：

\hat{z}_0^j
=
z_\tau^i
-
\tau
v_\theta
(
z_\tau^i,\tau,x_i^{ra},c_j
)

\hat{z}_0^k
=
z_\tau^i
-
\tau
v_\theta
(
z_\tau^i,\tau,x_i^{ra},c_k
)

然后做差分约束。

⸻

9.1 非身份区域一致

\mathcal{L}_{diff}^{nonid}
=
\left\|
M_{nonid,z}
\odot
(
\hat{z}_0^j-\hat{z}_0^k
)
\right\|_1

其中：

M_{nonid}=1-M_{id}^{strong}-M_{id}^{weak}

⸻

9.2 服装区域一致

\mathcal{L}_{diff}^{cloth}
=
\left\|
M_{cloth,z}^{safe}
\odot
(
\hat{z}_0^j-\hat{z}_0^k
)
\right\|_1

作用：

同一人台换不同身份，服装区域应保持一致。

⸻

9.3 身份区域响应

\mathcal{L}_{diff}^{sens}
=
\max
\left(
0,
\delta
-
\left\|
M_{id,z}^{strong}
\odot
(
\hat{z}_0^j-\hat{z}_0^k
)
\right\|_1
\right)

这个 loss 表达：

换身份条件时，身份区域应该发生变化。

注意它不是让脸像某个不存在的 GT，而只是要求身份区域对条件变化敏感。

⸻

9.4 低频 identity alignment

每隔 K 步 decode 一次：

\hat{h}^{j}=D(\hat{z}_0^j)

\hat{h}^{k}=D(\hat{z}_0^k)

检测人脸并对齐后：

\mathcal{L}_{id}^{j}
=
1-\cos(F(\hat{h}^{j}),F(c_j))

\mathcal{L}_{id}^{k}
=
1-\cos(F(\hat{h}^{k}),F(c_k))

只有 face quality 通过时才计算。

⸻

10. Teacher Cloth Invariance Regularizer

虽然没有反事实 GT，但服装区域可以用 paired human h_i 作为 teacher。

对某个身份 c_j 的输出：

\hat{z}_0^j

约束其服装区域接近 h_i：

\mathcal{L}_{teach}^{cloth}
=
\left\|
M_{cloth,z}^{safe}
\odot
(
\hat{z}_0^j-z_0^i
)
\right\|_1

这个 loss 只作用于服装安全区域。

不要作用于：

* face；
* hair；
* neck；
* hand；
* skin；
* edge ambiguous region。

否则会强行把身份区域拉回 h_i。

⸻

11. 总损失

最终：

\mathcal{L}
=
\mathcal{L}_{pair}^{flow}
+
\lambda_{diff}
(
\mathcal{L}_{diff}^{cloth}
+
\mathcal{L}_{diff}^{nonid}
+
\lambda_{sens}\mathcal{L}_{diff}^{sens}
)
+
\lambda_{teach}
\mathcal{L}_{teach}^{cloth}
+
\mathbb{1}_{decode}
\lambda_{id}
(
\mathcal{L}_{id}^{j}
+
\mathcal{L}_{id}^{k}
)
+
\lambda_{tri}\mathcal{L}_{tri}

第一版可以简化为：

\mathcal{L}
=
\mathcal{L}_{pair}^{flow}
+
\lambda_{diff}
\mathcal{L}_{diff}^{cloth}
+
\lambda_{teach}
\mathcal{L}_{teach}^{cloth}
+
\mathbb{1}_{decode}
\lambda_{id}
\mathcal{L}_{id}

后续再加入：

* \mathcal{L}_{diff}^{sens}；
* triplet；
* hand/skin consistency；
* face realism loss。

⸻

12. Timestep 策略

Paired branch

\tau\sim U(0,1)

Differential loss

覆盖更宽范围：

\tau\in[0.2,0.8]

因为它主要比较非身份区域一致性，不要求高质量人脸 decode。

Identity loss

更保守：

\tau\in[0.35,0.7]

避免：

* 低 \tau：原图锚太强；
* 高 \tau：single-step decode 太糊。

⸻

13. 训练阶段

Phase 0：数据与诊断

先可视化：

* source canvas；
* M_{cloth}^{sam}；
* M_{cloth}^{safe}；
* no-head DWPose；
* parsing；
* region-adaptive mask；
* identity crop。

必须确认：

服装 mask 准、source canvas 不复制塑料感、头部区域没有强人台假头轮廓。

⸻

Phase 1：Paired warmup

训练：

\mathcal{L}_{pair}^{flow}

目标：

学会基础 M2H。

⸻

Phase 2：Differential garment consistency

加入：

\mathcal{L}_{diff}^{cloth}
+
\mathcal{L}_{teach}^{cloth}

不加 identity loss。

目标：

同一人台换不同身份时，服装区域稳定。

⸻

Phase 3：Identity alignment

加入：

\mathcal{L}_{id}^{j}
+
\mathcal{L}_{id}^{k}

decode frequency：

K = 2~4

目标有效 identity loss rate：

>10\%\sim15\%

⸻

Phase 4：Full enhancement

逐步加入：

* \mathcal{L}_{diff}^{sens}；
* triplet；
* face realism metric；
* hand/skin consistency；
* garment grid token ablation；
* reference-editing backbone 对比。

⸻

14. 推荐配置

model:
  base: FLUX.1-Fill-dev
  train_lora: true
  lora_rank: 8_or_16
  train_text_encoder: false
  train_vae: false
inputs:
  mannequin_image: true
  paired_human_image: true
  human_parsing: true
  dwpose_no_head: true
  sam_cloth_mask: true
source_canvas:
  body_region: lowpass_mannequin
  head_region: weak_position_prior
  background: original
  avoid_original_mannequin_texture: true
conditioning:
  identity_adapter: PuLID_or_InstantID_style
  garment_tokens:
    type: grid_tokens
    source: mannequin_cloth_crop
    use_position_encoding: true
  pose_condition:
    type: no_head_dwpose
    use_for_body_only: true
  parsing_condition: optional
region_masks:
  cloth_mask: sam_cloth_mask
  cloth_safe: true
  id_strong: face_plus_hair_from_parsing
  id_weak: skin_hand_neck_from_parsing
  edge: dilate_minus_erode_cloth
training:
  resolution: 512
  batch_size_per_gpu: 1
  gradient_accumulation: 16
  precision: bf16
  optimizer: paged_adamw8bit
  lr: 5e-5
  gradient_checkpointing: true
schedule:
  total_steps: 30000
  paired_warmup_steps: 6000
  diff_start: 6000
  identity_start: 15000
loss:
  lambda_pair: 1.0
  lambda_diff: 0.2
  lambda_teach_cloth: 0.5
  lambda_id: 0.05
  lambda_tri: 0.02
  lambda_sens: 0.05
timestep:
  paired: [0.0, 1.0]
  differential: [0.2, 0.8]
  identity: [0.35, 0.7]
identity:
  decode_freq: 2_to_4
  face_quality_filter: true
  target_identity_loss_rate: 0.10
  heldout_eval_model: AdaFace_or_CurricularFace

⸻

15. 关键伪代码

# paired branch
z0_i = vae.encode(h_i)
z1 = torch.randn_like(z0_i)
tau = sample_tau()
z_tau = (1 - tau) * z0_i + tau * z1
v_target = z1 - z0_i
x_src = build_source_canvas(
    mannequin=m_i,
    parsing=parsing_i,
    cloth_mask=sam_cloth_mask_i,
)
cond_i = build_condition(
    canvas=x_src,
    garment_grid_tokens=garment_tokens_i,
    pose_no_head=dwpose_i,
    identity=c_i,
)
v_i = model(z_tau, tau, cond_i)
loss_pair = mse(v_i, v_target)
# differential counterfactual branch
c_j, c_k = sample_two_identities(identity_bank, i)
x_ra = build_region_adaptive_canvas(
    x_src=x_src,
    parsing=parsing_i,
    cloth_mask=sam_cloth_mask_i,
)
cond_j = build_condition(x_ra, garment_tokens_i, dwpose_i, identity=c_j)
cond_k = build_condition(x_ra, garment_tokens_i, dwpose_i, identity=c_k)
v_j = model(z_tau, tau, cond_j)
v_k = model(z_tau, tau, cond_k)
z0_j = z_tau - tau * v_j
z0_k = z_tau - tau * v_k
loss_diff_cloth = l1(
    mask=M_cloth_safe_z,
    pred=z0_j,
    target=z0_k,
)
loss_teach_cloth = l1(
    mask=M_cloth_safe_z,
    pred=z0_j,
    target=z0_i,
)
loss = loss_pair + lambda_diff * loss_diff_cloth + lambda_teach * loss_teach_cloth
# low-frequency identity loss
if step >= identity_start and step % decode_freq == 0 and tau_in_identity_range(tau):
    h_j = vae.decode(z0_j)
    h_k = vae.decode(z0_k)
    if face_quality_passes(h_j):
        loss += lambda_id * identity_loss(h_j, c_j)
    if face_quality_passes(h_k):
        loss += lambda_id * identity_loss(h_k, c_k)

⸻

16. Ablation 设计

必须做：

实验	设置	目的
B1	paired only	基础 M2H
B2	paired + identity adapter	adapter 是否足够
A1	B2 + teacher cloth only	teacher cloth 是否有效
A2	B2 + differential cloth consistency	差分反事实是否有效
A3	A2 + region-adaptive canvas	区域扰动是否有效
A4	A3 + identity loss	身份对齐是否提升
A5	A4 + diff sensitivity	身份区域响应是否有效
A6	no garment grid tokens	服装局部 token 必要性
A7	no DWPose	姿势条件必要性
A8	original mannequin canvas	是否复制塑料感
A9	full gray foreground canvas	是否丢空间对齐
A10	low-pass source canvas	最终 canvas 有效性

最关键的是：

\text{identity adapter only}
\quad vs \quad
\text{identity adapter + differential CF}

如果 differential CF 有效，应表现为：

* 换身份时服装更稳定；
* 姿势更稳定；
* \Delta ID 不下降；
* face realism 不下降；
* garment similarity 提升。

⸻

17. 评价指标

Paired generation

* FID / KID；
* LPIPS / SSIM；
* garment SSIM / LPIPS；
* DINO garment similarity；
* DWPose keypoint distance；
* parsing IoU。

Counterfactual transfer

同一人台 m_i，输入多个身份 c_j。

计算：

Sim(\hat{h}_{i\rightarrow j},c_j)

Sim(\hat{h}_{i\rightarrow j},c_i)

\Delta ID
=
Sim(\hat{h}_{i\rightarrow j},c_j)
-
Sim(\hat{h}_{i\rightarrow j},c_i)

同时计算：

GarmentSim(\hat{h}_{i\rightarrow j_1},\hat{h}_{i\rightarrow j_2})

以及：

* face realism；
* face FID；
* face detector confidence；
* human preference。

⸻

18. 最终贡献表述

建议论文贡献写成三点：

Contribution 1

提出以人台为锚点的反事实身份生成任务，将 M2H 建模为固定服装/姿势条件下的身份干预问题。

Contribution 2

提出 differential counterfactual training，不需要真实 h_{i\rightarrow j}，通过比较同一人台在不同身份条件下的两个预测，实现服装/姿势不变、身份区域变化的反事实监督。

Contribution 3

提出 region-adaptive condition corruption 和 garment-aware supervision，结合 SAM 服装 mask、人体分割和无头 DWPose，在现有数据条件下实现身份-服装-姿势解耦。

⸻

最终一句话总结

在你当前数据条件下，最终方案应是：

使用 FLUX.1 Fill-dev 作为 structure-guided generation backbone，以人台图构造 low-pass / edge / weak-head source canvas，以 SAM 服装 mask 和人体分割构造安全服装区域，以无头 DWPose 控制身体姿势，以 identity adapter 控制目标身份。训练时用 paired flow branch 学习基础人台真人化，再用 differential counterfactual branch 对同一人台输入两个不同身份，约束二者在服装/非身份区域一致、在身份区域有响应，并低频用 identity loss 对齐目标身份。这样不需要伪造反事实 GT，也避免了 c_j 对应 h_i endpoint 的矛盾，更适合你当前实际数据。