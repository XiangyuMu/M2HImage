# B 小位移对齐与边界对照

## 2026-09-07 执行补记（优先于下方交接时状态）

主线程已远端执行B32：128/128 outputs，24/32对齐接受，8/32原位回退，无方法失败。
v1全部19个数组测试通过；v2新增输入M内部距离<=16px固定过渡带，覆盖4px腐蚀与8px羽化区域。
原因：原M穿越边统计在四方法完全相同，不能测到内缩编辑接缝。
v2补充固定8/24 RGB偏移与1/3/6px错位经过同样羽化的合成校准，不改渲染方法或筛样。
v1与v2四方法的全部128张图SHA256逐一相同，因此可复用v1完成的dev-ID/DINO/HF-LPIPS/pose评测。
v2全部19个既有测试也通过；新增指标实测显示梯度/Laplacian不可靠单调，不能作为真实瑕疵晋级门槛。
`b_alignment32_v2`耗时118.864s CPU；32 unique M ×11 calibration cases。
完整dev128已由主线程监督器排队，在A同批128张teacher结果上重复四方法及客观评测。
所有失败、回退、覆盖率保留；不据此训练B或宣称第二个创新成立。

## 原始实现交接说明

本实现只新增独立脚本和数组单测；无 SSH 执行、无远端写入、无训练、无依赖下载、无 AdaFace。主 agent 负责上传至**新的脚本目录**和执行。读取本地实验计划及 `scripts/copy_controls.py`、`dev_identity_controls.py`；不修改它们或协议。

## 最快运行路径（由主 agent 执行）

远端已有 Python：`/home/muxiangyu/miniconda3/envs/refton_m2h/bin/python`。先把这两个 Python 文件放到新的 `$R/scripts_b_alignment_v1/`，不要覆盖远端已有文件。以下命令在服务器 shell 执行，变量必须先赋值；本次交付没有执行这些命令。

```sh
P=/data/muxiangyu/pythonPrograms/M2HImage
D=/data/muxiangyu/datasets/M2HImage/M2H_Final_v2
R="$P/artifacts/m2h_ab_objective_v1"
B_SCRIPT_DIR="$R/scripts_b_alignment_v1"
B_PY=/home/muxiangyu/miniconda3/envs/refton_m2h/bin/python

"$B_PY" -B -m unittest discover -s "$B_SCRIPT_DIR" -p test_b_alignment_boundary.py -v

# 可选首对烟测；输出目录必须不存在。
"$B_PY" -B -u "$B_SCRIPT_DIR/b_alignment_boundary.py" \
  --root "$D" --probe "$R/input_only_guard32" \
  --out "$R/b_alignment_smoke1_v1" --limit 1 --threads 2 --max-seconds 300

# B32：四方法，同一个固定配置；CPU，不加载任何模型。
"$B_PY" -B -u "$B_SCRIPT_DIR/b_alignment_boundary.py" \
  --root "$D" --probe "$R/input_only_guard32" \
  --out "$R/b_alignment32_v1" --threads 2 --max-seconds 1800 \
  --dev-identity-layout

# 128：读取新 audit.json 的全部 pairs，不假定一 mid 只能出现一次。
# 执行前由主 agent 检查 manifest 确为 64 M x 2 身份 = 128 对。
"$B_PY" -B -u "$B_SCRIPT_DIR/b_alignment_boundary.py" \
  --root "$D" --probe "$R/input_only_64m" \
  --out "$R/b_alignment128_v1" --threads 2 --max-seconds 3600 \
  --dev-identity-layout
```

拒绝已存在的 `--out`；不覆盖/恢复旧 run。`--limit 0` 默认全量，按 audit 顺序。只在每对开始前检查时间预算，最后一对可超出剩余预算；不是硬超时。预算耗尽后剩余全部方法记失败，写 summary，退出码 2。ECC 拒绝并正常完成原位回退本身不是脚本失败（退出码 0），必须另看 `alignment.identity_fallback`。每对最多两次 60 迭代 ECC，最多长边 512，对齐后在原生分辨率融合/评测。默认 2 CPU 线程、OpenCL 禁用、GPUh=0。

## 固定方法与坐标约定

| 方法目录 | 对齐 | 融合 |
| --- | --- | --- |
| `base` | 无 | 原 base 像素 |
| `inplace_feather` | 原位 | 腐蚀 4 px，羽化 8 px |
| `warp_feather` | 同一双向 ECC 平移 | 同样羽化 |
| `warp_multiband` | 同一双向 ECC 平移 | 4 层 Laplacian RGB / Gaussian alpha |

默认近恒等：两方向各从零平移初始化，不做全局搜索。ECC 用 `template=base, input=source`，返回 **base 坐标到 source 采样坐标**；`output[y,x] = source[y+dy,x+dx]`。如果 source 视觉上应右移 3、上移 2，则采样位移是 `(-3,+2)`。这里使用 `remap`；若改用 `warpAffine`，必须配 `WARP_INVERSE_MAP`。已知平移单测用独立的正向 `warpAffine` 构造目标，避免测试与实现共同用错符号。

位移单位固定为原生像素；缩放后的 ECC x/y 分别除以实际 x/y 缩放率。正反向 Euclidean 位移均须 <=6 px，前后向向量和长度 <=1 px，两相关值均 >=0.5，有效源支持/M >=0.90；低灰度纹理 std<0.015、腐蚀后 ECC ROI<64 个缩小像素、ECC 不收敛也显式拒绝。ROI 为 M 腐蚀 8 px，来自输入；不使用生成 mask。阈值是预先固定工程配置，不声称已对真实匹配正确率校准。

拒绝时两 warp 方法都采用零位移并照常融合，保留候选位移、失败原因和回退状态。不能把回退的漂亮结果算成对齐成功。全局平移前后向检查、纹理及源采样支持是可靠性代理；它们不能判断局部遮挡、真实可见性或人体形变。

编辑 alpha 为输入 M 的边界距离羽化，warp 再乘有效采样支持：图像内有效坐标且线性插值涉及的源 mask 像素全在 M 内。多频带重建后 `alpha<=0` 的像素强制恢复 base，防止 pyramid 空间扩散越过编辑支持；这会影响过渡效果，必须通过边界分数检查。严格掩码外像素不变是实现约束，旧 Poisson 测出的 3.49% 外溢不需复刻。alpha 不用于裁剪评分区域。

## 固定评测区域与可解释范围

所有方法共用输入 M、腐蚀 8 px 内区、膨胀 8 px 减内区的边界带、膨胀区补集。强纹理区在源 M 上冻结：源 Sobel 梯度 >= max(M 内第75分位数, 0.02)。不存在输出解析、q 选择或仅对齐成功区域计分。逐图保留全部区域面积及空区 null。

- M、内区、边界、强纹理区：相对未 warp 源 M 的 RGB MSE、Sobel 向量 L1、高频 L1（Gaussian sigma=2）。这些是源保真度诊断；合法小位移也会增加源坐标误差，不能单独排名对齐质量。HF-L1 不是 HF-LPIPS。
- 无真值边界代理：冻结 M 穿越边的 RGB 跳变均值/p95；跳变减去两侧同方向相邻跳变均值后的正部；固定边界带梯度 p95、Laplacian 绝对值 p95。RGB、梯度的输入强度归一化至 [0,1]，Sobel scale=1/8。真实服装边、阴影、强纹理可能高分，模糊可能低分，**均不是“真实瑕疵率”**。
- 外溢：同时保存严格 M 外和允许区（膨胀 M）外的面积、变化像素数、变化比例、最大 uint8 差值。判定阈值固定为任一 RGB 通道差值 >2，分母不改变。
- 覆盖率：有效采样支持/M、alpha>0/M、alpha 的均值/p10/p90、实际变化/M，以及全量对齐成功/回退/未尝试数。低覆盖仍评测整个固定 M。

没有读取配对真人 H，`paired_boundary_ground_truth` 明确为 null/不可用；没有实现配对 H 的边界 LPIPS。合成 clean-M 真值不可混称配对真人边界真值。此脚本不能判定身份/姿态护栏或 B 学习方法晋级。

## 无人工标注校准

每个成功加载的唯一 mid 做一次源 M 自扰动；两个身份不会重复增加校准 n。固定 6 张：clean、M 内 RGB 加 8/24（0–255 单位，截断）、M 内视觉向右错位 1/3/6 px。M 外保留 clean 像素；包括错位采样的反射边界策略，不作为真实对应真值。所有严重度共用原输入 M 的边界带。

`calibration.jsonl` 并列保存无真值代理分数和明确冠以 `synthetic_*_to_known_clean` 的 RGB MAE / 梯度 L1。聚合给出每个严重度相对 clean 的逐 M 配对分数增量、上升比例、严重度不下降比例及实际 n。没有拟合阈值、自动宣称排序通过或把上升比例称瑕疵率；平坦区域错位可能不可辨，周期纹理可能不单调，均如实反映。纯函数测试只验证受控夹具的预期敏感性；B32 数据实际敏感性必须看运行后的报告。

## 产物与失败解释

- `method_config.json`：完整固定方法配置、预算/命令参数、代码和输入 audit SHA256、库版本、信息来源与局限。
- `per_image.jsonl`：每 pair x method，成功/失败、逐图输入路径/hash、固定区域指标、覆盖率、时间。缺失输入不跳过，四方法均留失败行。
- `summary.json`：全量预期/实际行数、失败与原位回退、方法指标有效 n/缺失 n/均值/p50/p95、配对 base 差值、CPU耗时/GPUh。
- `failures.jsonl`：输入/预算/方法/校准错误和 ECC 拒绝；ECC 拒绝与图像无法生成需区分。
- `alignment.jsonl`、`alignment/<name>.png.json`：双向位移、相关值、FB误差、候选覆盖、接受与回退。
- 四方法目录保存全部原生分辨率 RGB PNG；`alpha/` 保存三编辑方法的 alpha PNG（实际计算为 float32，PNG 是可视化量化）。
- `calibration/<mid>/` 保存所有扰动和 clean PNG；`calibration.jsonl` / `calibration_summary.json` 保存校准结果。
- `audit.json` 保存实际所选输入对，`outputs -> base` 提供 evaluator 的 probe 视图。

统计为探索性条件均值与配对差值，无来源/身份聚类 CI、无真实瑕疵标签、无自动科研门槛。失败留在 expected/missing 分母，但不人为给无量纲边界指标编造失败罚分；不能只比较成功子集忽略失败。输出约为每 pair 4 RGB+3 alpha，另每唯一 M 6 RGB，128 对约512 RGB+384 alpha+384校准 RGB；没有实测磁盘或速度保证。

## 已有 dev_identity_controls 的具体接线

没有重新查找/下载身份模型。原脚本已固定 w600k_r50，且方法名硬编码。`--dev-identity-layout` 额外创建以下硬链接适配目录；**目录名只是旧接口标签，绝不代表真实算法**：

| 旧脚本输出标签 | 真实方法 |
| --- | --- |
| `base` | `base` |
| `feather_e4_f8` | `inplace_feather` |
| `feather_e8_f16` | `warp_feather` |
| `poisson_e4` | `warp_multiband`（不是 Poisson） |

映射另存 `dev_identity_controls/METHOD_MAP.json` 和 `method_config.json`。主 agent 可以后续用原有身份运行环境执行：

```sh
# B_DEV_PY 应使用 STATUS 中已验证 ONNX CUDA 的环境；此处不启动该任务。
B_DEV_PY=/home/muxiangyu/miniconda3/envs/StableAnimator/bin/python
"$B_DEV_PY" -B -u "$R/scripts/dev_identity_controls.py" \
  --root "$D" --probe "$R/b_alignment32_v1" \
  --controls "$R/b_alignment32_v1/dev_identity_controls" \
  --out "$R/b_alignment32_dev_identity_v1" --device 2
```

注意 `--probe` 指向本次 B 输出视图，特别在 `--limit` 烟测时才能维持同一 pair 分母。身份汇总必须按 METHOD_MAP 重标真实方法。这里不更改身份脚本、不读取人脸权重、不做 AdaFace 调用。

## 验证记录

2026-09-07 本机验证：**19/19 单测通过**（unittest 报告 5.320 s），CLI `--help` 退出 0。Python 3.12.14、OpenCV 4.10.0、NumPy 2.3.5。使用已存在的 uv OpenCV 缓存配合 bundled Python，未安装或下载依赖；`-B` 避免新增 pycache。

纯数组单测涵盖：采样恒等、已知整数平移方向、ECC 恒等、正反向及亚像素平移/缩放单位、位移上限/低相关/FB拒绝/非有限输入、低纹理原位回退、插值源支持、固定 ROI、纹理资格、外溢阈值、奇数尺寸多频带和 alpha=0/1、不敏感校准诚实报告、失败分母。远端环境兼容性、文件产物端到端集成、真实 B32/128 指标和预算尚需主 agent 运行验证，本次没有启动远端任务。

本机复验命令（从本地课题四目录）：

```sh
PYTHONPATH=/Users/muxy/.cache/uv/archive-v0/LkrcDAT8cj1GOVUQ \
  /Users/muxy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -B -m unittest discover -s .omx/experiments/m2h_ab_objective_v1/scripts \
  -p test_b_alignment_boundary.py -v
```
