# G2 标注工作区

## 文件夹结构

- `source_images/human/`：复制后的真人源图像。
- `source_images/mannequin/`：复制后的人台源图像。
- `stitched_pairs/`：每个 `image_id` 一张拼接图，左侧为真人，右侧为人台。
- `paired_images/rater_01/`、`rater_02/`、`rater_03/`：分别供三位标注者使用的拼接图入口。
- `annotations_rater_01_中文.csv`、`annotations_rater_02_中文.csv`、`annotations_rater_03_中文.csv`：首次提交的中文表，仅保留作审计，不能继续使用。
- `G2_rater_01_修订1.xlsx`、`G2_rater_02_修订1.xlsx`、`G2_rater_03_修订1.xlsx`：本次必须独立填写的修订版表格。
- `G2_adjudication_修订1.xlsx`：三份修订版原始表完成后由裁决者填写；在此之前保持空白。

## 标注顺序

1. 打开对应的 `paired_images/rater_0X/` 拼接图。
2. 在对应的 `G2_rater_0X_修订1.xlsx` 中按 `图像编号` 和 `标注单元编号` 填写黄色列。
3. 三位标注者必须独立完成，不能互看结果。
4. 三份修订版原始表完成后，裁决者填写 `G2_adjudication_修订1.xlsx`。

## 为什么需要修订1

首次提交的三份 CSV 内容和 SHA-256 完全相同，三份均使用 `rater_01`，并且边界多边形为空，无法证明独立标注，也无法计算 boundary IoU。独立 verifier 已判定该提交不能通过 G2。旧文件不得删除或覆盖。

## 标签

- `S_I`：身份图像专属，例如脸部形态、可靠的局部头发。
- `S_M`：人台/场景图像专属，例如头部姿态、服装设计、背景。
- `S_X`：交互或遮挡，例如头发遮脸、头发接触衣物、接触褶皱。
- `S_U`：不确定或无法归属。

`S_X` 和 `S_U` 不计入 exclusive leakage。

## 遮挡顺序

允许值：`none`、`hair_over_face`、`face_over_hair`、`hair_over_garment`、`garment_over_hair`、`ambiguous`。

## 必填字段

- 支持标签代码；
- 遮挡单元的遮挡顺序代码；
- 所有者是否可观察：`yes` 或 `no`；
- 单元权重：正数，建议统一填写 `1.0`；
- 边界多边形 JSON，例如 `[[120,80],[160,80],[170,140],[115,140]]`；
- 无明显伪影时，伪影填写 `none`。

固定字段和原始英文 CSV 不要修改。Excel 中蓝灰色列为固定索引，黄色列为人工输入，绿色列为自动解释。三份修订表完成后需要转换/同步回机器可读字段，完成裁决，才能运行自动门控。
