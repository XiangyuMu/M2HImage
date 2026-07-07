# Experiment Log Template

复制本模板为 `docs/experiments/<experiment_id>.md`，并将机器生成的结果文件保存在 `outputs/<experiment_id>/`。

## Run Identity

| Field | Value |
|---|---|
| Experiment ID | |
| Date / Operator | |
| Purpose / Hypothesis | |
| Status | planned / running / completed / invalid |
| Git commit or source snapshot | |

## Inputs And Configuration

| Field | Value |
|---|---|
| Config path | |
| CLI overrides | |
| Dataset root / revision | |
| Parsing backend and label schema | |
| Identity backend | |
| Split seed | |
| Initial checkpoint | |
| Base model / encoder weights | |

## Runtime

| Field | Value |
|---|---|
| Resolution | |
| GPUs / CUDA / PyTorch | |
| Batch size per GPU | |
| Gradient accumulation | |
| Total steps | |
| Wall time | |
| Peak GPU memory | |
| Output directory | |

## Results

| Metric | Value | Baseline | Change |
|---|---:|---:|---:|
| paired cloth L1 | | | |
| CF cloth L1 | | | |
| target identity similarity | | | |
| source identity similarity | | | |
| Delta ID | | | |

## Visual Inspection

| Check | Observation / artifact path |
|---|---|
| Typical successful outputs | |
| Identity failure cases | |
| Garment drift cases | |
| Boundary or mask artifacts | |

## Conclusion

State whether the hypothesis is supported, what is invalid or uncertain, and the single next experiment justified by this run.
