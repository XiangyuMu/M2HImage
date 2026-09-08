# Step-500 Spatial-Condition Response Probe

- checkpoint: `/data/muxiangyu/datasets/M2HImage/M2H_Final_v2/phase1/phase1_spatial_conditions_r16_4400_768x1024/checkpoints/step-000500`
- samples: 4
- taus: [0.8, 0.5, 0.2]
- metric: RMS velocity change divided by baseline velocity RMS; identity swap is the response reference.

| intervention | all | cloth | face | body/bg | fraction of identity response |
|---|---:|---:|---:|---:|---:|
| garment_swap | 0.112360 | nan | nan | nan | 2.817 |
| hair_swap | 0.017374 | nan | nan | nan | 0.436 |
| hair_off | 0.016932 | nan | nan | nan | 0.425 |
| identity_swap | 0.039884 | nan | nan | nan | 1.000 |

Interpretation: a near-zero swap response means the route is functionally ignored; a material
response with poor decoded fidelity means the route is connected but the flow objective has not
learned the required high-frequency correspondence yet.
