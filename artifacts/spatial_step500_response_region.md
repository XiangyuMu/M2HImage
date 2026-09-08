# Step-500 Spatial-Condition Response Probe

- checkpoint: `/data/muxiangyu/datasets/M2HImage/M2H_Final_v2/phase1/phase1_spatial_conditions_r16_4400_768x1024/checkpoints/step-000500`
- samples: 2
- taus: [0.5]
- metric: RMS velocity change divided by baseline velocity RMS; identity swap is the response reference.

| intervention | all | cloth | face | body/bg | fraction of identity response |
|---|---:|---:|---:|---:|---:|
| garment_swap | 0.202549 | 0.201335 | 0.203017 | 0.203342 | 5.810 |
| hair_swap | 0.013303 | 0.013025 | 0.013593 | 0.013446 | 0.382 |
| hair_off | 0.013562 | 0.013322 | 0.014055 | 0.013702 | 0.389 |
| identity_swap | 0.034861 | 0.034239 | 0.034777 | 0.035068 | 1.000 |

Interpretation: a near-zero swap response means the route is functionally ignored; a material
response with poor decoded fidelity means the route is connected but the flow objective has not
learned the required high-frequency correspondence yet.
