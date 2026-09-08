# Fine-grained Inference Conditions: Random 10

- Source selection: `/data/muxiangyu/pythonPrograms/M2HImage/artifacts/spatial_quality_repair_random10/manifest.json`
- Final resolved config: `/data/muxiangyu/datasets/M2HImage/M2H_Final_v2/phase1/phase1_spatial_quality_repair_r16_6400_768x1024/resolved_config.yaml`
- Selection seed: `20260824` (the original ten samples are unchanged)
- Overview: [overview_conditions_random10.png](overview_conditions_random10.png)
- Machine-readable provenance: [manifest_conditions.json](manifest_conditions.json)

## What Each Panel Shows

Top row (mannequin-side): mannequin source, with-head DWPose ControlNet image, exact eroded garment mask, exact masked garment image before VAE encoding, and the existing generated output.

Bottom row (identity-side): target person, exact FASHN hair mask, exact masked hair image before VAE encoding, exact face+hair-only crop before pooled CLIP, and the face crop used to build the PuLID embedding.

The bottom metadata band reports the actual cached tensor shapes and numeric head-pose token. Latent tensors and embeddings are not RGB images; their deterministic source images are shown instead.

`region_masks_z` is not an inference condition and is intentionally absent. Legacy garment CLIP tokens and legacy hair tokens are disabled in this run.

## Preprocessing

- Garment/hair reference background: RGB `227`; mask erosion: `2px`.
- Hair label: FASHN `2`; empty threshold: `0.500%`; model stride: `2` (`768` tokens).
- Appearance crop background: RGB `127`; retained labels: face=1 and hair=2.
- ControlNet: mode `4`, scale `0.75`.

## Provenance Warning

cache manifest reference_preprocessing is stale: manifest={'neutral_gray': 127, 'mask_erosion_px': 0, 'hair_ref_min_area_fraction': 0.01}, final resolved config={'neutral_gray': 227, 'mask_erosion_px': 2, 'hair_ref_min_area_fraction': 0.005}. Panels use the final run's resolved config, which is authoritative for the quality-repair cache rebuild.

## Samples

| # | mid | jid | seed | garment area | hair area | panel |
|---:|---|---|---:|---:|---:|---|
| 1 | `28705` | `12467` | 1 | 13.43% | 0.81% | [conditions_01__mid28705__jid12467__seed1.png](conditions_01__mid28705__jid12467__seed1.png) |
| 2 | `20743` | `32740` | 0 | 53.35% | 0.91% | [conditions_02__mid20743__jid32740__seed0.png](conditions_02__mid20743__jid32740__seed0.png) |
| 3 | `41016` | `32740` | 1 | 25.36% | 0.91% | [conditions_03__mid41016__jid32740__seed1.png](conditions_03__mid41016__jid32740__seed1.png) |
| 4 | `01203` | `16010` | 1 | 29.29% | 0.00% | [conditions_04__mid01203__jid16010__seed1.png](conditions_04__mid01203__jid16010__seed1.png) |
| 5 | `26832` | `20367` | 1 | 22.91% | 0.97% | [conditions_05__mid26832__jid20367__seed1.png](conditions_05__mid26832__jid20367__seed1.png) |
| 6 | `02725` | `12467` | 1 | 10.85% | 0.81% | [conditions_06__mid02725__jid12467__seed1.png](conditions_06__mid02725__jid12467__seed1.png) |
| 7 | `28345` | `42839` | 1 | 32.75% | 0.82% | [conditions_07__mid28345__jid42839__seed1.png](conditions_07__mid28345__jid42839__seed1.png) |
| 8 | `46129` | `18372` | 0 | 24.96% | 0.51% | [conditions_08__mid46129__jid18372__seed0.png](conditions_08__mid46129__jid18372__seed0.png) |
| 9 | `28853` | `47050` | 1 | 37.65% | 2.10% | [conditions_09__mid28853__jid47050__seed1.png](conditions_09__mid28853__jid47050__seed1.png) |
| 10 | `00954` | `35295` | 0 | 25.61% | 4.63% | [conditions_10__mid00954__jid35295__seed0.png](conditions_10__mid00954__jid35295__seed0.png) |
