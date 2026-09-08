# Code snapshot — 2026-09-08

This commit backs up the current M2HImage source code, configuration files,
tests, project documentation, and experiment scripts from the research server.
The server working tree and its existing staging area were not modified.

## Included

- Current training, inference, preprocessing, metrics, spatial conditioning,
  identity supervision, and baseline integration code.
- Objective A-D0/B diagnostics, input-protocol revalidation, A2 localization,
  and the new C-H / C-perm / E-match short-training scripts under `artifacts/`.
- The frozen 256-source-group / 512-pair state-pilot protocol (IDs and hashes,
  not images) and its explicit monitor hold status.

## Excluded

Datasets, input/generated images, cached tensors, model weights, checkpoints,
runtime logs, machine-local agent state, credentials, patch leftovers, and
third-party baseline repository checkouts. Paths in scripts and configuration
files still describe the original research environment and may need adapting.
This is a source backup, not a self-contained dataset/model distribution.

## Current experiment status

The three-arm state pilot is **not complete** and must not be restarted solely
because this snapshot exists. The server crashed during health/preparation.
C-H has 11 readable optimizer-update log records, but no new pilot checkpoint
was preserved. C-perm and E-match have not started. All 800 input/supervision
cache hashes survived. Of 29 saved teacher latent endpoints, all verified;
27 accompanying PNGs verified and two were zero-byte files.

Subsequently supplied kernel logs identify two fatal machine-check exceptions
followed by `Kernel panic - not syncing: Fatal machine check`, reported from
Socket 1 and Socket 0, both Bank 6. The specific faulty hardware/firmware
component has not been identified. Training remains on hold pending diagnosis.
Older experiment reports are historical artifacts, not evidence that this
pilot completed or that the server is now stable.

## Validation boundary

One server source file, `artifacts/m2h_a2_localization_20260908/scripts/recover_localization.py`,
was found to contain 6,173 zero bytes. The snapshot uses its intact local
authoring copy (same filename and byte length; Python syntax valid), rather
than publishing the corrupt file as executable source. The server original
was not changed, and a binary evidence copy was retained locally outside Git.
The cause and exact time of this source-file corruption are not established.
The intact copy SHA256 is
`ee3e82b41611a17ee51f388d4fa5e65e9d710d2f435d87e722e1d74fc6eeb893`;
the corrupt server-copy SHA256 is
`ffb0829688283901de0735fc69284f0fc765cc10f5151aa0a255eedadcb54f91`.

Source syntax and upload safety are checked for this backup. Nine targeted
state-pilot CPU/mock tests passed before the incident. Full GPU training,
checkpoint recovery, and the three-arm development evaluation are not claimed
to have passed. No dependency installation or training is part of this backup.
