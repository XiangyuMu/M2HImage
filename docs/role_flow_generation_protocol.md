# Role Flow Generation Protocol v2

`tools/generate_role_flow_eval.py` writes `generation_provenance.json` in every role-flow generation output directory. The file is the audit boundary for generated PNGs and is required for resuming partial output.

## Input Signature

The protocol computes `input_signature_sha256` from:

- config path and SHA-256;
- checkpoint path, type, recursive file SHA-256 records, and checkpoint tree SHA-256;
- evaluation manifest path, file SHA-256, and manifest `content_sha256`;
- selected split, limit, row keys, and exact expected PNG filenames;
- `eval.generate_steps` and `data.resolution`;
- actual generation code provenance.

If an output directory already contains PNGs, normal generation resumes only when the existing `generation_provenance.json` has the same `input_signature_sha256`. Directories with PNGs and no provenance fail closed.

## Code Provenance

`generation_provenance.json` separates the code that produced PNGs from the tool that finalized the audit:

- `code.generation` describes the actual generator.
- `code.finalize_tool` describes the current `tools/generate_role_flow_eval.py` process that wrote or rewrote provenance.

For live generation, `code.generation` uses the current script SHA-256 and git commit with `capture_mode: live` and `source: current_tool`.

For retrospective imports, pass the actual generator identity explicitly:

```bash
python tools/generate_role_flow_eval.py \
  --checkpoint /path/to/checkpoints/final \
  --config configs/role_selective/A_timestep_routed.yaml \
  --manifest /path/to/eval_manifest.json \
  --output-dir /path/to/generated/A_timestep_routed \
  --finalize-only \
  --generation-git-commit 95e015e4568b0ec4cf6862204e8506f34c1d7366 \
  --generation-script-sha256 ACTUAL_OLD_GENERATE_SCRIPT_SHA256
```

When either generation override is provided, `code.generation` is recorded with `capture_mode: retrospective` and `source: operator_supplied`. This prevents a later finalize-only audit from claiming that the current script produced older PNGs.

## Output Validation

The expected filename for each selected row is:

```text
{mid}__id{jid}__seed{seed}.png
```

Final validation requires:

- the number of readable PNGs equals the selected row count;
- every expected filename exists;
- no extra PNG exists in the output directory;
- every PNG can be opened;
- every PNG is RGB;
- every PNG size matches `data.resolution`;
- every PNG records SHA-256, width, height, mode, readability, row key, and status.

Failed validation writes `generation_provenance.json` with `status: failed` and the failure list before exiting non-zero. Provenance writes use an atomic temporary-file replace, including the final `status: complete` record.

## Finalize Existing Outputs

Use `--finalize-only` to audit an already completed directory without loading FLUX:

```bash
python tools/generate_role_flow_eval.py \
  --checkpoint /path/to/checkpoints/final \
  --config configs/role_selective/A_timestep_routed.yaml \
  --manifest /path/to/eval_manifest.json \
  --output-dir /path/to/generated/A_timestep_routed \
  --finalize-only
```

`--finalize-only` is intentionally strict. It requires a complete output directory and rejects missing, unreadable, or extra PNGs.
