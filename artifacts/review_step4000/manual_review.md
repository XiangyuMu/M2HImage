# Step-4000 Fixed-Set Hair Review

Protocol hash: `40826872040bb9738a5b5f3b4e06b527f6d3e6a8e74d24db92326a144ec48517`

The comparison order is `[reference face+hair crop | raw generated image | h_i]`.
Mask-tinted watcher panels were not used to judge color.

| sample | visual finding |
|---|---|
| 00041 | Dreadlock topology transfers, but strands are shorter, noisier, and less coherent than the reference. |
| 00571 | Short curly silhouette transfers; curl and fringe details remain approximate. |
| 00093 | Brown wavy/partly raised silhouette transfers, but length and tie structure are softened. |
| 00621 | Short bowl-cut silhouette and dark color transfer clearly; fine fringe geometry is approximate. |

## Conclusion

`COARSE-ROUTE-ALIVE / EXACT-HAIR-UNRESOLVED`

All four checked samples respond to the intended reference hairstyle at a coarse structural level, consistent with
`hair_ref_swap_concentration ~= 1.945`. Exact length, strand structure, and color fidelity are not stable, consistent
with fixed-set Hair-DINO `0.4966` and LAB distance `34.22` at step 4000. Garment and head-pose preservation remain
