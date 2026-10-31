The two Python files are copied from the user's YY retargeter checkout
(`scripts/bvh_adapters.py` and `scripts/motion_postprocess.py`, revision in
`.reference/yy-retargeter`). The Geno stance path alone is adapted to this
repository's `data/sources/lafan1-resolved/Geno_stance.bvh`.

`retarget/processing.py` also copies YY's `motion_refinement.py`
physical-time second-derivative coefficients into its focused per-joint QP.
No X2 model constants or correspondence slots are used. The production entry uses the actual native UMR NR surface optimizer under
`third_party/umr`, with newly trained Geno/OpenSprite correspondence.
