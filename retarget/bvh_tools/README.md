# BVH processing tools

These utilities parse BVH channels and transforms, validate the repository-owned Geno skeleton and stance, and resample motion on a physical-time grid with zero-phase anti-aliasing.

The canonical stance is `data/sources/lafan1-resolved/Geno_stance.bvh`. Source samples remain unchanged. Native UMR optimization is implemented separately under `third_party/umr`; these utilities handle input validation and explicit postprocessing.
