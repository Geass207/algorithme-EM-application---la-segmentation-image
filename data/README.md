# Data

The MRI volume is not versioned. The experiments use a simulated T1-weighted
brain volume from [BrainWeb](https://brainweb.bic.mni.mcgill.ca/) (181 × 217 × 181
voxels) and a binary brain mask of the same shape.

Place the two files here:

```
data/brain.nii.gz      T1-weighted volume (NIfTI)
data/brain_mask.nii    binary brain mask, 1 inside the brain
```

Any skull-stripped T1 volume with its mask works; pass other paths with
`--image` and `--mask`.
