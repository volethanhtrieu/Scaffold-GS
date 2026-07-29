# Scaffold-GS versus SOGS Experiment Plan

This document prepares later experiments; none of the commands below were
executed during repository migration. Replace the placeholders only after
checking the primary machine's GPU memory, each scene's image resolution, and
the current competition round requirements.

## Controlled variables

Keep these values identical across A--D unless the row explicitly changes one:

- organizer-provided dataset and train/test split
- image resolution and camera intrinsics
- 30,000 iterations (or the same authorized budget)
- random seed
- voxel size and anchor-growing schedule
- evaluation cameras/test poses
- background configuration
- appearance embedding configuration
- number of offsets per anchor

Only organizer-provided scene data may be used. Generated predictions must keep
the exact scene names, image names, image counts, widths, and heights requested
by `test_poses.csv`.

## Prepared commands

Set shared paths before running any experiment:

```bash
SCENE_PATH=/absolute/path/to/scene/train
RUN_ROOT=/absolute/path/to/experiment/output
COMMON_ARGS="--eval --iterations 30000 --voxel_size 0.001 --update_init_factor 16 --appearance_dim 0 --ratio 1"
```

The string above is illustrative; when actually running, prefer the argument
arrays in `scripts/` so paths remain safely quoted.

### A. Original Scaffold-GS

```bash
python train.py -s "$SCENE_PATH" -m "$RUN_ROOT/A_scaffold" \
  --use_second_order False --feat_dim 32 --num_eigenvectors 2 \
  --lambda_sgl 0 $COMMON_ARGS
```

### B. SOGS with the same base feature dimension

```bash
python train.py -s "$SCENE_PATH" -m "$RUN_ROOT/B_sogs_d32" \
  --use_second_order True --feat_dim 32 --num_eigenvectors 2 \
  --lambda_sgl 0.01 $COMMON_ARGS
```

### C. SOGS with reduced base feature dimension

```bash
python train.py -s "$SCENE_PATH" -m "$RUN_ROOT/C_sogs_d16" \
  --use_second_order True --feat_dim 16 --num_eigenvectors 2 \
  --lambda_sgl 0.01 $COMMON_ARGS
```

After the 16-dimensional profile is validated, the paper's 12-dimensional
compact profile can be measured as a separate sub-experiment rather than
silently replacing C.

### D. SOGS without selective gradient loss

```bash
python train.py -s "$SCENE_PATH" -m "$RUN_ROOT/D_sogs_no_sgl" \
  --use_second_order True --feat_dim 16 --num_eigenvectors 2 \
  --lambda_sgl 0 $COMMON_ARGS
```

## Measurements

Record per scene and aggregate:

| Run | PSNR | SSIM | LPIPS | Training time | Rendering FPS | Peak GPU memory | Checkpoint size | Anchors | Neural Gaussians |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | | | | | | | | | |
| B | | | | | | | | | |
| C | | | | | | | | | |
| D | | | | | | | | | |

Use the competition score only after computing the organizer-specified
normalization:

```text
score = 0.4 * (1 - LPIPS) + 0.3 * SSIM + 0.3 * PSNR_norm
```

Do not select a final configuration from paper claims alone. Inspect actual
quality, peak memory, runtime, checkpoint size, scene resolution, and the
competition's reproducibility requirements first; retain source, configs,
dependency versions, checkpoints, and training logs for every candidate.
