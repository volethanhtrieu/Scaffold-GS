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

## A100 selection protocol

The A100-SXM4-80GB removes the 24-GiB memory constraint, but it does not make a
single profile simultaneously fastest and highest-scoring. Use two stages once
training is explicitly authorized:

1. Run a short throughput-only probe on one public scene with identical model
   settings, changing only checkpointing/chunking, TF32, and Adam backend.
   Record iterations/second and peak allocated/reserved memory. Do not use this
   short run to choose image quality.
2. Run full controlled candidates A--D plus the two A100 candidates below on a
   fixed organizer-provided validation split. Select by the official aggregate
   score, using runtime only as a tie-breaker or deadline constraint.

Prepared A100 commands:

```bash
DRY_RUN=1 scripts/train_sogs_a100.sh throughput \
  "$SCENE_PATH" "$RUN_ROOT/E_a100_throughput"

DRY_RUN=1 scripts/train_sogs_a100.sh quality \
  "$SCENE_PATH" "$RUN_ROOT/F_a100_quality"
```

Remove `DRY_RUN=1` only for an authorized run. Candidate E uses `D=16`, TF32,
automatic fused/foreach Adam, retained activations, and reduced finite/logging
synchronization. Candidate F uses `D=32`, IEEE FP32, the original Adam backend,
retained activations, and finite checks. Both retain full resolution, `M=2`,
ten offsets, SGL weight 0.01, and 30,000 iterations.

For a fair runtime ablation of E, compare these one at a time while keeping
`D=16` fixed:

- checkpointing on versus off;
- `tf32_mode=disabled` versus `enabled`;
- `optimizer_backend=default` versus `auto`;
- finite checks on versus off.

The requested/resolved runtime, PyTorch/CUDA versions, GPU model/capability, and
memory controls are written to `runtime_config.json`; the complete CLI
Namespace is written to `cfg_args`. Preserve both beside the checkpoint and
training log.
