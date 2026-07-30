# Seven-Scene SOGS Training and Submission Guide

This is the primary runbook for training SOGS on all seven supplied scenes,
rendering the requested test poses, and creating the competition submission.
Run the numbered sections in order from the repository root.

The seven scenes used by every command in this guide are:

```text
HCM0421
HCM0539
HCM0540
HCM0644
HCM0674
bonsai
chair
```

The workflow uses these repository-relative locations:

```text
data/<scene>/train/                 training images and COLMAP model
data/<scene>/test/test_poses.csv    requested competition poses
outputs/round1/<scene>/             trained SOGS models
predictions/round1/<scene>/         rendered test-pose images
submission_round1.zip               final validated archive
```

Training is sequential on one GPU. Nothing in this guide starts training until
the command in Step 5 is explicitly run.

## 1. Enter the repository and activate the environment

```bash
cd /home/trieu_kernel/Desktop/Viettel_drone/Scaffold-GS
conda activate scaffold_gs
```

If the `scaffold_gs` environment does not exist yet, create it once from the
repository root, then activate it:

```bash
conda env create -f environment.yml
conda activate scaffold_gs
```

Do not recreate an already working environment. `environment.yml` contains
the original Scaffold-GS Python, PyTorch, CUDA, Python packages, and local CUDA
extension requirements.

## 2. Verify the environment and code without training

Run the strict verifier on GPU 0 and include the seven-scene data audit:

```bash
scripts/verify_sogs_environment.sh \
  --gpu 0 \
  --data-root data
```

The verifier checks the required repository files, GPU driver, required Python
packages, a small CUDA tensor operation, both compiled Scaffold-GS extensions,
Python and shell syntax, the SOGS synthetic tests, and the seven input scenes.
It never starts training.

Do not continue if it reports any `[FAIL]`. A skipped tensor test means the
active environment is not suitable for training. Fix dependency or extension
errors and rerun the verifier until its final line says:

```text
Environment verification PASSED. No training was started.
```

## 3. Confirm the seven-scene audit

The data is already in the expected layout; do not move it within the
repository. Confirm that the verifier checked:

```text
HCM0421 HCM0539 HCM0540 HCM0644 HCM0674 bonsai chair
```

The final line must report:

```text
Result: 7/7 scene(s) valid.
```

The top-level `data` directory is passed to the competition launcher. The
launcher passes each scene's `data/<scene>/train` directory to `train.py`.

## 4. Preview the seven SOGS training commands

Run the dry run first. It validates all scenes and prints the seven resolved
`train.py` commands, but does not create models or start GPU training:

```bash
python train_competition.py \
  --data-root data \
  --output-root outputs/round1 \
  --scenes \
    HCM0421 \
    HCM0539 \
    HCM0540 \
    HCM0644 \
    HCM0674 \
    bonsai \
    chair \
  --gpu 0 \
  --use-second-order True \
  --feat-dim 16 \
  --num-eigenvectors 2 \
  --lambda-sgl 0.01 \
  --sogs-chunk-size 2048 \
  --iterations 30000 \
  --dry-run
```

Verify that the preview contains seven commands, uses the intended physical
GPU, and maps every scene to:

```text
source: data/<scene>/train
model:  outputs/round1/<scene>
```

The selected `D=16`, `M=2`, and `lambda_sgl=0.01` values are SOGS
reference-style settings, not guaranteed competition-optimal settings.

## 5. Train SOGS on all seven scenes

This is the command that starts the long GPU workload. Run it only after Steps
1-4 pass:

```bash
python train_competition.py \
  --data-root data \
  --output-root outputs/round1 \
  --scenes \
    HCM0421 \
    HCM0539 \
    HCM0540 \
    HCM0644 \
    HCM0674 \
    bonsai \
    chair \
  --gpu 0 \
  --use-second-order True \
  --feat-dim 16 \
  --num-eigenvectors 2 \
  --lambda-sgl 0.01 \
  --sogs-chunk-size 2048 \
  --iterations 30000
```

The launcher trains one scene at a time on GPU 0 and stops on the first
failure. It does not train all seven simultaneously, so it does not multiply
GPU memory usage by seven. For a fresh run, `outputs/round1` may be absent or
empty. The launcher intentionally refuses to reuse a non-empty per-scene model
directory.

After successful completion, the layout will be:

```text
outputs/round1/
├── HCM0421/
├── HCM0539/
├── HCM0540/
├── HCM0644/
├── HCM0674/
├── bonsai/
└── chair/
```

Each scene should contain `cfg_args`, `runtime_config.json`, `outputs.log`, and
the final checkpoint:

```text
outputs/round1/<scene>/point_cloud/iteration_30000/
├── point_cloud.ply
├── opacity_mlp.pt
├── cov_mlp.pt
├── color_mlp.pt
├── second_order_mlp_0.pt
├── second_order_mlp_1.pt
└── sogs_config.json
```

List the seven final point clouds:

```bash
find outputs/round1 \
  -path '*/point_cloud/iteration_30000/point_cloud.ply' \
  -type f \
  -print \
  | sort
```

Do not continue to rendering unless this prints one final point cloud for each
of the seven scenes.

## 6. Render all requested competition poses

Use `render_test_poses.py`, not the ordinary train/test-view `render.py`, for
the competition CSV poses:

```bash
python render_test_poses.py \
  --data-root data \
  --model-root outputs/round1 \
  --predictions-root predictions/round1 \
  --scenes \
    HCM0421 \
    HCM0539 \
    HCM0540 \
    HCM0644 \
    HCM0674 \
    bonsai \
    chair \
  --iteration -1 \
  --gpu 0
```

`--iteration -1` selects the latest saved iteration. The renderer reads
`cfg_args` and `sogs_config.json` from every trained model, restores both
second-order MLPs, and rejects incompatible checkpoint settings.

Successful rendering produces:

```text
predictions/round1/
├── HCM0421/*.png
├── HCM0539/*.png
├── HCM0540/*.png
├── HCM0644/*.png
├── HCM0674/*.png
├── bonsai/*.png
└── chair/*.png
```

The renderer uses the black-background setting used by the training command in
Step 5. If `--white-background` is added to training, it must also be added to
this rendering command.

## 7. Validate predictions and create the submission

The following command checks every expected filename and image dimension before
atomically creating the ZIP:

```bash
python generate_submission.py \
  --data-root data \
  --predictions-root predictions/round1 \
  --output submission_round1.zip \
  --scenes \
    HCM0421 \
    HCM0539 \
    HCM0540 \
    HCM0644 \
    HCM0674 \
    bonsai \
    chair \
  --filename-mode png
```

On success, the command reports the number of images, confirms all seven
scenes, and prints the archive size. The archive members use:

```text
<scene>/<image_name stem>.png
```

Run the standard-library ZIP integrity check:

```bash
python -m zipfile -t submission_round1.zip
```

The file to upload is:

```text
submission_round1.zip
```

### Optional size-limited JPEG archive

Use this alternative only if the active competition portal requires literal
CSV filenames and enforces a 350 MB limit:

```bash
python generate_submission.py \
  --data-root data \
  --predictions-root predictions/round1 \
  --output submission_round1_jpeg.zip \
  --scenes \
    HCM0421 \
    HCM0539 \
    HCM0540 \
    HCM0644 \
    HCM0674 \
    bonsai \
    chair \
  --filename-mode image-name \
  --transcode-jpeg \
  --jpeg-quality 92 \
  --jpeg-subsampling 420 \
  --max-size-mb 350
```

JPEG transcoding requires Pillow. Do not resize predictions. If quality 92 is
too large, lower `--jpeg-quality` gradually; if it is comfortably below the
limit, test a higher value to preserve more image quality.

## 8. Handling interruptions or existing output

The full command is intended for a fresh `outputs/round1` run. It refuses to
train into a non-empty `outputs/round1/<scene>` directory to avoid mixing
checkpoints.

If training stops after some scenes are complete, keep those completed
directories and rerun the launcher with `--scenes` containing only scenes that
have not started. For example:

```bash
python train_competition.py \
  --data-root data \
  --output-root outputs/round1 \
  --scenes bonsai chair \
  --gpu 0 \
  --use-second-order True \
  --feat-dim 16 \
  --num-eigenvectors 2 \
  --lambda-sgl 0.01 \
  --iterations 30000
```

If a failed scene already has a partial non-empty directory, inspect its
`outputs.log` first. Prefer a new clean output directory for retrying that
scene. Use `--allow-existing-output` only when intentionally continuing with
the risk that files in that directory may be overwritten; it does not
automatically resume optimizer state.

Do not delete or overwrite a completed model merely to rerun the launcher.

## 9. SOGS configuration reference

The training parameters used above mean:

| Launcher option | `train.py` option | Meaning |
|---|---|---|
| `--use-second-order True` | `--use_second_order True` | Enable SOGS augmentation |
| `--feat-dim 16` | `--feat_dim 16` | Base anchor width `D=16` |
| `--num-eigenvectors 2` | `--num_eigenvectors 2` | Select `M=2` directions |
| `--lambda-sgl 0.01` | `--lambda_sgl 0.01` | Selective-gradient-loss weight |
| `--sogs-checkpointing False` | `--sogs_checkpointing False` | Retain activations instead of recomputing them |
| `--sogs-cache-render-features True` | `--sogs_cache_render_features True` | Cache frozen augmented features across eval cameras |
| `--tf32-mode enabled` | `--tf32_mode enabled` | Opt into Ampere TF32 matmul/convolution |
| `--optimizer-backend auto` | `--optimizer_backend auto` | Select fused, then foreach, when supported |
| `--densification-chunk-size 8192` | `--densification_chunk_size 8192` | Tile the anchor duplicate check independently |
| `--log-interval 50` | `--log_interval 50` | Synchronize scalar logs every 50 iterations |

The render feature width is:

```text
D * (1 + M) = 16 * (1 + 2) = 48
```

The PLY keeps the base 16-channel anchor feature. The two second-order
branches are saved separately and restored during rendering.

Reference profiles are documented in:

- [`configs/scaffold_gs_baseline.yaml`](../configs/scaffold_gs_baseline.yaml)
- [`configs/sogs_default.yaml`](../configs/sogs_default.yaml)
- [`configs/sogs_compact.yaml`](../configs/sogs_compact.yaml)
- [`configs/sogs_one_hour.yaml`](../configs/sogs_one_hour.yaml)
- [`configs/sogs_a100_throughput.yaml`](../configs/sogs_a100_throughput.yaml)
- [`configs/sogs_a100_ultra.yaml`](../configs/sogs_a100_ultra.yaml)
- [`configs/sogs_a100_quality.yaml`](../configs/sogs_a100_quality.yaml)

The YAML files document profiles; `train_competition.py` receives the settings
through command-line arguments and stores the resolved configuration in each
scene's `cfg_args`.

## 9A. A100-SXM4-80GB profiles

The user-provided `nvidia-smi` output reports one SM80 A100 with 80 GiB.
CUDA `13.0` in that output is the maximum CUDA version supported by the
driver; the PyTorch CUDA runtime and the toolkit used to compile the two local
extensions must still match each other. Do not rebuild the extensions merely
because the driver reports a newer version.

The seven-scene runner defaults to the existing full-resolution `throughput`
profile. Select `ultra` explicitly for the workload-reduced speed ablation:

```bash
# Preview the unchanged default without touching the GPU or outputs.
scripts/run_sogs_a100_max_speed.sh --dry-run

# Preview the aggressive profile without touching the GPU or outputs.
scripts/run_sogs_a100_max_speed.sh --profile ultra --dry-run

# Train all seven scenes after verification and the idle-GPU check pass.
scripts/run_sogs_a100_max_speed.sh --profile ultra
```

The launcher refuses a busy GPU or non-empty per-scene output directory; it
does not kill processes or delete results. Profile-specific default roots keep
ultra models and logs under `outputs/a100_ultra_speed/` and
`logs/a100_ultra_speed/`.

First verify that no unrelated process is consuming the GPU. The supplied
snapshot showed PID 8129 at 87% utilization, so a second training process would
compete with it. Then inspect either command without launching training:

```bash
DRY_RUN=1 scripts/train_sogs_a100.sh throughput \
  data/HCM0421/train \
  outputs/a100_throughput/HCM0421

DRY_RUN=1 scripts/train_sogs_a100.sh ultra \
  data/HCM0421/train \
  outputs/a100_ultra_speed/HCM0421

DRY_RUN=1 scripts/train_sogs_a100.sh quality \
  data/HCM0421/train \
  outputs/a100_quality/HCM0421
```

The launcher requires a new or empty output path. Remove `DRY_RUN=1` only when
training is intentional.

| Setting | Throughput | Ultra speed | Quality control |
|---|---:|---:|---:|
| Base feature `D` | 16 | 8 | 32 |
| Selected directions `M` | 2 | 1 | 2 |
| Render width `D*(1+M)` | 48 | 16 | 96 |
| SGL weight | 0.01 | 0 | 0.01 |
| Offsets | 10 | 5 | 10 |
| Training pixel scale | Full | 1/4 width and height | Full |
| Initial point ratio | 1 | 2 (take every second point) | 1 |
| Anchor growth ends | 15,000 | 7,500 | 15,000 |
| SOGS/densification chunks | 65,536 / 8,192 | 262,144 / 16,384 | 65,536 / 8,192 |
| Appearance embedding | Disabled (`A=0`) | Disabled (`A=0`) | Disabled (`A=0`) |
| Activation checkpointing | Off | Off | Off |
| Full finite reductions | Off | Off | On |
| Eval feature cache | On | On | On |
| TF32 | Enabled | Enabled | Disabled |
| Adam backend | Auto | Auto | Original default |
| Log interval | 250 in seven-scene runner | 1,000 | 250 |
| Iterations | 30,000 | 30,000 | 30,000 |

Disabling checkpointing is the principal 80-GiB optimization: it avoids
recomputing every SOGS and attribute-MLP activation during backward. The
throughput candidate also avoids full-tensor finite checks, writes synchronized
logs every 50 iterations when launched directly (250 in the seven-scene
runner), enables cuDNN autotuning, and opts into TF32. Those changes should
improve throughput but can change floating-point rounding; they must be
compared against the FP32 control using the competition metrics.

`sogs_cache_render_features=True` is active only in eval mode under
`torch.no_grad()`. It materializes the frozen `N x D*(1+M)` tensor once and
reuses it for the 40--70 target cameras. Returning to training invalidates the
cache.

Ultra is the only prepared profile intended to move a 1.66 iteration/s job
toward the requested 10--20 iteration/s range. Its `resolution=4` setting
processes approximately 1/16 as many training pixels, while the other capacity
reductions lower anchor/MLP work. This is a plausible target, not a guarantee:
scene visibility and anchor growth still change iteration cost. At exactly
1.66 iteration/s, 30,000 iterations take about 5.02 hours per scene and 35.1
hours for seven scenes; 10 and 20 iteration/s would take about 5.83 and 2.92
hours total, respectively, before dataset loading and checkpoint-saving
overhead.

Ultra is not mathematically or quality equivalent to the full-resolution
profiles. It may reduce fine detail, PSNR, SSIM, and the competition score.
`render_test_poses.py` still uses every CSV camera's requested output width and
height, so submission image dimensions remain correct; lower-resolution
training can nevertheless produce softer full-size renders. Benchmark one
scene before committing to all seven:

```bash
# Actual training: one scene only.
scripts/run_sogs_a100_max_speed.sh \
  --profile ultra \
  --output-root outputs/a100_ultra_probe \
  --log-root logs/a100_ultra_probe \
  HCM0421
```

Read the measured iteration rate from the progress display/log and inspect
GPU utilization in another terminal:

```bash
nvidia-smi dmon -s pucm -d 1
```

If the probe is fast enough and its validation renders are acceptable, use a
fresh output root for all seven scenes:

```bash
scripts/run_sogs_a100_max_speed.sh \
  --profile ultra \
  --output-root outputs/a100_ultra_all \
  --log-root logs/a100_ultra_all
```

## 9B. From-scratch one-hour fallback

The following launcher targets one scene and one 24-GiB GPU:

```bash
scripts/train_sogs_one_hour.sh \
  data/HCM0421/train \
  outputs/one_hour/HCM0421
```

The output path must be new or empty. The profile uses `D=12`, `M=1`, five
offsets, `ratio=4`, a 0.002 voxel size, and a 3,000-iteration ceiling. It keeps
the original image resolution so later competition rendering is not silently
reduced. Selective gradient loss is disabled because this is a speed-oriented
SOGS ablation rather than the paper-default objective.

`--max_runtime_minutes 55` reserves approximately five minutes of the
one-hour allocation for serialization. The exact completed iteration depends
on the scene. At the deadline, training finishes the active optimizer step and
saves both:

```text
point_cloud/iteration_<N>/point_cloud.ply
chkpnt<N>.pth
```

The first path is renderable and the second contains optimizer/module state for
later resumption. Additional resumable checkpoints are written every 500
iterations. No quality or score equivalence with a 30,000-iteration SOGS run
is implied.

To resume the same compact configuration after an interruption, keep the
existing output directory and provide one of its checkpoints:

```bash
START_CHECKPOINT=outputs/one_hour/HCM0421/chkpnt1500.pth \
scripts/train_sogs_one_hour.sh \
  data/HCM0421/train \
  outputs/one_hour/HCM0421
```

Without `START_CHECKPOINT`, the launcher rejects a non-empty output directory
to protect previous results.

The general competition launcher also accepts:

```text
--max-runtime-minutes <minutes>
--checkpoint-interval <iterations>
```

The runtime value is applied independently to every selected scene. Use
`--scenes` with exactly one name when the budget is intended for one scene.

## 10. Troubleshooting

### `nvrtc: error: invalid value for --gpu-architecture (-arch)`

The long CUDA source dump followed by this message means the legacy
PyTorch/CUDA runtime tried to JIT-compile an operator for a compute capability
that its bundled NVRTC does not recognize.  In this repository,
`environment.yml` pins PyTorch 1.12.1 with CUDA 11.6, so this is especially
likely on a newer GPU.

The training volume term now uses explicit multiplication of the three XYZ
scale components instead of PyTorch's NVRTC-backed `prod` reduction.  Run the
small preflight again before retrying:

```bash
scripts/verify_sogs_environment.sh --gpu 0 --data-root data
```

It must print both `CUDA volume regularization: OK` and
`CUDA zero-variance SOGS backward: OK`. If another operator later reports the
same NVRTC error, record the GPU name and compute capability:

```bash
python -c "import torch; p=torch.cuda.get_device_properties(0); print(torch.__version__, torch.version.cuda, p.name, (p.major, p.minor), torch.cuda.get_arch_list())"
```

That recurrence requires a PyTorch/CUDA build which supports the reported GPU,
followed by rebuilding both local CUDA extensions against the same
environment.  Do not copy extension binaries from a different PyTorch/CUDA
environment.

### CUDA or compiled-module import failure

Training and rendering require the differentiable rasterizer and `simple-knn`
compiled against compatible PyTorch and CUDA versions. Rebuild the local
extensions if their imports fail. A static test passing in a different CPU
environment does not prove the GPU training environment is ready.

### Out of GPU memory

The conservative SOGS path computes scene-global statistics, materializes
augmented features only for visible anchors, and checkpoint-recomputes the SOGS
branches and attribute-MLP activations in bounded chunks.
`--sogs-chunk-size` controls those activation chunks.
`--densification-chunk-size` separately controls the duplicate-check tile,
whose temporary tensor has different scaling. Smaller values use less peak
VRAM but can reduce throughput. For a 24-GiB card, retry a fresh scene output
with:

```bash
python train_competition.py \
  --data-root data \
  --output-root outputs/round1 \
  --scenes HCM0421 HCM0539 HCM0540 HCM0644 HCM0674 bonsai chair \
  --gpu 0 \
  --use-second-order True \
  --feat-dim 16 \
  --num-eigenvectors 2 \
  --lambda-sgl 0.01 \
  --sogs-chunk-size 2048 \
  --densification-chunk-size 1024 \
  --iterations 30000
```

If the first scene still reports CUDA out-of-memory, stop it and retry with
`--sogs-chunk-size 1024`. If that remains too large, use the documented compact
profile (`feat_dim=12`, `num_eigenvectors=1`, `n_offsets=5`) or add
`--n-offsets 5` to the launcher command; those choices change model capacity
and must be compared experimentally.
Move the incomplete per-scene output aside or choose a new output root before
retrying, because the launcher protects non-empty directories. Do not assume
that a lower-dimensional profile is competition-optimal without measuring
quality. The prepared comparison procedure is in
[`docs/sogs_experiment_plan.md`](sogs_experiment_plan.md).

### Old or incompatible checkpoint

Do not load an old Scaffold-GS checkpoint in SOGS mode and do not bypass strict
loading with `strict=False`. Train a SOGS checkpoint with the chosen
configuration or render the old checkpoint with `use_second_order=False`.

### Missing or wrong-size submission image

`generate_submission.py` reports the exact missing file or dimension mismatch.
Fix the corresponding scene render and rerun the submission command. It does
not silently omit invalid images.

## 11. Scope and result claims

This migration did not execute full training, rendering, or submission
generation. The commands above are prepared for the primary GPU machine. No
claim is made about PSNR, SSIM, LPIPS, FPS, peak GPU memory, archive size, or
competition score until an authorized experiment is actually completed.

For controlled Scaffold-GS/SOGS comparisons, read
[`docs/sogs_experiment_plan.md`](sogs_experiment_plan.md).

## 12. Repository status at migration time

The current A100 pass started on `feature/sogs-integration` at
`ddef7063f56201e1b4b046d504845c4f31ea382a`. The working tree already contained
the one-hour/runtime-budget changes listed in the verification report; they
were preserved. No commit was created by this preparation pass.

For the exact branch, commit, push, training-machine clone, environment setup,
and separate data-transfer sequence, follow
[`docs/push_sogs_branch.md`](push_sogs_branch.md).
