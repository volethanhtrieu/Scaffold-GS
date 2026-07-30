# SOGS Verification Report

## Repository Baseline

- Branch: `feature/sogs-integration`
- Commit before the A100 pass:
  `ddef7063f56201e1b4b046d504845c4f31ea382a`
- Dirty files before the A100 pass: `README.md`, `docs/sogs_run_guide.md`,
  `docs/sogs_verification_report.md`, `test_sogs.py`, `train.py`, and
  `train_competition.py`; untracked `configs/sogs_one_hour.yaml`,
  `scripts/train_sogs_one_hour.sh`, and `utils/training_budget.py`
- Safety: all pre-existing changes were retained; no reset, stash, checkout,
  commit, or submodule modification was performed

### NVRTC compatibility follow-up (2026-07-30)

- Branch before the fix: `feature/sogs-integration`
- Commit before the fix: `36a31b5f8f596469fa0ba743f3b80c049f02fe67`
- Dirty files before the fix: none
- Reported failure: PyTorch's CUDA `reduction_prod_kernel` stopped at iteration
  zero with `nvrtc: error: invalid value for --gpu-architecture (-arch)`.
- Resolution: preserve the original mean XYZ-volume objective using explicit
  component-wise multiplication, avoiding the legacy NVRTC Jiterator path.

### Zero-variance gradient follow-up (2026-07-30)

- Reported failure: SOGS rejected non-finite anchor features during the first
  few iterations.
- Reproduction: a synthetic backward pass from Scaffold-GS's zero-initialized
  anchor features produced NaN feature gradients before this fix.
- Root cause: `sqrt(variance)` was evaluated at zero. Its infinite derivative
  contaminated backward even though zero-variance entries were masked later.
- Resolution: clamp variance to `eps²` before `sqrt`, while retaining the
  explicit zero-variance correlation mask and identity diagonal.

### 24-GiB VRAM follow-up (2026-07-30)

- Reported failure: SOGS training exceeded the available GPU memory after
  second-order features were enabled.
- Resolution: keep covariance/eigendecomposition scene-global, augment only
  anchors selected by the renderer visibility mask, checkpoint branch MLP
  and renderer MLP activations in bounded chunks, and mask renderer attributes
  without constructing the former `N*n_offsets x 22` concatenation. Expose
  `sogs_chunk_size` through the launcher, `cfg_args`, and checkpoint metadata.
  The competition launcher also exposes `--n-offsets` for the renderer's
  largest per-anchor tensors. Densification statistics now map selected offsets
  directly instead of allocating and cloning scene-wide boolean masks, while
  duplicate checks reduce each chunk in-place instead of retaining every
  intermediate mask. The later A100 pass separates their tile size into
  `densification_chunk_size`.
- Runtime guidance: start with `--sogs-chunk-size 2048`; use `1024` if the
  first setting still reaches CUDA out-of-memory. This is a memory/performance
  control and does not change the augmented feature dimension.

Current working-tree follow-up baseline:

- Branch: `feature/sogs-integration`
- Commit before these VRAM edits: `707931c8` (zero-variance fix)
- Training: not executed

### A100-SXM4-80GB follow-up (2026-07-30)

- Target: the user reported one SM80 A100 with 80 GiB. The snapshot showed
  about 2.4 GiB allocated and 87% utilization by an existing process.
- Local limitation: this workspace exposes neither `nvidia-smi` nor a CUDA
  device, so A100 throughput, memory, rasterizer behavior, and image quality
  were not measured here.
- Resolution: add opt-in retained-activation, finite-check, frozen-render-cache,
  TF32, cuDNN autotuner, Adam-backend, densification-chunk, GUI, and logging
  controls. Reuse the existing symmetric `eigh` result instead of running a
  second eigenvalue-only solve, and fuse the four SGL Sobel launches into one
  value-equivalent grouped convolution. Keep all baseline defaults compatible.
- Profiles: add separate throughput (`D=16`, TF32/auto Adam) and
  quality-control (`D=32`, IEEE FP32/default Adam) launchers. Both retain full
  resolution, ten offsets, no unknown test-pose appearance embedding, `M=2`,
  SGL weight 0.01, and 30,000 iterations.
- Full-speed runner: add one wrapper for all seven scenes that performs the
  strict verifier, rejects competing compute PIDs, waits out small transient
  utilization, preserves fresh-output guards, runs scenes sequentially, and
  stores console logs outside model directories. It never kills a process or
  changes GPU clocks/power limits.
- Reproducibility: CLI training records the complete Namespace in `cfg_args`
  and writes the resolved PyTorch/CUDA/GPU/optimizer/runtime values to
  `runtime_config.json`.

### A100 ultra-speed follow-up (2026-07-30)

- Baseline: branch `feature/sogs-integration`, commit
  `d1476dca9b2fbac2fa31689e532e3d5dfbcc2aa6`; the only pre-existing dirty path
  was untracked `docs/runningScript.txt`, which was preserved.
- Resolution: add an explicit `ultra` profile to the existing A100 launchers
  without changing their default `throughput` path. Ultra uses
  `resolution=4`, `D=8`, `M=1`, five offsets, input ratio 2, no SGL, and
  densification through iteration 7,500. These are workload/quality changes,
  not method-equivalent runtime optimizations.
- Runner: `--profile throughput|ultra|quality` is validated before data/output
  inspection. Every profile has a separate default output/log root and a
  dynamic runtime summary; no arbitrary value is interpolated into a command
  or path.
- Verification: all three one-scene commands passed non-mutating synthetic
  dry-run tests. The real local seven-scene `data/` tree passed the ultra
  runner's all-scene dry-run; no GPU work or output/log creation occurred.
- Training status: no training or rendering was executed. The requested
  10--20 iteration/s range and all quality metrics remain unmeasured.

## Files Modified

| File | Reason | Main risk |
|---|---|---|
| `arguments/__init__.py` | Safe boolean parsing and validated SOGS settings | Invalid legacy config values must fail clearly |
| `utils/sogs_utils.py` | Correlation/eigendecomposition and learned feature branches | Eigenvector stability and per-render compute cost |
| `scene/sogs.py` | Compatibility re-export | None beyond import-time dependency behavior |
| `scene/gaussian_model.py` | Feature accessor, MLP dimensions, optimizer, checkpoint/metadata state | Architecture/checkpoint compatibility |
| `gaussian_renderer/__init__.py` | Single augmented-feature path and dimension assertions | Renderer input shape/per-view overhead |
| `utils/loss_utils.py` | Isolated selective gradient loss, cached windows, and value-equivalent grouped Sobel evaluation | Sobel border/reduction conventions |
| `utils/runtime_utils.py` | Explicit TF32/cuDNN runtime policy | Floating-point policy can change metrics |
| `utils/checkpoint_utils.py` | Preserve checkpoint behavior across legacy/current PyTorch | Version-specific checkpoint semantics |
| `train.py` | Opt-in SGL term and component logging | Training-only numerical/quality behavior |
| `render.py` | Reuse saved SOGS settings and safer device probing | Inference environment differences |
| `render_test_poses.py` | Reuse SOGS settings from `cfg_args` | Competition checkpoint compatibility |
| `train_competition.py`, `train.sh`, `single_train.sh` | Expose opt-in SOGS command settings | Shell/launcher argument propagation |
| `README.md`, `COMPETITION_PIPELINE.md` | Document modes, metadata, and reproducibility | Documentation drift |
| `configs/*.yaml`, `scripts/*.sh` | Reference profiles, opt-in launchers, and strict environment verification | Defaults are reference values, not tuned results |
| `test_sogs.py` | Small synthetic configuration/shape/numerical/compatibility tests | Full CUDA rasterizer is intentionally not exercised |
| `docs/scaffold_to_sogs_plan.md`, `docs/sogs_experiment_plan.md`, `docs/sogs_run_guide.md`, `docs/push_sogs_branch.md` | Analysis, run instructions, branch transfer, and later experiment commands | Experiments remain unexecuted |

The VRAM follow-up additionally modified `arguments/__init__.py`,
`utils/sogs_utils.py`, `scene/gaussian_model.py`,
`gaussian_renderer/__init__.py`, `train.py`, `render.py`,
`render_test_poses.py`, `train_competition.py`, `test_sogs.py`,
`configs/sogs_default.yaml`, `configs/sogs_compact.yaml`, `train.sh`,
`single_train.sh`, `scripts/train_sogs.sh`, `README.md`,
`COMPETITION_PIPELINE.md`, `docs/scaffold_to_sogs_plan.md`, and this report/run
guide. The main risk is a throughput reduction from activation recomputation;
the base feature and checkpoint architecture remain unchanged.

The NVRTC follow-up modified these files:

| File | Reason | Main risk |
|---|---|---|
| `train.py` | Route the existing volume term through the compatibility helper | Accidental loss change |
| `utils/loss_utils.py` | Compute `sx * sy * sz` without CUDA `Tensor.prod` JIT compilation | CUDA behavior must be checked on the primary GPU |
| `test_sogs.py` | Verify the exact scalar, gradient, shape contract, and training call site | CPU tests cannot reproduce the reported GPU architecture |
| `scripts/verify_sogs_environment.sh` | Add a tiny CUDA forward/backward smoke test for the patched loss path | Requires the primary GPU environment |
| `docs/sogs_run_guide.md` | Document the error signature, preflight, and environment fallback | GPU-specific support still depends on the installed stack |
| `docs/scaffold_to_sogs_plan.md`, `docs/sogs_verification_report.md` | Record the compatibility rationale and verification status | None |

The zero-variance follow-up modified `utils/sogs_utils.py`, `test_sogs.py`, and
these two reports. Its dedicated regression test starts with an all-zero
`N x D` parameter, applies a non-uniform downstream target, runs backward
through the eigendecomposition path, and requires finite feature and MLP
gradients.

The one-hour follow-up adds `utils/training_budget.py`, bounded-run arguments
in `train.py` and `train_competition.py`, the compact
`scripts/train_sogs_one_hour.sh` launcher, and
`configs/sogs_one_hour.yaml`. On deadline, the loop saves both renderable model
files and a structured optimizer checkpoint. This behavior is an operational
inference for time-constrained runs, not part of the SOGS paper.

The A100 follow-up modifies the core/configuration/renderer/loss files above,
adds `utils/runtime_utils.py`, `utils/checkpoint_utils.py`,
`configs/sogs_a100_throughput.yaml`, `configs/sogs_a100_quality.yaml`, and
`scripts/train_sogs_a100.sh`, later adds
`scripts/run_sogs_a100_max_speed.sh`, and updates the reports/run guide. Its
main risks are unmeasured A100 peak memory and possible score changes from
TF32/fused Adam; both are isolated to explicit profile settings.

The ultra-speed follow-up adds `configs/sogs_a100_ultra.yaml`, extends both
A100 launchers and their static/dry-run tests, registers the config in the
environment verifier, and updates `README.md` plus the three SOGS planning/run
documents. It does not change core model, renderer, loss, or baseline defaults.

## Static Checks

| Check | Result | Notes |
|---|---|---|
| `python -m py_compile train.py` | Passed | Active base Python environment |
| `python -m py_compile render.py` | Passed | Active base Python environment |
| `python -m py_compile scene/gaussian_model.py` | Passed | Active base Python environment |
| `python -m py_compile` on all modified Python files | Passed | Includes tests and SOGS utility modules |
| `bash -n` on modified launcher scripts | Passed | No launcher was executed |
| `git diff --check` | Passed | No whitespace errors in the migration diff |
| Parser smoke test (`True`, `False`, `--use_second_order`) | Passed | Safe explicit and bare-flag forms |
| `python test_sogs.py` | Passed (53; 33 skipped) | Base environment has no PyTorch; parser/static/runtime-profile and all A100 non-mutating launcher tests ran |
| `conda run -n depth_anything python test_sogs.py` | Passed (53 tests) | CPU PyTorch 2.12.1+cu130; includes all three A100 profile dry-runs, retained/checkpointed gradient equivalence, cache invalidation, optimizer/config/checkpoint round trips, and cached loss kernels |
| Grouped Sobel numerical-equivalence regression | Passed | Uses float32 tolerance because PyTorch 1.12 may accumulate grouped and separate convolutions in a different order; the loss implementation is unchanged |
| PyTorch 2.12 runtime-policy probe | Passed | Explicit `fp32_precision` API resolved to `tf32`/`ieee` as requested without mixing legacy controls |
| Adam `auto` construction probe | Passed | CPU fallback resolved to `foreach`; A100 fused execution remains unmeasured |
| 30-step zero-initialized SOGS optimizer probe | Passed | Features and gradients remained finite for every synthetic step |
| `python -m py_compile train.py render.py scene/gaussian_model.py utils/sogs_utils.py utils/loss_utils.py test_sogs.py` | Passed | NVRTC and zero-variance compatibility follow-ups |
| `bash -n scripts/verify_sogs_environment.sh` | Passed | CUDA preflight script syntax only |
| `python train_competition.py --help` | Passed | Exposes memory, runtime-budget, and checkpoint-interval controls without starting training |
| `bash -n scripts/train_sogs_one_hour.sh` | Passed | The bounded launcher was not executed |
| `bash -n scripts/train_sogs_a100.sh` | Passed | Launcher syntax only |
| A100 throughput/ultra/quality launcher dry-runs | Passed | All resolved commands printed their unique settings exactly once; no output directory or training was created |
| `scripts/run_sogs_a100_max_speed.sh --profile ultra --dry-run` | Passed | Resolved the real local seven-scene data tree; GPU checks, output/log creation, and training remained disabled |
| A100 runner error-path dry-runs | Passed | Missing/unknown profiles fail before filesystem or GPU mutation; default remains throughput |
| One-scene A100 `train_competition.py --dry-run` | Passed | HCM0421 validated (240 train images, 60 poses); all new flags propagated and no training started |
| Seven-scene `train_competition.py --dry-run` with chunk size 2048 | Passed | Validated all scenes, propagated the memory setting, and created no output directory |
| `bash -n train.sh single_train.sh scripts/train_sogs.sh` | Passed | Chunk-size propagation syntax only |
| CUDA volume/SOGS-backward preflight | Not run here | This workspace has no NVIDIA runtime; run the verifier on the primary GPU |
| `scripts/verify_sogs_environment.sh --gpu 0 --data-root data` | Correctly failed strict environment check; data passed | Active base environment has no required PyTorch/CUDA stack; NVIDIA CLI tools are absent and all seven scenes passed |
| `python train.py --help` | Blocked by environment | Active base environment has no NumPy/PyTorch stack |
| `conda run -n depth_anything python train.py --help` | Blocked by environment | That environment lacks `einops` and the compiled renderer dependencies |

## Tensor Shapes

| Tensor | Scaffold-GS shape | SOGS shape | Verified |
|---|---|---|---|
| Stored anchor feature | `N x D` | `N x D` | Yes |
| Centered/covariance statistics | Not used | covariance/correlation `D x D` | Yes (synthetic CPU tensors) |
| Selected eigenvectors | Not used | `D x M` | Yes (synthetic CPU tensors) |
| Each learned branch | Not used | `N x D` | Yes |
| Render feature | `N x D` | `N x D*(1+M)` | Yes |
| Cached eval feature | Not needed | `N x D*(1+M)` only in no-grad eval | Yes (synthetic CPU cache/invalidation test) |
| Opacity input (no distance) | `N x (D+3)` | `N x (D*(1+M)+3)` | Yes |
| Covariance input (no distance) | `N x (D+3)` | `N x (D*(1+M)+3)` | Yes |
| Color input (no distance/appearance) | `N x (D+3)` | `N x (D*(1+M)+3)` | Yes |
| Selective-gradient image input | `CHW`/`BCHW` accepted | Same | Yes (synthetic CPU tensors) |

## Checkpoint Compatibility

- Old Scaffold-GS split MLP/PLY checkpoints without SOGS metadata remain
  loadable only when `use_second_order=False`.
- Loading an old checkpoint in SOGS mode stops with a missing-metadata/module
  error; it does not initialize SOGS modules randomly.
- New checkpoints include `sogs_config.json`, one
  `second_order_mlp_<index>.pt` per selected eigenvector, and structured
  in-memory capture state containing SOGS MLP and optimizer parameters.
- Strict module-state loading and configuration comparisons report missing,
  unexpected, or incompatible state.
- New PLY files continue to store base `N x D` features; augmented features are
  recomputed from the saved configuration.
- Runtime-only SOGS controls are recorded but may change at load time because
  they do not alter learned tensor shapes; architecture and SGL settings remain
  strict.

## Training Status

Training was not executed in this environment.

## Remaining Risks

- `torch.linalg.eigh` and scene-wide statistics add per-render computation and
  may need profiling on the primary GPU.
- Eigenvector gradients can be sensitive when correlation eigenvalues are nearly
  repeated; a deterministic diagonal tie-breaker is documented as an inference.
- The exact hidden width/activation convention of the paper's two-layer
  branch MLP is not specified; this implementation keeps the local base width
  `D`.
- The paper does not specify Sobel border handling or batch/channel reduction;
  the isolated loss uses zero padding and a mean reduction, documented as an
  assumption in the migration plan.
- Full differentiable rasterizer, competition-resolution rendering, and
  organizer submission validation were not run.
- Retained activations, an 8,192-row densification tile, fused/foreach Adam,
  and the full eval feature cache have not been profiled on the reported A100;
  lower the independent chunk controls if the measured peak is unsafe.
- TF32 and fused/foreach Adam can change floating-point ordering. The
  throughput profile must be scored against the IEEE-FP32 quality control
  before it is used for a submission.
- The ultra profile processes about 1/16 as many training pixels and reduces
  model/anchor capacity and the loss. It may be much faster, but it can
  materially reduce detail, PSNR, SSIM, LPIPS, and competition score; neither
  10--20 iteration/s nor acceptable quality is guaranteed without an A100
  probe.
- The compact one-hour profile has not been benchmarked on the primary RTX
  4090; its wall-clock stop is deterministic, but completed iterations and
  resulting quality remain scene-dependent.
- No PSNR, SSIM, LPIPS, memory, FPS, model-size, or competition-score claim is
  made without authorized experiments.
