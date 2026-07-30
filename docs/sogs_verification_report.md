# SOGS Verification Report

## Repository Baseline

- Branch: `main` (the recommended `feature/sogs-integration` branch could not be
  created because `.git` is read-only in this environment)
- Commit: `9718569d385c618f551242402a26a7db34259c56`
- Dirty files: migration files listed below; no user changes were present at
  baseline

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

## Files Modified

| File | Reason | Main risk |
|---|---|---|
| `arguments/__init__.py` | Safe boolean parsing and validated SOGS settings | Invalid legacy config values must fail clearly |
| `utils/sogs_utils.py` | Correlation/eigendecomposition and learned feature branches | Eigenvector stability and per-render compute cost |
| `scene/sogs.py` | Compatibility re-export | None beyond import-time dependency behavior |
| `scene/gaussian_model.py` | Feature accessor, MLP dimensions, optimizer, checkpoint/metadata state | Architecture/checkpoint compatibility |
| `gaussian_renderer/__init__.py` | Single augmented-feature path and dimension assertions | Renderer input shape/per-view overhead |
| `utils/loss_utils.py` | Isolated selective gradient loss | Sobel border/reduction conventions |
| `train.py` | Opt-in SGL term and component logging | Training-only numerical/quality behavior |
| `render.py` | Reuse saved SOGS settings and safer device probing | Inference environment differences |
| `render_test_poses.py` | Reuse SOGS settings from `cfg_args` | Competition checkpoint compatibility |
| `train_competition.py`, `train.sh`, `single_train.sh` | Expose opt-in SOGS command settings | Shell/launcher argument propagation |
| `README.md`, `COMPETITION_PIPELINE.md` | Document modes, metadata, and reproducibility | Documentation drift |
| `configs/*.yaml`, `scripts/*.sh` | Reference profiles, opt-in launchers, and strict environment verification | Defaults are reference values, not tuned results |
| `test_sogs.py` | Small synthetic configuration/shape/numerical/compatibility tests | Full CUDA rasterizer is intentionally not exercised |
| `docs/scaffold_to_sogs_plan.md`, `docs/sogs_experiment_plan.md`, `docs/sogs_run_guide.md`, `docs/push_sogs_branch.md` | Analysis, run instructions, branch transfer, and later experiment commands | Experiments remain unexecuted |

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
| `python test_sogs.py` | Passed (29; 22 skipped) | Base environment has no PyTorch; parser/static tests ran |
| `conda run -n depth_anything python test_sogs.py` | Passed (29 tests) | CPU PyTorch; includes volume-loss and zero-initialized SOGS gradient checks |
| 30-step zero-initialized SOGS optimizer probe | Passed | Features and gradients remained finite for every synthetic step |
| `python -m py_compile train.py render.py scene/gaussian_model.py utils/sogs_utils.py utils/loss_utils.py test_sogs.py` | Passed | NVRTC and zero-variance compatibility follow-ups |
| `bash -n scripts/verify_sogs_environment.sh` | Passed | CUDA preflight script syntax only |
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
- No PSNR, SSIM, LPIPS, memory, FPS, model-size, or competition-score claim is
  made without authorized experiments.
