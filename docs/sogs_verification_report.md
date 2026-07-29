# SOGS Verification Report

## Repository Baseline

- Branch: `main` (the recommended `feature/sogs-integration` branch could not be
  created because `.git` is read-only in this environment)
- Commit: `9718569d385c618f551242402a26a7db34259c56`
- Dirty files: migration files listed below; no user changes were present at
  baseline

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
| `python test_sogs.py` | Passed (25; 19 skipped) | Base environment has no PyTorch; parser/static tests ran |
| `conda run -n depth_anything python test_sogs.py` | Passed (25 tests) | CPU PyTorch; unused CUDA/PLY imports were stubbed |
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
