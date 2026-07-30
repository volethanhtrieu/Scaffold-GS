# Scaffold-GS to SOGS Migration Plan

## Repository baseline

| Item | Value |
|---|---|
| Repository root | `/home/trieu_kernel/Desktop/Viettel_drone/Scaffold-GS` |
| Branch | `feature/sogs-integration` |
| Commit | `ddef7063f56201e1b4b046d504845c4f31ea382a` |
| Working tree | Dirty before this A100 pass: `README.md`, `docs/sogs_run_guide.md`, `docs/sogs_verification_report.md`, `test_sogs.py`, `train.py`, and `train_competition.py`; untracked `configs/sogs_one_hour.yaml`, `scripts/train_sogs_one_hour.sh`, and `utils/training_budget.py` |
| Dedicated branch | Already on `feature/sogs-integration`; existing user changes will be preserved |
| Training status | Training has not been and will not be executed during preparation |

The local repository contains competition-specific data validation and rendering
helpers in addition to the upstream Scaffold-GS code. Those interfaces remain in
place; SOGS is an opt-in extension. The current branch already contains the
core SOGS migration, while the dirty files contain an in-progress bounded
one-hour profile. No unofficial SOGS source tree or URL is present in this
workspace. Consequently, paper-specified equations take precedence and
implementation gaps are explicitly marked as inferences rather than attributed
to unavailable reference code.

The competition PDFs require exact output names/counts/resolutions, prohibit
external scene data and manual output editing, and may request source,
configuration, dependency versions, checkpoints, and logs for reproducibility.
The score weights LPIPS at 0.4, SSIM at 0.3, and normalized PSNR at 0.3. No
training-time or model-size term is stated. The round-one document gives a
20-GiB RTX A4000 as an inference reference, not a training limit.

## Migration components

| Component       | Current Scaffold-GS behavior | Required SOGS behavior    | Target files                    | Risk   | Verification           |
| --------------- | ---------------------------- | ------------------------- | ------------------------------- | ------ | ---------------------- |
| CLI arguments   | `feat_dim` is fixed through the generic parameter group; boolean options use `store_true` | Add explicit `use_second_order`, `num_eigenvectors`, `lambda_sgl`, and validated `feat_dim`; accept `True`/`False` safely | `arguments/__init__.py` | Low    | Parser smoke test |
| Anchor features | Each anchor stores `N x D` first-order features and the renderer consumes them directly | Center features, form a correlation matrix, use symmetric eigendecomposition, run one two-layer MLP per selected eigenvector, and concatenate `D + M*D` features | `utils/sogs_utils.py`, `scene/sogs.py`, `scene/gaussian_model.py` | High   | Tensor-shape and numerical tests |
| Renderer        | Opacity/covariance/color MLPs receive `D+3`, `D+4`, or `D+4+A` inputs | Obtain features only through `gaussians.get_render_features()`, derive every MLP input from `D*(1+M)`, materialize visible rows in checkpointed chunks, and mask attributes without a wide temporary concatenation | `gaussian_renderer/__init__.py`, `scene/gaussian_model.py`, `utils/sogs_utils.py` | High   | Forward-interface, dimension, and chunked-gradient tests |
| Loss            | L1, D-SSIM, and `0.01 *` volume regularization are used in `train.py` | Preserve those coefficients and add isolated Sobel selective-gradient loss weighted by `lambda_sgl` only for enabled SOGS | `utils/loss_utils.py`, `train.py` | High   | Synthetic image loss tests |
| Checkpoints     | PLY stores anchor tensors; split MLP TorchScript files and a legacy tuple checkpoint are used | Store SOGS configuration and second-order MLP state, validate metadata, restore all trainable modules, and reject incompatible/missing SOGS state | `scene/gaussian_model.py`, `scene/__init__.py`, `train.py`, `render.py`, `render_test_poses.py` | High   | Save/load round-trip and incompatibility tests |
| Configuration   | `cfg_args` records the parsed `Namespace`; scripts expose only baseline knobs | Persist all SOGS knobs in `cfg_args` and metadata; add baseline/default/compact config examples and opt-in scripts | `arguments/__init__.py`, `configs/`, `scripts/`, `train.sh`, `train_competition.py` | Medium | Configuration dump and shell lint |

## A100 optimization delta

The target reported by the user is one NVIDIA A100-SXM4-80GB (SM80). The
reported active process used about 2.4 GiB while showing 87% GPU utilization.
This telemetry is useful for profile design but is not a benchmark: the A100 is
not exposed in the present workspace.

| Change | Compatibility behavior | A100 intent | Target files | Risk | Verification |
|---|---|---|---|---|---|
| SOGS activation checkpoint switch | Keep checkpointing enabled by default | Disable recomputation when 80 GiB is available | `arguments/__init__.py`, `utils/sogs_utils.py`, `scene/gaussian_model.py`, `gaussian_renderer/__init__.py` | Medium | Forward/backward equivalence test |
| Larger visible-anchor chunks | Keep the existing conservative default | Feed larger matrices to the A100 and reduce Python/kernel-launch overhead | Config and launcher files | Medium | Shape test and later authorized benchmark |
| Evaluation feature cache | Disabled by default and only active in eval with gradients disabled | Reuse frozen augmented features over the 40--70 competition target views | `scene/gaussian_model.py` | Medium | Cache invalidation/value test |
| Single symmetric eigensolve | Preserve the same selected eigenvector path | Reuse eigenvalues from `eigh` instead of launching a redundant `eigvalsh` | `utils/sogs_utils.py` | Low | Numerical/gradient tests |
| Runtime math controls | FP32/legacy optimizer behavior remains the default | Opt in to TF32, cuDNN autotuning, and fused Adam when supported | `arguments/__init__.py`, `train.py`, `scene/gaussian_model.py` | Medium | Parser/runtime smoke tests; score A/B test required |
| Loss-kernel caching/fusion | Preserve the same Sobel and SSIM equations | Cache fixed kernels and evaluate prediction/target x/y Sobel maps in one grouped convolution | `utils/loss_utils.py` | Low | Synthetic image equality test |
| Reduced host synchronization | Preserve loss and optimizer math; only logging cadence changes when requested | Avoid per-iteration `.item()`/CUDA timing stalls | `train.py` | Low | Static test and later profiler trace |
| Separate profiles | Baseline and paper-default profiles remain available | Expose speed-first and quality-first A100 commands without claiming either wins | `configs/`, `scripts/`, documentation | Low | Dry-run command checks |

## Feature-dimension flow

The local `generate_neural_gaussians` path currently concatenates:

```text
base anchor feature                 D
view direction                      +3
distance (when enabled)             +1
appearance embedding (color only)  +A
```

With SOGS enabled, the model-owned feature path changes only the first line:

```text
base anchor feature                 D
selected eigenvector branches       M * D
render feature                      D * (1 + M)
opacity input                       D*(1+M) + 3 [+1]
covariance input                    D*(1+M) + 3 [+1]
color input                         D*(1+M) + 3 [+1] [+A]
```

When the optional local feature bank is enabled, its three view-adaptive
weights are applied after augmentation and the width is explicitly trimmed
back to `D*(1+M)` so odd compact dimensions remain valid.

The hidden width of the existing attribute MLPs remains the local base
`feat_dim` for compatibility. Each second-order branch is a two-layer
`Linear(2D, D) -> ReLU -> Linear(D, D)` module. The `M` eigenvectors are global
shared directions, while each branch output is `N x D`; therefore the stored
PLY feature remains `N x D` and the augmented dimension is computed at render
time.

## Implementation order

1. Add validated configuration parsing and a standalone second-order feature
   module.
2. Add the model-owned feature accessor and resize the three attribute MLP input
   layers without changing the baseline dimensions when SOGS is disabled.
3. Route the renderer through that accessor and use device/dtype-inherited
   temporary tensors.
4. Add the isolated selective-gradient loss and opt-in training term while
   preserving the existing L1, D-SSIM, and volume terms.
5. Extend checkpoint capture, split/united MLP serialization, PLY/config metadata,
   and all train/render constructors.
6. Add lightweight CPU synthetic tests, configuration examples, and verification
   documentation.

## Assumptions and Unresolved Details

| Decision | Source | Confidence | Alternative |
|---|---|---:|---|
| Use the paper's unbiased `1/(N-1)` covariance normalization, with a safe denominator for `N <= 1`, and form the correlation matrix before eigendecomposition | Paper Eq. (4)--(9) | High | Biased `1/N` covariance |
| Compute statistics over all anchors, then mask visible anchors for rendering | Paper says the anchor set is the scene-wide observation set; local renderer has a visibility mask | High | Compute statistics only over visible anchors (would make features view-dependent) |
| Each selected eigenvector is concatenated with the per-anchor feature and passed through an independent two-layer `2D -> D` MLP | Paper Eq. (10)--(11); MLP width is not otherwise specified | Medium | Shared MLP with an eigenvector-index embedding |
| Keep existing attribute-MLP hidden width at base `D` while increasing only the first input width | Local Scaffold-GS interface; paper does not prescribe hidden widths | Medium | Use augmented width for hidden layers too |
| Use `lambda_sgl=0.01`, `M=2` as reference defaults while leaving SOGS disabled by default | Paper implementation details | High | Tune per scene after authorized experiments |
| Keep dynamic SGL weight maps differentiable because the paper does not direct gradients to be blocked | Paper Eq. (15)--(16) plus compatibility requirement | Medium | Detach the weights and differentiate only the discrepancy term |
| Use zero-padding for the 3x3 Sobel convolution and mean-reduce the per-channel/batch weighted map for the scalar SGL | Paper specifies the kernels and scalar equations but not border or batch-reduction conventions | Medium | Valid convolution, channel sum, or a global batch sum |
| Legacy checkpoints without SOGS metadata are accepted only in baseline mode; SOGS mode fails clearly | Compatibility requirement and local checkpoint format | High | Automatic random initialization of missing second-order modules (not safe) |
| The existing PLY format stores only base features; second-order features are recomputed from them | Paper feature augmentation and local PLY serializer | High | Store augmented features (would make configuration-dependent PLY files) |
| Apply the optional view-adaptive feature-bank weights after second-order augmentation and preserve the full augmented width | Local feature-bank consumer; SOGS paper does not discuss this optional path | Medium | Apply the bank before augmentation or disable it in SOGS mode |
| Implement without copying an unofficial fork because no such source tree/URL is available in this workspace | Repository inspection and public project page | High | Compare against a user-supplied unofficial repository in a later review |
| Compute the existing XYZ volume term with explicit component-wise multiplication | Reported PyTorch 1.12.1/CUDA 11.6 NVRTC failure and the renderer's verified `K x 3` scaling shape | High | Upgrade PyTorch/CUDA and rebuild extensions before using the original `prod` reduction |
| Floor channel variances at `eps²` before taking their square root | Inference required for finite gradients from Scaffold-GS's zero-initialized anchor features; the paper does not specify zero-variance handling | High | Detach covariance eigenvectors, which would change the gradient path and is not specified by the paper |
| Compute scene-global statistics but checkpoint SOGS branches and renderer attribute-MLP activations over visible-anchor chunks | Inference for 24-GiB runtime constraints; preserves the paper's global statistics and differentiable feature path while reducing retained activations | Medium | Disable checkpointing for maximum throughput, or detach statistics for a larger but method-altering memory reduction |
| Map visible/selected offsets to global densification-statistics indices directly | Local Scaffold-GS mask semantics; allocation-only compatibility optimization | High | Build and clone full scene-wide boolean masks as in the original implementation |
| Disable SOGS activation checkpointing only in the A100 profiles | User-provided 80-GiB capacity plus PyTorch checkpoint semantics | Medium | Retain checkpointing if measured peak memory is unsafe |
| Cache full augmented anchor features only while the model is in eval mode and autograd is disabled | Inference from frozen inference state and multi-view competition rendering | High | Cache only eigenvectors, which saves less work but uses less memory |
| Offer TF32 as an opt-in speed setting, not a quality guarantee | Official PyTorch CUDA semantics for Ampere; competition metrics are sensitive to output changes | Medium | Full IEEE FP32 matmul in the quality-control run |
| Offer fused Adam only after checking that the installed PyTorch exposes it | Official PyTorch optimizer interface; local legacy environment is PyTorch 1.12.1 | Medium | Use PyTorch's default Adam implementation |
| Use `D=32, M=2, lambda_sgl=0.01` as the quality-first A100 candidate | Paper reports that quality rises with D and gains diminish beyond D=16; competition does not score model size | Medium | Paper-size `D=16` candidate, selected by controlled scene validation |
| Keep full input/output resolution, ten offsets, and 30,000 iterations in quality candidates | Competition requirements and SOGS/Scaffold-GS paper protocol | High | Change only after an authorized, controlled resource/quality experiment |
| Do not enable whole-model `torch.compile` or AMP in this pass | The renderer is a custom CUDA autograd extension, anchor tensors/optimizer state change during densification, and neither mixed-precision rasterization nor graph capture was verified | High | Add an isolated experiment after extension-level CUDA tests |

## Verification targets

No dataset or full training run is part of this migration. Verification will use
static compilation, parser checks, small CPU tensors, synthetic gradients, and
in-memory serialization checks where dependencies permit. Reported rendering
quality, speed, memory, and model-size claims require later authorized
experiments and will not be inferred from these checks.
