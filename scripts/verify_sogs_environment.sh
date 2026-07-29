#!/usr/bin/env bash
set -uo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/verify_sogs_environment.sh [--gpu <physical-id>] [--data-root <path>]

Checks the active Python environment for Scaffold-GS/SOGS training without
starting training. CUDA is required because this verifier targets the training
machine.

Options:
  --gpu <id>         Physical GPU to expose during the smoke test (default: 0).
                     Use -1 to keep the current CUDA_VISIBLE_DEVICES setting.
  --data-root <path> Also validate the seven competition scenes at this path.
  -h, --help         Show this help.
EOF
}

selected_gpu=0
data_root=""

while (($# > 0)); do
    case "$1" in
        --gpu)
            if (($# < 2)); then
                echo "ERROR: --gpu requires a value" >&2
                usage >&2
                exit 2
            fi
            selected_gpu=$2
            shift 2
            ;;
        --data-root)
            if (($# < 2)); then
                echo "ERROR: --data-root requires a path" >&2
                usage >&2
                exit 2
            fi
            data_root=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "${script_dir}/.." && pwd)
cd "${repository_root}" || exit 1

failure_count=0
warning_count=0

pass_check() {
    printf '[PASS] %s\n' "$1"
}

fail_check() {
    printf '[FAIL] %s\n' "$1" >&2
    failure_count=$((failure_count + 1))
}

warn_check() {
    printf '[WARN] %s\n' "$1" >&2
    warning_count=$((warning_count + 1))
}

section() {
    printf '\n== %s ==\n' "$1"
}

section "Repository"
printf 'Root: %s\n' "${repository_root}"

required_paths=(
    environment.yml
    train.py
    train_competition.py
    render_test_poses.py
    generate_submission.py
    test_sogs.py
    utils/sogs_utils.py
    submodules/diff-gaussian-rasterization/setup.py
    submodules/simple-knn/setup.py
)

missing_paths=()
for required_path in "${required_paths[@]}"; do
    if [[ ! -e "${required_path}" ]]; then
        missing_paths+=("${required_path}")
    fi
done

if ((${#missing_paths[@]} == 0)); then
    pass_check "Required repository files are present"
else
    printf 'Missing repository paths:\n' >&2
    printf '  %s\n' "${missing_paths[@]}" >&2
    fail_check "Required repository files are missing"
fi

if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    printf 'Branch: %s\n' "$(git branch --show-current 2>/dev/null || true)"
    printf 'Commit: %s\n' "$(git rev-parse HEAD 2>/dev/null || true)"
    pass_check "Git worktree detected"
else
    warn_check "Git metadata is unavailable; runtime checks can still continue"
fi

printf 'Filesystem capacity:\n'
df -h "${repository_root}" | tail -n 1

section "GPU driver"
if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi \
        --query-gpu=index,name,driver_version,memory.total \
        --format=csv,noheader; then
        pass_check "nvidia-smi can query the GPU driver"
    else
        warn_check "nvidia-smi exists but cannot query a GPU; the PyTorch CUDA operation remains authoritative"
    fi
else
    warn_check "nvidia-smi is unavailable; the PyTorch CUDA operation remains authoritative"
fi

if command -v nvcc >/dev/null 2>&1; then
    nvcc --version | tail -n 1
    pass_check "nvcc is available for rebuilding CUDA extensions"
else
    warn_check "nvcc is unavailable; training can run only if compatible extensions are already installed"
fi

section "Python, packages, CUDA, and compiled extensions"
if ! command -v python >/dev/null 2>&1; then
    fail_check "python is not available on PATH"
else
    printf 'Python command: %s\n' "$(command -v python)"
    printf 'Conda environment: %s\n' "${CONDA_DEFAULT_ENV:-<not active>}"

    if [[ "${selected_gpu}" != "-1" ]]; then
        export CUDA_VISIBLE_DEVICES="${selected_gpu}"
        printf 'Exposed physical GPU: %s\n' "${selected_gpu}"
    else
        printf 'Keeping CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-<unset>}"
    fi

    if python - <<'PY'
from __future__ import print_function

import importlib
import os
import platform
import sys
import traceback

required_modules = (
    ("numpy", "NumPy"),
    ("torch", "PyTorch"),
    ("torchvision", "torchvision"),
    ("PIL", "Pillow"),
    ("tqdm", "tqdm"),
    ("einops", "einops"),
    ("plyfile", "plyfile"),
    ("torch_scatter", "torch-scatter"),
    ("cv2", "OpenCV"),
    ("colorama", "colorama"),
    ("jaxtyping", "jaxtyping"),
)
optional_modules = (
    ("laspy", "laspy (optional point-cloud formats)"),
    ("wandb", "Weights & Biases (optional logging)"),
    ("lpips", "LPIPS (optional metric evaluation)"),
    ("tensorboard", "TensorBoard (optional logging)"),
)

print("Executable:", sys.executable)
print("Python:", platform.python_version())

missing = []
for module_name, label in required_modules:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "version unavailable")
        print("Required import OK: {} ({})".format(label, version))
    except Exception as error:
        missing.append("{}: {}".format(label, error))

for module_name, label in optional_modules:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "version unavailable")
        print("Optional import OK: {} ({})".format(label, version))
    except Exception as error:
        print("Optional import missing: {} ({})".format(label, error))

if missing:
    print("Missing required imports:", file=sys.stderr)
    for message in missing:
        print("  " + message, file=sys.stderr)
    sys.exit(1)

import torch
from torch_scatter import scatter_max

print("PyTorch CUDA build:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))

if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False", file=sys.stderr)
    sys.exit(1)

try:
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    total_gib = properties.total_memory / float(1024 ** 3)
    print("Selected GPU:", properties.name)
    print("Compute capability: {}.{}".format(properties.major, properties.minor))
    print("GPU memory: {:.2f} GiB".format(total_gib))

    left = torch.randn((32, 32), device=device)
    right = torch.randn((32, 32), device=device)
    product = left @ right
    if not bool(torch.isfinite(product).all().item()):
        raise RuntimeError("the CUDA matrix result is non-finite")
    torch.cuda.synchronize()
    print("Small CUDA tensor operation: OK")
except Exception:
    traceback.print_exc()
    sys.exit(1)

try:
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
    from simple_knn._C import distCUDA2

    assert GaussianRasterizationSettings is not None
    assert GaussianRasterizer is not None
    assert distCUDA2 is not None
    assert scatter_max is not None

    extension_points = torch.randn((16, 3), device=device)
    extension_distances = distCUDA2(extension_points)
    if extension_distances.shape[0] != extension_points.shape[0]:
        raise RuntimeError("simple-knn returned an unexpected output shape")
    if not bool(torch.isfinite(extension_distances).all().item()):
        raise RuntimeError("simple-knn returned non-finite distances")
    torch.cuda.synchronize()

    print("Compiled rasterizer import: OK")
    print("Compiled simple-knn CUDA smoke test: OK")
except Exception:
    traceback.print_exc()
    sys.exit(1)
PY
    then
        pass_check "Python packages, CUDA execution, and compiled extensions"
    else
        fail_check "Python/CUDA dependency verification"
    fi
fi

section "Static and synthetic checks"
python_sources=(
    arguments/__init__.py
    scene/gaussian_model.py
    scene/sogs.py
    utils/sogs_utils.py
    utils/loss_utils.py
    gaussian_renderer/__init__.py
    train.py
    render.py
    render_test_poses.py
    train_competition.py
    generate_submission.py
)

if python -m py_compile "${python_sources[@]}"; then
    pass_check "Python syntax compilation"
else
    fail_check "Python syntax compilation"
fi

shell_sources=(
    train.sh
    single_train.sh
    scripts/train_scaffold_baseline.sh
    scripts/train_sogs.sh
    scripts/render_sogs.sh
    scripts/verify_sogs_environment.sh
)

if bash -n "${shell_sources[@]}"; then
    pass_check "Shell syntax"
else
    fail_check "Shell syntax"
fi

if python test_sogs.py; then
    pass_check "SOGS synthetic test suite"
else
    fail_check "SOGS synthetic test suite"
fi

if [[ -n "${data_root}" ]]; then
    section "Seven-scene data audit"
    if python prepare_data.py \
        --data-root "${data_root}" \
        --scenes \
            HCM0421 \
            HCM0539 \
            HCM0540 \
            HCM0644 \
            HCM0674 \
            bonsai \
            chair; then
        pass_check "All seven competition scenes"
    else
        fail_check "Seven-scene data audit"
    fi
else
    section "Seven-scene data audit"
    warn_check "Skipped; rerun with --data-root data after transferring the dataset"
fi

section "Summary"
printf 'Failures: %d\n' "${failure_count}"
printf 'Warnings: %d\n' "${warning_count}"

if ((failure_count > 0)); then
    echo "Environment verification FAILED. Do not start training."
    exit 1
fi

echo "Environment verification PASSED. No training was started."
