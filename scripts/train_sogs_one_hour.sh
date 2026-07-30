#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <scene-train-path> <model-output-path> [extra train.py args...]" >&2
    exit 2
fi

scene_path=$1
model_path=$2
shift 2
start_checkpoint=${START_CHECKPOINT:-}

if [[ ! -d "${scene_path}" ]]; then
    echo "Scene path does not exist: ${scene_path}" >&2
    exit 2
fi

if [[ -n "${start_checkpoint}" ]] && [[ ! -f "${start_checkpoint}" ]]; then
    echo "Checkpoint does not exist: ${start_checkpoint}" >&2
    exit 2
fi

if [[ -z "${start_checkpoint}" ]] && [[ -d "${model_path}" ]] && [[ -n "$(find "${model_path}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Output directory is not empty: ${model_path}" >&2
    echo "Choose a fresh output path for this from-scratch run." >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "${script_dir}/.." && pwd)
python_command=${PYTHON_COMMAND:-python}
gpu_id=${GPU_ID:-0}
runtime_minutes=${MAX_RUNTIME_MINUTES:-55}

# INFERENCE: this compact SOGS ablation targets a useful model inside a
# one-hour wall-clock window on a 24-GiB RTX 4090. It keeps full image
# resolution, but reduces anchors, offsets, feature width, and SOGS branches.
# Selective gradient loss is disabled because its full-resolution Sobel passes
# were expensive and dominated the reported objective in the initial run.
command=(
    "${python_command}"
    "${repository_root}/train.py"
    --source_path "${scene_path}"
    --model_path "${model_path}"
    --gpu "${gpu_id}"
    --use_second_order True
    --feat_dim 12
    --num_eigenvectors 1
    --lambda_sgl 0
    --sogs_chunk_size 4096
    --sogs_checkpointing True
    --sogs_validate_numerics True
    --sogs_cache_render_features False
    --densification_chunk_size 1024
    --n_offsets 5
    --voxel_size 0.002
    --ratio 4
    --appearance_dim 0
    --iterations 3000
    --test_iterations 1000000000
    --save_iterations 3000
    --checkpoint_interval 500
    --max_runtime_minutes "${runtime_minutes}"
    --start_stat 100
    --update_from 500
    --update_interval 100
    --update_until 2500
    --position_lr_max_steps 3000
    --offset_lr_max_steps 3000
    --mlp_opacity_lr_max_steps 3000
    --mlp_cov_lr_max_steps 3000
    --mlp_color_lr_max_steps 3000
    --mlp_featurebank_lr_max_steps 3000
    --appearance_lr_max_steps 3000
    --skip_postprocess
)

if [[ -n "${start_checkpoint}" ]]; then
    command+=(--start_checkpoint "${start_checkpoint}")
fi
command+=("$@")

printf 'Resolved command:'
printf ' %q' "${command[@]}"
printf '\n'
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1; training was not started."
    exit 0
fi
"${command[@]}"
