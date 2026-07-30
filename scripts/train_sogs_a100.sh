#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <throughput|ultra|quality> <scene-source-path> <model-output-path> [extra train.py args...]" >&2
    exit 2
fi

profile=$1
scene_path=$2
model_path=$3
shift 3

case "${profile}" in
    throughput|ultra|quality) ;;
    *)
        echo "Unknown profile: ${profile}; expected throughput, ultra, or quality." >&2
        exit 2
        ;;
esac

if [[ ! -d "${scene_path}" ]]; then
    echo "Scene path does not exist: ${scene_path}" >&2
    exit 2
fi

if [[ -d "${model_path}" ]] && [[ -n "$(find "${model_path}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Output directory is not empty: ${model_path}" >&2
    echo "Use a fresh path, or resume explicitly with train.py and --start_checkpoint." >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "${script_dir}/.." && pwd)
python_command=${PYTHON_COMMAND:-python}
gpu_id=${GPU_ID:-0}

common_args=(
    --source_path "${scene_path}"
    --model_path "${model_path}"
    --gpu "${gpu_id}"
    --use_second_order True
    # COMPATIBILITY: competition CSV poses have no learned camera ID.
    --appearance_dim 0
    --iterations 30000
    --test_iterations 1000000000
    --save_iterations 30000
    --cudnn_benchmark True
    --disable_gui
    --skip_postprocess
)

case "${profile}" in
    throughput)
        # INFERENCE: paper-size D=16 plus TF32/automatic fused Adam prioritizes
        # examples per second. Numerical equivalence is not assumed.
        profile_default_log_interval=50
        profile_args=(
            --feat_dim 16
            --num_eigenvectors 2
            --lambda_sgl 0.01
            --n_offsets 10
            --resolution 1
            --ratio 1
            --voxel_size 0.001
            --update_until 15000
            --sogs_chunk_size 65536
            --sogs_checkpointing False
            --sogs_validate_numerics False
            --sogs_cache_render_features True
            --densification_chunk_size 8192
            --tf32_mode enabled
            --optimizer_backend auto
        )
        ;;
    ultra)
        # INFERENCE: this is a workload-reduced speed ablation. Quarter-width
        # and quarter-height training images, fewer offsets, smaller SOGS
        # features, and earlier densification can materially reduce quality.
        profile_default_log_interval=1000
        profile_args=(
            --feat_dim 8
            --num_eigenvectors 1
            --lambda_sgl 0
            --n_offsets 5
            --resolution 4
            --ratio 2
            --voxel_size 0.001
            --update_until 7500
            --sogs_chunk_size 262144
            --sogs_checkpointing False
            --sogs_validate_numerics False
            --sogs_cache_render_features True
            --densification_chunk_size 16384
            --tf32_mode enabled
            --optimizer_backend auto
        )
        ;;
    quality)
        # INFERENCE: D=32 and IEEE FP32 form the quality-control candidate.
        profile_default_log_interval=50
        profile_args=(
            --feat_dim 32
            --num_eigenvectors 2
            --lambda_sgl 0.01
            --n_offsets 10
            --resolution 1
            --ratio 1
            --voxel_size 0.001
            --update_until 15000
            --sogs_chunk_size 65536
            --sogs_checkpointing False
            --sogs_validate_numerics True
            --sogs_cache_render_features True
            --densification_chunk_size 8192
            --tf32_mode disabled
            --optimizer_backend default
        )
        ;;
    *)
        echo "Unknown profile: ${profile}; expected throughput, ultra, or quality." >&2
        exit 2
        ;;
esac

log_interval=${LOG_INTERVAL:-${profile_default_log_interval}}
if [[ ! "${log_interval}" =~ ^[1-9][0-9]*$ ]]; then
    echo "LOG_INTERVAL must be a positive integer: ${log_interval}" >&2
    exit 2
fi
profile_args+=(--log_interval "${log_interval}")

command=(
    "${python_command}"
    "${repository_root}/train.py"
    "${common_args[@]}"
    "${profile_args[@]}"
    "$@"
)

printf 'Resolved command:'
printf ' %q' "${command[@]}"
printf '\n'
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1; training was not started."
    exit 0
fi
"${command[@]}"
