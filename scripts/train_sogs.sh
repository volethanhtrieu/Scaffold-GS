#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <scene-source-path> <model-output-path> [extra train.py args...]" >&2
    exit 2
fi

scene_path=$1
model_path=$2
shift 2
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "${script_dir}/.." && pwd)
python_command=${PYTHON_COMMAND:-python}

# REFERENCE: D=16, M=2, lambda_sgl=.01 are the SOGS paper defaults.
command=(
    "${python_command}"
    "${repository_root}/train.py"
    --source_path "${scene_path}"
    --model_path "${model_path}"
    --use_second_order True
    --feat_dim 16
    --num_eigenvectors 2
    --lambda_sgl 0.01
    "$@"
)

printf 'Resolved command:'
printf ' %q' "${command[@]}"
printf '\n'
"${command[@]}"

