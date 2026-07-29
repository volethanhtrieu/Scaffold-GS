#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <model-path> [extra render.py args...]" >&2
    exit 2
fi

model_path=$1
shift
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "${script_dir}/.." && pwd)
python_command=${PYTHON_COMMAND:-python}

# render.py reads the saved cfg_args and sogs_config.json metadata, so it
# reuses the exact training architecture instead of restating SOGS dimensions.
command=(
    "${python_command}"
    "${repository_root}/render.py"
    --model_path "${model_path}"
    "$@"
)

printf 'Resolved command:'
printf ' %q' "${command[@]}"
printf '\n'
"${command[@]}"
