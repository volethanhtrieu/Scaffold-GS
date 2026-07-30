#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/run_sogs_a100_max_speed.sh [options] [scene ...]

Train the seven competition scenes sequentially with a selected A100 profile.
When no scene names are supplied, the defaults are:

  HCM0421 HCM0539 HCM0540 HCM0644 HCM0674 bonsai chair

Options:
  --data-root PATH       Scene root (default: data)
  --profile PROFILE      throughput, ultra, or quality (default: throughput)
  --output-root PATH     Model root (default: profile-specific outputs/a100_*)
  --log-root PATH        Console-log root (default: profile-specific logs/a100_*)
  --gpu ID               Physical GPU index (default: 0)
  --idle-timeout SEC     Wait for transient GPU activity (default: 60)
  --skip-verify          Skip the non-training environment/data verifier
  --dry-run              Print every command; do not touch the GPU or outputs
  -h, --help             Show this help

Examples:
  # Preview all seven commands without training:
  scripts/run_sogs_a100_max_speed.sh --dry-run

  # Start all seven scenes, one after another:
  scripts/run_sogs_a100_max_speed.sh

  # Aggressive speed/quality tradeoff; preview before running:
  scripts/run_sogs_a100_max_speed.sh --profile ultra --dry-run

  # Train only two scenes:
  scripts/run_sogs_a100_max_speed.sh bonsai chair

The launcher never kills GPU processes and never overwrites a non-empty model
directory. If the selected GPU is busy, it reports the competing PIDs and exits.
EOF
}

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

make_absolute() {
    local value=$1
    if [[ "${value}" == /* ]]; then
        printf '%s\n' "${value}"
    else
        printf '%s/%s\n' "${repository_root}" "${value}"
    fi
}

gpu_process_rows() {
    nvidia-smi \
        --id="${physical_gpu}" \
        --query-compute-apps=pid,process_name,used_gpu_memory \
        --format=csv,noheader,nounits
}

assert_no_compute_processes() {
    local rows
    if ! rows=$(gpu_process_rows 2>/dev/null); then
        fail "could not query compute processes on GPU ${physical_gpu}"
    fi
    if printf '%s\n' "${rows}" | awk -F, '$1 ~ /^[[:space:]]*[0-9]+[[:space:]]*$/ {found=1} END {exit !found}'; then
        printf 'GPU %s already has a compute process:\n%s\n' \
            "${physical_gpu}" "${rows}" >&2
        printf 'Stop that job intentionally, then rerun this launcher. No process was killed.\n' >&2
        exit 1
    fi
}

wait_for_idle_gpu() {
    local deadline=$((SECONDS + idle_timeout_seconds))
    local sample utilization memory_used

    while true; do
        assert_no_compute_processes
        sample=$(
            nvidia-smi \
                --id="${physical_gpu}" \
                --query-gpu=utilization.gpu,memory.used \
                --format=csv,noheader,nounits
        )
        IFS=',' read -r utilization memory_used <<<"${sample}"
        utilization=${utilization//[[:space:]]/}
        memory_used=${memory_used//[[:space:]]/}

        [[ "${utilization}" =~ ^[0-9]+$ ]] \
            || fail "could not parse GPU utilization from: ${sample}"
        [[ "${memory_used}" =~ ^[0-9]+$ ]] \
            || fail "could not parse GPU memory usage from: ${sample}"

        # INFERENCE: a process-free GPU below these small transient thresholds
        # is sufficiently idle; P0 is entered automatically when training starts.
        if ((utilization <= 5 && memory_used <= 1024)); then
            printf 'GPU %s is idle: utilization=%s%%, memory=%s MiB.\n' \
                "${physical_gpu}" "${utilization}" "${memory_used}"
            return
        fi
        if ((SECONDS >= deadline)); then
            fail "GPU ${physical_gpu} did not become idle within ${idle_timeout_seconds}s (utilization=${utilization}%, memory=${memory_used} MiB)"
        fi
        printf 'Waiting for transient GPU activity: utilization=%s%%, memory=%s MiB...\n' \
            "${utilization}" "${memory_used}"
        sleep 5
    done
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "${script_dir}/.." && pwd)
per_scene_launcher="${script_dir}/train_sogs_a100.sh"
environment_verifier="${script_dir}/verify_sogs_environment.sh"

data_root="data"
output_root=""
log_root=""
profile="throughput"
physical_gpu=0
idle_timeout_seconds=60
skip_verify=0
dry_run=0
scenes=()
default_scenes=(
    HCM0421
    HCM0539
    HCM0540
    HCM0644
    HCM0674
    bonsai
    chair
)

while (($# > 0)); do
    case "$1" in
        --data-root)
            (($# >= 2)) || fail "--data-root requires a path"
            data_root=$2
            shift 2
            ;;
        --profile)
            (($# >= 2)) || fail "--profile requires a value"
            profile=$2
            shift 2
            ;;
        --output-root)
            (($# >= 2)) || fail "--output-root requires a path"
            output_root=$2
            shift 2
            ;;
        --log-root)
            (($# >= 2)) || fail "--log-root requires a path"
            log_root=$2
            shift 2
            ;;
        --gpu)
            (($# >= 2)) || fail "--gpu requires an index"
            physical_gpu=$2
            shift 2
            ;;
        --idle-timeout)
            (($# >= 2)) || fail "--idle-timeout requires seconds"
            idle_timeout_seconds=$2
            shift 2
            ;;
        --skip-verify)
            skip_verify=1
            shift
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            while (($# > 0)); do
                scenes+=("$1")
                shift
            done
            ;;
        -*)
            fail "unknown option: $1"
            ;;
        *)
            scenes+=("$1")
            shift
            ;;
    esac
done

case "${profile}" in
    throughput)
        profile_summary="A100 throughput (full resolution, D=16, M=2, SGL=0.01, 10 offsets, densification to 15,000)"
        profile_default_output="outputs/a100_max_speed"
        profile_default_logs="logs/a100_max_speed"
        profile_log_interval=250
        profile_finite_scans="disabled"
        profile_tf32="enabled"
        profile_adam="automatic fused/foreach fallback"
        ;;
    ultra)
        # INFERENCE: this explicit profile changes the workload to target much
        # higher iteration throughput; it is not a quality-equivalent profile.
        profile_summary="A100 ultra speed (quarter width/height, D=8, M=1, SGL=0, 5 offsets, densification to 7,500)"
        profile_default_output="outputs/a100_ultra_speed"
        profile_default_logs="logs/a100_ultra_speed"
        profile_log_interval=1000
        profile_finite_scans="disabled"
        profile_tf32="enabled"
        profile_adam="automatic fused/foreach fallback"
        ;;
    quality)
        profile_summary="A100 quality control (full resolution, D=32, M=2, SGL=0.01, 10 offsets, densification to 15,000)"
        profile_default_output="outputs/a100_quality"
        profile_default_logs="logs/a100_quality"
        profile_log_interval=250
        profile_finite_scans="enabled"
        profile_tf32="disabled"
        profile_adam="original default"
        ;;
    *)
        fail "unknown profile: ${profile}; expected throughput, ultra, or quality"
        ;;
esac

[[ "${physical_gpu}" =~ ^[0-9]+$ ]] \
    || fail "--gpu must be a non-negative integer"
[[ "${idle_timeout_seconds}" =~ ^[0-9]+$ ]] \
    || fail "--idle-timeout must be a non-negative integer"
[[ -x "${per_scene_launcher}" ]] \
    || fail "missing executable launcher: ${per_scene_launcher}"

if ((${#scenes[@]} == 0)); then
    scenes=("${default_scenes[@]}")
fi

output_root=${output_root:-${profile_default_output}}
log_root=${log_root:-${profile_default_logs}}
data_root=$(make_absolute "${data_root}")
output_root=$(make_absolute "${output_root}")
log_root=$(make_absolute "${log_root}")

declare -A selected_scenes=()
for scene in "${scenes[@]}"; do
    [[ "${scene}" =~ ^[A-Za-z0-9._-]+$ ]] \
        || fail "invalid scene name: ${scene}"
    [[ -z "${selected_scenes[${scene}]+selected}" ]] \
        || fail "scene listed more than once: ${scene}"
    selected_scenes["${scene}"]=1

    scene_path="${data_root}/${scene}/train"
    model_path="${output_root}/${scene}"
    [[ -d "${scene_path}" ]] \
        || fail "scene training directory does not exist: ${scene_path}"
    if [[ -d "${model_path}" ]] \
        && [[ -n "$(find "${model_path}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        fail "model directory is not empty: ${model_path}"
    fi
done

printf 'Repository: %s\n' "${repository_root}"
printf 'Data root:  %s\n' "${data_root}"
printf 'Output root: %s\n' "${output_root}"
printf 'Scenes:'
printf ' %s' "${scenes[@]}"
printf '\n'
printf 'Profile: %s; 30,000 iterations\n' "${profile_summary}"

if ((dry_run)); then
    printf 'Dry run: GPU checks, verification, output creation, and training are disabled.\n'
    for scene in "${scenes[@]}"; do
        DRY_RUN=1 GPU_ID=-1 LOG_INTERVAL="${profile_log_interval}" \
            bash "${per_scene_launcher}" \
            "${profile}" \
            "${data_root}/${scene}/train" \
            "${output_root}/${scene}"
    done
    exit 0
fi

command -v nvidia-smi >/dev/null 2>&1 \
    || fail "nvidia-smi is unavailable"
command -v python >/dev/null 2>&1 \
    || fail "python is unavailable; activate the training environment first"
[[ -x "${environment_verifier}" ]] \
    || fail "missing executable verifier: ${environment_verifier}"

gpu_name=$(
    nvidia-smi \
        --id="${physical_gpu}" \
        --query-gpu=name \
        --format=csv,noheader
)
[[ "${gpu_name}" == *A100* ]] \
    || fail "GPU ${physical_gpu} is not an A100: ${gpu_name}"

printf '\nSelected GPU state:\n'
nvidia-smi \
    --id="${physical_gpu}" \
    --query-gpu=index,name,pstate,persistence_mode,temperature.gpu,power.limit,memory.used,memory.total,utilization.gpu \
    --format=csv

# Never run even the small verifier beside another user's/agent's CUDA job.
assert_no_compute_processes
if ((!skip_verify)); then
    printf '\nRunning non-training environment and data verification...\n'
    bash "${environment_verifier}" \
        --gpu "${physical_gpu}" \
        --data-root "${data_root}"
fi
wait_for_idle_gpu

# COMPATIBILITY: expose exactly one physical GPU and let train.py address it
# as its only logical device instead of remapping the physical ID a second time.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${physical_gpu}"
export PYTHONUNBUFFERED=1
export TORCH_CUDA_ARCH_LIST=8.0
# INFERENCE: debug-only synchronous launching and allocator-cache disabling
# severely reduce throughput and are inappropriate for this explicit profile.
unset CUDA_LAUNCH_BLOCKING
unset PYTORCH_NO_CUDA_MEMORY_CACHING

mkdir -p "${log_root}"
run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_log_root="${log_root}/${run_stamp}"
mkdir -p "${run_log_root}"

printf '\nRuntime controls:\n'
printf '  CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES}"
printf '  TORCH_CUDA_ARCH_LIST=%s\n' "${TORCH_CUDA_ARCH_LIST}"
printf '  CUDA debug synchronization: disabled\n'
printf '  PyTorch CUDA allocator cache: enabled\n'
printf '  Per-iteration finite scans: %s\n' "${profile_finite_scans}"
printf '  TF32: %s\n' "${profile_tf32}"
printf '  Adam backend: %s\n' "${profile_adam}"
printf '  Activation checkpoint recomputation: disabled\n'
printf '  Training-time evaluation/post-processing: disabled\n'
printf '  Scalar/progress synchronization: every %s iterations\n' \
    "${profile_log_interval}"
printf '  Logs: %s\n\n' "${run_log_root}"

run_started_epoch=$(date +%s)
for scene in "${scenes[@]}"; do
    scene_path="${data_root}/${scene}/train"
    model_path="${output_root}/${scene}"
    scene_log="${run_log_root}/${scene}.log"
    scene_started_epoch=$(date +%s)

    printf '\n[%s] Starting %s training.\n' "${scene}" "${profile}" \
        | tee -a "${scene_log}"
    set +e
    GPU_ID=-1 LOG_INTERVAL="${profile_log_interval}" \
        bash "${per_scene_launcher}" \
        "${profile}" \
        "${scene_path}" \
        "${model_path}" \
        2>&1 | tee -a "${scene_log}"
    training_status=${PIPESTATUS[0]}
    set -e

    scene_elapsed_seconds=$(($(date +%s) - scene_started_epoch))
    if ((training_status != 0)); then
        printf '[%s] FAILED after %ss (exit %s). See %s\n' \
            "${scene}" \
            "${scene_elapsed_seconds}" \
            "${training_status}" \
            "${scene_log}" >&2
        exit "${training_status}"
    fi
    printf '[%s] Completed in %ss.\n' \
        "${scene}" "${scene_elapsed_seconds}" \
        | tee -a "${scene_log}"
done

total_elapsed_seconds=$(($(date +%s) - run_started_epoch))
printf '\nAll %s scene(s) completed in %ss.\nModels: %s\nLogs: %s\n' \
    "${#scenes[@]}" \
    "${total_elapsed_seconds}" \
    "${output_root}" \
    "${run_log_root}"
