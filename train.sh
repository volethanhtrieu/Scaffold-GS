function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))  
}

port=$(rand 10000 30000)

lod=0
iterations=30_000
warmup="False"
feat_dim=32
use_second_order="False"
num_eigenvectors=2
lambda_sgl=0.01
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -l|--logdir) logdir="$2"; shift ;;
        -d|--data) data="$2"; shift ;;
        --lod) lod="$2"; shift ;;
        --gpu) gpu="$2"; shift ;;
        --warmup) warmup="$2"; shift ;;
        --voxel_size) vsize="$2"; shift ;;
        --update_init_factor) update_init_factor="$2"; shift ;;
        --appearance_dim) appearance_dim="$2"; shift ;;
        --ratio) ratio="$2"; shift ;;
        --feat_dim) feat_dim="$2"; shift ;;
        --use_second_order) use_second_order="$2"; shift ;;
        --num_eigenvectors) num_eigenvectors="$2"; shift ;;
        --lambda_sgl) lambda_sgl="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

time=$(date "+%Y-%m-%d_%H:%M:%S")

# Viettel AI Race scenes keep the competition's train/test split one level
# below the scene directory. Resolve that layout without copying any data;
# the fallback preserves the original repository behavior for public sets.
source_path="data/${data}"
if [ -d "${source_path}/train/images" ] && [ -d "${source_path}/train/sparse/0" ]; then
    source_path="${source_path}/train"
fi

if [ "$warmup" = "True" ]; then
    python train.py --eval -s "${source_path}" --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --feat_dim "${feat_dim}" --use_second_order "${use_second_order}" --num_eigenvectors "${num_eigenvectors}" --lambda_sgl "${lambda_sgl}" --warmup --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time
else
    python train.py --eval -s "${source_path}" --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --feat_dim "${feat_dim}" --use_second_order "${use_second_order}" --num_eigenvectors "${num_eigenvectors}" --lambda_sgl "${lambda_sgl}" --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time
fi
