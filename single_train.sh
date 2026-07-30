scene='mipnerf360/bicycle'
exp_name='baseline'
voxel_size=0.001
update_init_factor=16
appearance_dim=0
ratio=1
gpu=-1
feat_dim=32
n_offsets=10
use_second_order=False
num_eigenvectors=2
lambda_sgl=0.01
sogs_chunk_size=2048
densification_chunk_size=4096

# example:
./train.sh -d "${scene}" -l "${exp_name}" --gpu "${gpu}" --voxel_size "${voxel_size}" --update_init_factor "${update_init_factor}" --appearance_dim "${appearance_dim}" --ratio "${ratio}" --feat_dim "${feat_dim}" --n_offsets "${n_offsets}" --use_second_order "${use_second_order}" --num_eigenvectors "${num_eigenvectors}" --lambda_sgl "${lambda_sgl}" --sogs_chunk_size "${sogs_chunk_size}" --densification_chunk_size "${densification_chunk_size}"
