EXP_NAME="mask75_64X64X64_RECON_Suvr_resume_0620"
EXP_DIR="/data/junyan/LiangJ/PET_Foundation_Model/experiments/${EXP_NAME}"

cd /data/junyan/LiangJ/PET_Foundation_Model

echo "============== Pretraining starts =============="
touch ~/wait1
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python launch.py \
  --main_py_relpath main_pet_foundation_model.py \
  --exp_name "${EXP_NAME}" \
  --exp_dir "${EXP_DIR}" \
  --num_nodes=1 \
  --ngpu_per_node=6 \
  --node_rank=0 \
  --master_address=128.0.1.3 \
  --master_port=5200 \
  --data_path=/data/junyan/PET_MNI_1mm \
  --opt=adamw \
  --amp=True\
  --bs=96 \
  --ep=160 \
  --wp_ep=10 \
  --input_size=64 \
  --dataloader_workers=2 \
  --base_lr=1e-4 \
  --wd=0.2 \
  --mim_ratio=0.75 \
  --patch_size=8 \
  --weight_recon=1.0\
  --weight_clip=0.0\
  --weight_matching=0.0\
  --weight_suvr=1.0\
  --resume_from=/data/junyan/LiangJ/PET_Foundation_Model/experiments/mask75_64X64X64_RECON_Suvr_0616/unet_40.pth

echo "============== Pretraining ends =============="
rm ~/wait1