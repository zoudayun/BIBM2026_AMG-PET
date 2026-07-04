EXP_NAME=""
EXP_DIR="//${EXP_NAME}"

cd 

echo "============== Pretraining starts =============="
touch ~/wait1
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=  python launch.py \
  --main_py_relpath main_pet_foundation_model.py \
  --exp_name "${EXP_NAME}" \
  --exp_dir "${EXP_DIR}" \
  --num_nodes=1 \
  --ngpu_per_node= \
  --node_rank=0 \
  --master_address=128.0.1.3 \
  --master_port=5200 \
  --data_path= \
  --opt=adamw \
  --amp=True\
  --bs=48 \
  --ep=200 \
  --wp_ep=10 \
  --input_size=64 \
  --dataloader_workers= \
  --base_lr=1e-4 \
  --wd=0.2 \
  --mim_ratio=0.75 \
  --patch_size=8 \
  --weight_recon=1.0\
  --weight_clip=1.0\
  --weight_matching=1.0\
  --weight_suvr=1.0\
  --resume_from=

echo "============== Pretraining ends =============="
rm ~/wait1
