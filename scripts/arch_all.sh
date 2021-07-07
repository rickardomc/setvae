#! /bin/bash

input_dim=3
max_outputs=2500
init_dim=32
n_mixtures=4
z_dim=16
hidden_dim=64
num_heads=4

lr=1e-3
beta=1.0
epochs=8000
scheduler="linear"
dataset_type=arch
log_name=gen/arch-scale/camera-ready
shapenet_data_dir="/content/arch_by_class_npy_traslato"

deepspeed train.py \
  --cates scale \
  --input_dim ${input_dim} \
  --max_outputs ${max_outputs} \
  --init_dim ${init_dim} \
  --n_mixtures ${n_mixtures} \
  --z_dim ${z_dim} \
  --z_scales 1 1 2 4 8 16 32 \
  --hidden_dim ${hidden_dim} \
  --num_heads ${num_heads} \
  --kl_warmup_epochs 2000 \
  --fixed_gmm \
  --train_gmm \
  --lr ${lr} \
  --beta ${beta} \
  --epochs ${epochs} \
  --dataset_type ${dataset_type} \
  --log_name ${log_name} \
  --shapenet_data_dir ${shapenet_data_dir} \
  --resume_optimizer \
  --save_freq 100 \
  --viz_freq 1000 \
  --log_freq 10 \
  --val_freq 1000 \
  --scheduler ${scheduler} \
  --slot_att \
  --ln \
  --eval \
  --seed 42 \
  --distributed \
  --deepspeed_config batch_size.json

echo "Done"
exit 0
