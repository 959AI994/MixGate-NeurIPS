# scripts/encoder/train_hoga.sh
#!/bin/bash
torchrun --nproc_per_node=2 --master_port=29918 \
train_encoder.py \
--gpus 0,1 \
--aig_encoder hoga \
--batch_size 256 \
--exp_id 02_origin_hoga