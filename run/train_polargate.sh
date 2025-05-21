# scripts/encoder/train_polargate.sh
#!/bin/bash
torchrun --nproc_per_node=2 --master_port=29918 \
train_encoder.py \
--gpus 2,3 \
--aig_encoder pg \
--batch_size 256 \
--exp_id 02_origin_polargate