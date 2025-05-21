# scripts/encoder/train_dg3.sh
#!/bin/bash
torchrun --nproc_per_node=2 --master_port=25518 \
train_encoder.py \
--gpus 6,7 \
--aig_encoder dg3 \
--batch_size 256 \
--exp_id 02_origin_dg3