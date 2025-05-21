# scripts/pretrain/pretrain_mixgate_deepgate2.sh
#!/bin/bash
torchrun --nproc_per_node=8 --master_port=29919 \
train_mixgate.py \
--exp_id 01_mixgate_deepgate2 \
--batch_size 128 \
--num_epochs 60 \
--gpus 0,1,2,3,4,5,6,7 \
--hier_tf \
--aig_encoder dg2