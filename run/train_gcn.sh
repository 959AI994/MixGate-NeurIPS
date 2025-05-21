# scripts/encoder/train_gcn.sh
#!/bin/bash
torchrun --nproc_per_node=2 --master_port=26618 \
train_encoder.py \
--gpus 4,5 \
--aig_encoder gcn \
--batch_size 256 \
--exp_id 02_origin_gcn