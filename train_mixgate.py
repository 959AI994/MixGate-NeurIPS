from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mixgate
import torch
import os
from config import get_parse_args
import mixgate.top_model
import mixgate.top_trainer 
import torch.distributed as dist

# Replace with your own dataset path
DATA_DIR = './datasets'

if __name__ == '__main__':
    args = get_parse_args()

    # Replace with your own dataset path
    circuit_path ='datasets/mixgate_data/train_mixgate.npz'
    num_epochs = args.num_epochs
    
    print('[INFO] Parse Dataset')
    dataset = mixgate.NpzParser_Pair(DATA_DIR, circuit_path)

    train_dataset, val_dataset = dataset.get_dataset()

    print('[INFO] Create Model and Trainer')

    # Replace with your own pre-traine encoder model weights
    model = mixgate.top_model.TopModel(
        args, 
        dg_ckpt_aig='./ckpt/model_func_aig.pth',
        dg_ckpt_xag='./ckpt/model_func_xag.pth',
        dg_ckpt_xmg='./ckpt/model_func_xmg.pth',
        dg_ckpt_mig='./ckpt/model_func_mig.pth'
    )

    trainer = mixgate.top_trainer.TopTrainer(args, model, distributed=True)
    trainer.set_training_args(lr=1e-4, lr_step= 50, loss_weight = [1.0, 0.0, 0.0])
    print('[INFO] Stage 1 Training ...')
    trainer.train(num_epochs, train_dataset, val_dataset)

    trainer.save(os.path.join(trainer.log_dir, 'stage1_model.pth'))
    print('[INFO] Loading Stage 1 Checkpoint...')
    trainer.load(os.path.join(trainer.log_dir, 'stage1_model.pth'))

    print('[INFO] Stage 2 Training ...')
    trainer.set_training_args(loss_weight = [1.0, 0.0, 1.0], lr=1e-4, lr_step= 50)
    trainer.train(num_epochs, train_dataset, val_dataset)
    trainer.save(os.path.join(trainer.log_dir, 'stage2_model.pth'))


    
    