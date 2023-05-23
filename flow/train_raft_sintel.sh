#!/bin/bash
mkdir -p checkpoints
python -u train_raft.py --name raft-ft-sintel --stage sintel --validation anime --restore_ckpt models/raft-sintel.pth --gpus 0 1 --num_steps 20000 --batch_size 12 --lr 0.0001 --image_size 368 768 --wdecay 0.00001 --gamma=0.85 --mixed_precision
