#!/bin/bash
mkdir -p checkpoints
python -u train_pwc.py --name pwc-ft-creative --stage creative --validation anime --restore_ckpt models/network-default.pytorch --gpus 0 --num_steps 100000 --batch_size 12 --lr 0.0001 --image_size 480 480 --wdecay 0.00001 --gamma=0.85 --mixed_precision
