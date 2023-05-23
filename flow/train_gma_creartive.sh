#!/bin/bash
mkdir -p checkpoints
python -u train_gma.py --name gma-creative-ft  --stage creative --validation anime --restore_ckpt models/gma-sintel.pth --gpus 0 1 --num_steps 20000 --batch_size 12 --lr 0.0001 --image_size 480 480 --wdecay 0.00001 --gamma=0.85 --mixed_precision 
