#!/bin/bash
mkdir -p checkpoints
# python -u train.py --name raft-chairs --stage chairs --validation chairs --gpus 0 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision 
# python -u train.py --name raft-things --stage things --validation sintel --restore_ckpt checkpoints/raft-chairs.pth --gpus 0 --num_steps 120000 --batch_size 5 --lr 0.0001 --image_size 400 720 --wdecay 0.0001 --mixed_precision
# python -u train.py --name raft-ft-anime-20000 --stage anime --validation anime --restore_ckpt models/raft-sintel.pth --gpus 0 1 --num_steps 20000 --batch_size 12 --lr 0.0001 --image_size 368 768 --wdecay 0.00001 --gamma=0.85 --mixed_precision
# python -u train.py --name raft-kitti  --stage kitti --validation kitti --restore_ckpt checkpoints/raft-sintel.pth --gpus 0 --num_steps 50000 --batch_size 5 --lr 0.0001 --image_size 288 960 --wdecay 0.00001 --gamma=0.85 --mixed_precision
# python -u train.py --name trytry --stage things --validation anime --restore_ckpt models/raft-sintel.pth --gpus 0 1 --num_steps 20000 --batch_size 12 --lr 0.0001 --image_size 368 768 --wdecay 0.00001 --gamma=0.85 --mixed_precision

python -u train_gmflow.py --name gmflow-animerun-v2-ft --stage anime --validation anime --restore_ckpt models/gmflow_with_refine_sintel-3ed1cf48.pth --gpus 0 1 --num_steps 20000 --batch_size 6 --lr 0.0001 --image_size 384 768 --wdecay 0.00001 --gamma=0.85 --mixed_precision
