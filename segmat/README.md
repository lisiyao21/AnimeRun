# AnimeRun (Reimplemented AnT)

Code for self-implemented Animation Transformer (AnT). Our experimental environment uses Pytorch1.7

(I am sorry if the code is a mess. Please give me more time to clean it.)


### Train
if using slurm:

    sh srun.sh configs/[your config] train [partation name] 1

else:

    python -u main.py --config configs/[your config] --train


### Test
if using slurm:
    
    sh srun.sh configs/[your config] eval [partation name] 1

else:

    python -u main.py --config configs/[your config] --eval

# TODO
- [ ] Clean the code
- [ ] Clean & add functional scripts (visualization on segment matching...)
- [ ] Double check values & upload pretrained weights
- [ ] Add more content (link to paper, to project page ...)
- [ ] Add liscence