# Segment Matching (Reimplemented AnT)

Code for self-implemented Animation Transformer (AnT). Our experimental environment uses Pytorch1.7

## TODO
- [x] Clean the code
- [x] Clean & add functional scripts (visualization on segment matching...)
- [ ] Double check values & upload pretrained weights
- [ ] Add more content (link to paper, to project page ...)
- [x] Add liscence

### Training
if using slurm:

    sh srun.sh configs/[your config] train [partation name] 1

else:

    python -u main.py --config configs/[your config] --train


### Testing
if using slurm:
    
    sh srun.sh configs/[your config] eval [partation name] 1

else:

    python -u main.py --config configs/[your config] --eval

### License

This project is licensed under [NTU S-Lab License 1.0](https://github.com/lisiyao21/Bailando/blob/main/LICENSE). Redistribution and use should follow this license.

