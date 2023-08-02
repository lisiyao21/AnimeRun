# Segment Matching (Reimplemented AnT)

Code for self-implemented Animation Transformer (AnT). Our experimental environment uses Pytorch1.7.

In concerete, AnT takes two RGB frames and their segment labels as input and predicts segment-wise correspondece between these two images. Please refer to the original [paper](https://openaccess.thecvf.com/content/ICCV2021/html/Casey_The_Animation_Transformer_Visual_Correspondence_via_Segment_Matching_ICCV_2021_paper.html).

## TODO
- [x] Clean the code
- [x] Clean & add functional scripts (visualization on segment matching...)
- [x] Double check values & upload pretrained weights
- [ ] Add more content (link to paper, to project page ...)
- [x] Add liscence

### Pretrained weights
The pretrained weights can be downloaded from [[Google Drive](https://drive.google.com/file/d/1Ryt4ngytVRCp-FJdIKxw4M5OpSRw8bwO/view?usp=sharing)] link.


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

In detail, after downloading the pretrained weights, move the two folders 'ant_softmax_anime_v2_bsz16' and 'ant_softmax_anime_contour_bsz16_anime_v2' into 'experiments' folder. These two weights are self-implemented AnT trained on colored frames and contour lines, respectively. To test them, run

    sh srun.sh configs/superglue_softmax_anime.yaml eval [partation name] 1

and 

    sh srun.sh configs/superglue_softmax_anime_contour.yaml eval [partation name] 1

respectively.

### License

This project is licensed under [NTU S-Lab License 1.0](https://github.com/lisiyao21/Bailando/blob/main/LICENSE). Redistribution and use should follow this license.

