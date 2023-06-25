## Data root 

Before training and testing, please download the AnimeRun dataset, extract it and modify the flow data root in 'datasets/flow_datasets.py' (Line 155) to your path.

## Training

The training scripts on the four datasets MPI-Sintel (T+S), Creative Flow+ (T+Cr) and AnimeRun (T+R) are ended with '_sintel.sh', '_creative.sh' and '_anime.sh', respectively. For example, if you wish to train GMFlow using AnimeRun, please run

    sh train_gmflow_anime.sh

## Testing

Similar to the training code, the test code is warped as 'evaluate_animerun_xxx.sh' where xxx is the method name. For example, to test RAFT, run

    sh eval_animerun_raft.sh


### Pretrained weights

Pretrained weights can refer to [Google Drive](https://drive.google.com/drive/folders/16fA0w1vaaU4gD7QVKRSfTRUBejan0ib8?usp=sharing).

## License

This project is licensed under [NTU S-Lab License 1.0](https://github.com/lisiyao21/Bailando/blob/main/LICENSE). Redistribution and use should follow this license.
