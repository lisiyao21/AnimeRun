# NeurIPS 2022 Dataset & Benchmark Track: AnimeRun: 2D Animation Visual Correspondence from Open Source 3D Movies

This repository contains the api for our proposed AnimeRun dataset and relevant methods.
Complete and clean code is on the way! Coming soon!!

## AnimeRun structure

```Shell
├── AnimeRun1
    ├── test
         ├── contour
         ├── Flow
              ├── forward
              ├── backward
         ├── Frame_Anime
         ├── LineArea
         ├── SegMatching
         ├── Segment
         ├── UnMatched
         
    ├── train
         ├── contour
         ├── Flow
              ├── forward
              ├── backward
         ├── Frame_Anime
         ├── SegMatching
         ├── Segment
         ├── UnMatched

```

## Unit test
    python flow/test_flow_datasets.py

    python segmat/test_segmat_datasets.py

# TODO
 - [ ] More content
 - [x] Release re-implemented AnT
 - [ ] Add liscense
