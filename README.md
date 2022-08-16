# AnimeRun API
This repository contains the api for our proposed AnimeRun dataset.

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
 - [ ] Release re-implemented AnT
 - [ ] Add liscense
