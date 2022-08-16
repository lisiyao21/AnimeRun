import argparse
import numpy as np
from datasets.anime_seg_mat_dataset import fetch_dataloader


if __name__ == '__main__':

    args = argparse.Namespace()
    args.dstype = 'Frame_Anime'
    # args.dstype = 'contour'
    args.batch_size = 1
    args.stage = 'anime'
    # args.image_size = (368, 368)
    loader = fetch_dataloader(args, 'test')
    count_len = []
    
    print(len(loader))
    for data in loader:
        dict1 = data
        mi = dict1['all_matches']
        count_len.append(len(mi[0]))
    print(np.max(count_len))
