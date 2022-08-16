import argparse
from datasets.flow_datasets import AnimeRun, fetch_dataloader


if __name__ == '__main__':
    data = AnimeRun(split='test', dstype='Frame_Anime')
    print(len(data))

    args = argparse.Namespace()
    args.stage = 'anime'
    args.dstype = 'Frame_Anime'
    args.image_size = [384, 768]
    args.batch_size = 12
    loader = fetch_dataloader(args)
    print(len(loader))
    
