# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
from glob import glob
import os.path as osp

from .utils import frame_utils
from .utils.augmentor import FlowAugmentor, SparseFlowAugmentor

import sys

class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

        self.occ_list = []
        self.line_list = []

    def __getitem__(self, index):
        # print('Index is {}'.format(index))
        # sys.stdout.flush()
        index = index % len(self.image_list)
        if self.is_test:
            self.augmentor = None

        valid = None
        flow = frame_utils.read_gen(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            
        if len(img2.shape) == 2:
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img2 = img2[..., :3]

        if self.augmentor is not None:
            img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        if self.is_test:
            occ = torch.from_numpy(np.load(self.occ_list[index]))
            line = torch.from_numpy(np.load(self.line_list[index]))
            return img1, img2, flow, occ, line, self.extra_info[index], valid.float()
        else:
            return img1, img2, flow, valid.float()


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)
        

class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='/mnt/lustre/syli/AnimeRun/flow/data/Sintel', dstype='clean'):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
                self.extra_info += [ (scene, i) ] # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))


class AnimeRun(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='/mnt/lustre/syli/AnimeRun/flow/data/AnimeRun1', dstype='Frame_Anime/original'):
        super(AnimeRun, self).__init__(aug_params)
        # root = root + subset
        flow_root = osp.join(root, split, 'Flow')
        image_root = osp.join(root, split, dstype)
        unmatch_root = osp.join(root, split, 'Unmatched')
        line_root = osp.join(root, split, 'LineArea')

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            for color_pass in os.listdir(osp.join(image_root, scene)):
                if color_pass != 'original':
                    continue
                image_list = sorted(glob(osp.join(image_root, scene, color_pass, '*.png')))
                for i in range(len(image_list)-1):
                    self.image_list += [ [image_list[i], image_list[i+1]] ]
                    self.extra_info += [ (scene, i) ] # scene and frame_id
                self.occ_list += sorted(glob(osp.join(unmatch_root, scene, '*.npy')))
                self.flow_list += sorted(glob(osp.join(flow_root, scene, 'forward', '*.flo')))
                self.line_list += [ path.replace(unmatch_root, line_root) for path in sorted(glob(osp.join(unmatch_root, scene, '*.npy')))]

        print('Len of Flow is ', len(self.flow_list))
        print('Len of Anime is ', len(self.image_list))
        print('Len of Occlusion is ', len(self.occ_list))
        print('Len of Line Area is ', len(self.line_list))



class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='/mnt/lustre/syli/AnimeRun/flow/data/FlyingThings', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            self.image_list += [ [images[i], images[i+1]] ]
                            self.flow_list += [ flows[i] ]
                        elif direction == 'into_past':
                            self.image_list += [ [images[i+1], images[i]] ]
                            self.flow_list += [ flows[i+1] ]
      


class CreativeFlow(FlowDataset):
    def __init__(self, aug_params=None, split='test', root='/mnt/lustre/syli/AnimeRun/flow/data/CreativeFlow/decompressed'):
        super(CreativeFlow, self).__init__(aug_params)

        # creative flow only used for train
        flow_root = osp.join(root, split)
        image_root = osp.join(root, split)
        for scene in os.listdir(image_root):
            if split == 'test':
                render_folder = os.listdir(osp.join(image_root, scene, 'cam0', 'renders', 'composite'))[0]
                image_list = sorted(glob(osp.join(image_root, scene, 'cam0', 'renders', 'composite', render_folder, '*.png')))
                flow_list = sorted(glob(osp.join(flow_root, scene, 'cam0', 'metadata', 'flow', '*.flo')))
            else:
                render_folder = os.listdir(osp.join(image_root, scene, 'renders', 'composite'))[0]
                image_list = sorted(glob(osp.join(image_root, scene, 'renders', 'composite', render_folder, '*.png')))
                flow_list = sorted(glob(osp.join(flow_root, scene, 'metadata', 'flow', '*.flo')))

            if  len(image_list)-1 == len(flow_list):
                for i in range(len(image_list)-1):
                    self.image_list += [ [image_list[i], image_list[i+1]] ]
                    self.extra_info += [ (scene, i) ] # scene and frame_id
                    self.flow_list += [ flow_list[i] ]
        print(len(self.image_list), len(self.extra_info), len(self.flow_list))


def fetch_dataloader(args, TRAIN_DS='C+T+K/S'):
    """ Create the data loader for the corresponding trainign set """

    if args.stage == 'sintel':
        print('Training Sintel Stage...')
        sys.stdout.flush()
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')        

        if TRAIN_DS == 'C+T+K/S':
            train_dataset = 100*sintel_clean + 100*sintel_final + things

    elif args.stage == 'anime':
        print('Training AnimeRun Stage...')
        sys.stdout.flush()
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        print('Here!')
        anime_run = 200 * AnimeRun(aug_params, split='train', dstype='Frame_Anime')

        
        train_dataset = anime_run + things

    elif args.stage == 'creative':
        print('Training Using Creative Flow...')
        sys.stdout.flush()
        aug_params =  {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        creative = CreativeFlow(aug_params, split='test')

        train_dataset =  20*creative + things

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True, num_workers=4, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader

