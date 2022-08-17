# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import networkx as nx
import os
import math
import random
from glob import glob
import os.path as osp

from .utils import frame_utils
from .utils.augmentor import SegMatAugmentor

import sys
import argparse
# import cv2
from collections import Counter

# import pdb


class AnimeSegMatDataset(data.Dataset):
    def __init__(self, aug_params=None):
        self.augmentor = None
        if aug_params is not None:
            self.augmentor = SegMatAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False

        # image, seg, matching index
        self.image_list = []
        self.seg_list = []
        self.index_list = []
        # self.extra_info = []

    def __getitem__(self, index):

        file_name = self.image_list[index][0][:-4]
        if self.is_test:
            self.augmentor = None

        index = index % len(self.image_list)
        valid = None
        
        # read images
        img1 = frame_utils.read_gen(self.image_list[index][0]).convert('RGB')
        img2 = frame_utils.read_gen(self.image_list[index][1]).convert('RGB')
        
        # load segmetns
        seg1 = frame_utils.read_gen(self.seg_list[index][0])
        seg2 = frame_utils.read_gen(self.seg_list[index][1])

        # flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        seg1 = np.array(seg1).astype(np.int64)
        seg2 = np.array(seg2).astype(np.int64)



        # here we only compute the index as first-order matching
        if self.index_list is not None:
            mat_index = frame_utils.read_gen(self.index_list[index])
            mat_index = np.array(mat_index).astype(np.int64)
            self.augmentor = None

        # image --> 3 channels
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        # autmentation
        if self.augmentor is not None:
            img1, img2, seg1, seg2, mat_index = self.augmentor(img1, img2, seg1, seg2, mat_index)

        kpt1 = []
        kpt2 = []
        cpt1 = []
        cpt2 = []
        numpt1 = []
        numpt2 = []

        h, w = seg1.shape
        hh = np.arange(h)
        ww = np.arange(w)
        xx, yy = np.meshgrid(ww, hh)

        # To avoid mass center lying out of the segment, we randomly select one in segment

        sys.stdout.flush()

        # record segment boundary
        for ii in range(len(Counter(seg1.reshape(-1)))):
            xs = xx[seg1 == ii]
            ys = yy[seg1 == ii]

            xmin = xs.min()
            xmax = xs.max()
            ymin = ys.min()
            ymax = ys.max()
            xmean = xs.mean()
            ymean = ys.mean()
            
            cpt1.append([xmean, ymean])
            numpt1.append((seg1 == ii).sum())
            kpt1.append([xmin, xmax, ymin, ymax])

        for ii in range(len(Counter(seg2.reshape(-1)))):
            xs = xx[seg2 == ii]
            ys = yy[seg2 == ii]

            xmin = xs.min()
            xmax = xs.max()
            ymin = ys.min()
            ymax = ys.max()
            xmean = xs.mean()
            ymean = ys.mean()
            
            cpt2.append([xmean, ymean])
            numpt2.append((seg2 == ii).sum())
            kpt2.append([xmin, xmax, ymin, ymax])
        

        kpt1 = np.stack(kpt1)
        kpt2 = np.stack(kpt2)
        cpt1 = np.stack(cpt1)
        cpt2 = np.stack(cpt2)
        numpt1 = np.stack(numpt1)
        numpt2 = np.stack(numpt2)


        # image output [0, 1]
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float() * 2 / 255.0 - 1.0 
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float() * 2 / 255.0 - 1.0
        seg1 = torch.from_numpy(seg1)[None]
        seg2 = torch.from_numpy(seg2)[None]
        numpt1 = torch.from_numpy(numpt1)[None]
        numpt2 = torch.from_numpy(numpt2)[None]


        

        if self.index_list is not None:
            mat_index = torch.from_numpy(mat_index).float()
            return{
                'keypoints0': kpt1,
                'keypoints1': kpt2,
                'center_points0': cpt1,
                'center_points1': cpt2,

                'image0': img1,
                'image1': img2,
                'num0': numpt1,
                'num1': numpt2,
                'segment0': seg1,
                'segment1': seg2,
                'all_matches': mat_index, # segment index matching 
                'file_name': file_name,
                # 'with_match': True
            } 
        else:
            mat_index = None
            return{
                'keypoints0': kpt1,
                'keypoints1': kpt2,
                'center_points0': cpt1,
                'center_points1': cpt2,

                'image0': img1,
                'image1': img2,
                'num0': numpt1,
                'num1': numpt2,
                'segment0': seg1,
                'segment1': seg2,

                # 'with_match': False,
                'file_name': file_name
            } 
        

    def __rmul__(self, v):
        self.index_list = v * self.index_list
        self.seg_list = v * self.seg_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)
        


class AnimeRunSegMat(AnimeSegMatDataset):
    def __init__(self, aug_params=None, split='training', root='/mnt/lustre/syli/AnimeRun/flow/data/AnimeRun1', dstype='Frame_Anime'):
        super(AnimeRunSegMat, self).__init__(aug_params)
        # root = root
        seg_root = osp.join(root, split, 'Segment')
        image_root = osp.join(root, split, dstype)
        mat_root = osp.join(root, split, 'SegMatching')

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            if dstype == 'Frame_Anime':
                image_list = sorted(glob(osp.join(image_root, scene, 'original', '*.png')))
            else:
                image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            seg_list = sorted(glob(osp.join(seg_root, scene, '*.npy')))
            mat_list = sorted(glob(osp.join(mat_root, scene, 'forward', '*.json')))

            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
            for i in range(len(seg_list) - 1):
                self.seg_list += [ [seg_list[i], seg_list[i+1]] ]

            self.index_list += mat_list
        print('Len of Segis ', len(self.seg_list))
        print('Len of SegMat is ', len(self.index_list))
        print('Len of Anime is ', len(self.image_list))
        # print('')

class CreativeFlow(AnimeSegMatDataset):
    def __init__(self, aug_params=None, split='test', root='/mnt/lustre/syli/AnimeRun/flow/data/CreativeFlow/decompressed', dstype='composite'):
        super(CreativeFlow, self).__init__(aug_params)

        # if split is 'train':
        # flow_root = osp.join(root, split)
        image_root = osp.join(root, split)
        index_root = osp.join(root, 'test_seg_matching')
        seg_root = osp.join(root, 'test_segment')

        for scene in os.listdir(index_root):
            # if split == 'test':
            render_folder = os.listdir(osp.join(image_root, scene, 'cam0', 'renders', dstype))[0]
            image_list = sorted(glob(osp.join(image_root, scene, 'cam0', 'renders', dstype, render_folder, '*.png')))
            index_list = sorted(glob(osp.join(index_root, scene, '*.json')))
            seg_list = sorted(glob(osp.join(seg_root, scene, '*.npy')))

            if  (len(index_list) == len(seg_list) - 1) and (len(seg_list) == len(image_list)):
                for i in range(len(seg_list)-1):
                    self.image_list += [ [image_list[i], image_list[i+1]] ]
                    # self.extra_info += [ (scene, i) ] # scene and frame_id
                    self.seg_list += [ [seg_list[i], seg_list[i+1]] ]
                    self.index_list += [ index_list[i] ]
        print('Len of Seg is ', len(self.seg_list))
        print('Len of SegMat is ', len(self.index_list))
        print('Len of Anime is ', len(self.image_list))


class ATDSeg(AnimeSegMatDataset):
    def __init__(self, aug_params=None, image_root='/mnt/lustre/syli/AnimeRun/interp/data/datasets/test_2k_540p', seg_root='/mnt/lustre/syli/AnimeRun/interp/data/datasets/test_2k_540p_segment'):
        super(ATDSeg, self).__init__(aug_params)
        # ATD only support test
        self.index_list = None

        for scene in os.listdir(seg_root):
            
            self.seg_list += [ [osp.join(seg_root, scene, 'labelmap_1.npy'), osp.join(seg_root, scene, 'labelmap_3.npy') ]]
            self.image_list += [[ osp.join(image_root, scene, 'frame1.png'), osp.join(image_root, scene, 'frame3.png') ]]

        print('Len of segment pair is ', len(self.seg_list))
        print('Len of Anime pair is ', len(self.image_list))

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def fetch_dataloader(args, type='training'):

    if args.stage == 'anime':
        if type == 'training':
            #print('Training AnimeRun Segments')
            sys.stdout.flush()
            aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
            anime_run = AnimeRunSegMat(aug_params, split='train', dstype=args.dstype if hasattr(args, 'dstype') else 'Frame_Anime')

            train_dataset = anime_run 
            # batch size must be 1 since segment number not unified
            train_loader = data.DataLoader(train_dataset, batch_size=1, 
                pin_memory=True, shuffle=True, num_workers=32, drop_last=True, worker_init_fn=worker_init_fn)
            return train_loader
        else:
            sys.stdout.flush()
            aug_params = None
            anime_run = AnimeRunSegMat(aug_params, split='test', dstype=args.dstype if hasattr(args, 'dstype') else 'Frame_Anime')
        
            test_dataset = anime_run 
            test_loader = data.DataLoader(test_dataset, batch_size=1, 
                pin_memory=True, shuffle=False, num_workers=16, drop_last=False, worker_init_fn=worker_init_fn)
        
            return test_loader
    elif args.stage == 'atd12k':
        # ATD only can be used as test without GT labels
        aug_params = None
        atd12k = ATDSeg(aug_params, args.img_root, args.seg_root)

        test_loader = data.DataLoader(atd12k, batch_size=1, 
            pin_memory=True, shuffle=False, num_workers=4, drop_last=False, worker_init_fn=worker_init_fn)
        return test_loader

    elif args.stage == 'creative':
        # creative only used for train
        if type == 'training':
            aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
            creative = CreativeFlow(aug_params, split='test', dstype=args.dstype if hasattr(args, 'dstype') else 'composite')

            train_dataset = creative 
            train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
                pin_memory=True, shuffle=True, num_workers=32, drop_last=True, worker_init_fn=worker_init_fn)
            return train_loader
    else:
        raise NotImplementedError
