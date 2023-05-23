from configparser import Interpolation
from genericpath import exists
import numpy as np
import cv2


from models.utils import seg_seq_to_color
from datasets.utils.frame_utils import read_gen

import os
import os.path as osp
from glob import glob

root_folder = '/mnt/lustre/syli/AnimeRun/flow/data/AnimeRun_v2'
subfolders = ['train', 'test']

if not exists(osp.join(root_folder, 'visualize', 'Seg_viz2')):
    os.mkdir(osp.join(root_folder, 'visualize', 'Seg_viz2'))

for subfolder in subfolders:
    root_subfodler = osp.join(root_folder, subfolder)
    
    for scene in os.listdir(osp.join(root_subfodler, 'SegMatching2')):
        if scene not in ['agent_indoor3_headbutt']:
            continue
        print(scene, flush=True)
        scene_folder = osp.join(root_subfodler, 'SegMatching2', scene, 'backward')
        if not exists(osp.join(root_folder, 'visualize', 'Seg_viz2', scene)):
            os.mkdir(osp.join(root_folder, 'visualize', 'Seg_viz2', scene))
        if not exists(osp.join(root_folder, 'visualize', 'Seg_viz2', scene, 'backward')):
            os.mkdir(osp.join(root_folder, 'visualize', 'Seg_viz2', scene, 'backward'))

        for flo_name in sorted(glob(osp.join(root_subfodler, 'SegMatching2', scene, 'backward', '*.json'))):
            # print(flo_name)
            seg0_name = flo_name.replace('SegMatching2', 'Segment2').replace('backward', '').replace('.json', '.npy')
            seg0 = np.load(seg0_name)
            seg1_name = seg0_name[:-8] +  str(int(seg0_name[-8:-4]) - 1).zfill(4) + '.npy'
            seg1 = np.load(seg1_name)

            mat = read_gen(flo_name).astype(np.int32)

            # print(seg0.max()+1, seg1.max()+1, mat.shape)

            

            color0, color1 = seg_seq_to_color(seg0, seg1, mat, seg0_name, seg1_name)
            # print(color0.shape)
            color0 = cv2.cvtColor(color0.astype(np.float32), cv2.COLOR_RGB2BGR)
            color1 = cv2.cvtColor(color1.astype(np.float32), cv2.COLOR_RGB2BGR)
            # print(flo.shape)

            # cv2.resize(flo_color)
        
            width = int(color1.shape[1] * 2)
            height = int(color1.shape[0] * 2)
            dim = (width, height)
            color0 = cv2.resize(color0, dim, interpolation=cv2.INTER_NEAREST)
            color1 = cv2.resize(color1, dim, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(osp.join(root_folder, 'visualize', 'Seg_viz2', scene, 'backward', seg0_name.split('/')[-1][:-4] + '.jpg'), color0)
            cv2.imwrite(osp.join(root_folder, 'visualize', 'Seg_viz2', scene, 'backward', seg1_name.split('/')[-1][:-4] + '.jpg'), color1)



# flo = readFlow('/mnt/lustre/syli/AnimeRun/flow/data/CreativeFlow/decompressed/test/eve_j_gonzales.Brutal_Assassination-37-26/cam0/metadata/flow/flow000012.flo')

# flo_color = flow_to_image(flo, convert_to_bgr=True)
# cv2.imwrite('demodemo.png', flo_color)

