from configparser import Interpolation
from genericpath import exists
import numpy as np
import cv2

from core.utils.flow_viz import flow_to_image
from core.utils.frame_utils import readFlow
import os
import os.path as osp
from glob import glob

root_folder = '/mnt/lustre/syli/AnimeRun/flow/data/CreativeFlow/decompressed/test/'

subfolders = ['02871439_de3b28f255111570bc6a557844fbbce9', 'castle_guard_02.Running_Right_Turn-28-22', 'eve_j_gonzales.Standing_To_Crouched-45-16', 'peasant_man.Piano_Playing-47-28']



# if not exists(osp.join(root_folder, 'visualize', 'Flow_viz')):
    # os.mkdir(osp.join(root_folder, 'visualize', 'Flow_viz'))

for subfolder in subfolders:
    root_subfodler = osp.join(root_folder, subfolder)
    print(root_subfodler)
    
    # for scene in os.listdir(osp.join(root_subfodler, 'metadata/flow')):
        # print(scene)

    if not exists(osp.join('/mnt/lustre/syli/AnimeRun/flow/data/AnimeRun1', 'visualize', 'Flow_viz', subfolder)):
        os.mkdir(osp.join('/mnt/lustre/syli/AnimeRun/flow/data/AnimeRun1', 'visualize', 'Flow_viz', subfolder))


    for flo_name in sorted(glob(osp.join(root_subfodler, 'cam0/metadata', 'flow', '*.flo'))):
        print(flo_name)
        flo = readFlow(flo_name)
        # print(flo.shape)
        flo_color = flow_to_image(flo, convert_to_bgr=True)
        # cv2.resize(flo_color)
    
        width = int(flo_color.shape[1] * 2)
        height = int(flo_color.shape[0] * 2)
        dim = (width, height)
        flo_color = cv2.resize(flo_color, dim, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(osp.join('/mnt/lustre/syli/AnimeRun/flow/data/AnimeRun1', 'visualize', 'Flow_viz', subfolder, flo_name.split('/')[-1][:-4] + '.png'), flo_color)



# flo = readFlow('/mnt/lustre/syli/AnimeRun/flow/data/CreativeFlow/decompressed/test/eve_j_gonzales.Brutal_Assassination-37-26/cam0/metadata/flow/flow000012.flo')

# flo_color = flow_to_image(flo, convert_to_bgr=True)
# cv2.imwrite('demodemo.png', flo_color)

