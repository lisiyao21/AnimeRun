import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(flo):
    # img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    floo = flow_viz.flow_to_image(flo)
    # img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.waitKey()
    return floo

def backwarp(img, flow):
    _, _, H, W = img.size()

    u = flow[:, 0, :, :]
    v = flow[:, 1, :, :]

    gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))

    gridX = torch.tensor(gridX, requires_grad=False).cuda()
    gridY = torch.tensor(gridY, requires_grad=False).cuda()
    x = gridX.unsqueeze(0).expand_as(u).float() + u
    y = gridY.unsqueeze(0).expand_as(v).float() + v
    # range -1 to 1
    x = 2*(x/(W-1) - 0.5)
    y = 2*(y/(H-1) - 0.5)
    # stacking X and Y
    grid = torch.stack((x,y), dim=3)
    # Sample pixels using bilinear interpolation.
    imgOut = torch.nn.functional.grid_sample(img, grid, align_corners=True)

    return imgOut

def save_tensor_to_img(I, name='tmp.png'):
    I = I.data[0].permute(1, 2, 0).cpu().numpy()
    cv2.imwrite(name, I*255)



def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        # build store_path
        store_path = os.path.join(args.path, 'flow_and_warp_rslts')
        if not os.path.exists(store_path):
            os.mkdir(store_path)
            
        
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            print(imfile1)

            # padder = InputPadder(image1.shape)
            # image1, image2 = padder.pad(image1, image2)
            N, C, H, W = image1.size()
            H8 = (H // 8 + 1) * 8
            W8 = (W // 8 + 1) * 8
            image1x = torch.nn.functional.interpolate(image1, size=(H8, W8))
            image2x = torch.nn.functional.interpolate(image2, size=(H8, W8))
            
            flow_low, flow_upx = model(image1x, image2x, iters=12, test_mode=True)

            flow_up = torch.nn.functional.interpolate(flow_upx, (H, W))
            flow_up[:, 0] *= (H/(H8*1.0))
            flow_up[:, 1] *= (W/(W8*1.0))
            print(len(flow_up))

            I2_warp = backwarp(image2.cuda(), flow_up.cuda())
            tmp = I2_warp[:, 0].clone()
            I2_warp[:, 0] = I2_warp[:, 2].clone()
            I2_warp[:, 2] = tmp

            save_tensor_to_img(I2_warp[:, :, :, :]/255.0, os.path.join(store_path, imfile1.split('/')[-1][:-4] + '_next_warp_this.png'))
            flo_img = viz(flow_up)
            cv2.imwrite(os.path.join(store_path, imfile1.split('/')[-1][:-4] + '_flo.png'), flo_img)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
