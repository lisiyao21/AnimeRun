import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os.path as osp
import datasets
from utils import flow_viz
from utils import frame_utils

from gmflow.gmflow import GMFlow
from utils.utils import InputPadder, forward_interpolate
import cv2

@torch.no_grad()
def create_sintel_submission(model, iters=32, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)
        
        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None
            
            padder = InputPadder(image1.shape, padding_factor=32)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()
            
            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_chairs(model, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}


@torch.no_grad()
def validate_sintel(model, iters=32):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)
        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_kitti(model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


@torch.no_grad()
def validate_anime(model, path=None, iters=32):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['Frame_Anime']:
        val_dataset = datasets.AnimeRun(split='test', dstype=dstype)
        epe_list = []
        epe_flat_list = []
        epe_line_list = []
        epe_nonocc_list = []
        epe_occ_list = []
        epe_ds10_list = []
        epe_ds1050_list = []
        epe_ds50_list = []


        for val_id in range(len(val_dataset)):
            # if val_id % 100 == 0:
            # print('Testing the {}-th image pair...'.format(val_id))
            image1, image2, flow_gt, occ, flat, extra_info, valid = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()



            padder = InputPadder(image1.shape, padding_factor=32)
            image1, image2 = padder.pad(image1, image2)


            # flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            results_dict = model(image1, image2,
                        attn_splits_list=[2, 8],
                        corr_radius_list=[-1, 4],
                        prop_radius_list=[-1, 1],
                        )

            flow = padder.unpad(results_dict['flow_preds'][-1][0]).cpu()

            mag =torch.sum(flow_gt ** 2, dim=0).sqrt()
            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            

            flo_gt_ = flow_gt.clone()

            valid = valid.bool()

            # print((~valid).size(), flo_gt_[0, :, :].size())
            flo_gt_[0, :, :][~valid] = 0
            flo_gt_[1, :, :][~valid] = 0
            if epe[valid].numpy().mean() > 50:
                print(extra_info[0], extra_info[1], epe[valid].numpy().mean(), flow_gt.abs().max(), (flow_gt.abs() < 0.01).sum())
            
            if path is not None:
                flo_color = flow_viz.flow_to_image(flow.permute(1, 2, 0).data.numpy(), convert_to_bgr=True)
                flo_color = cv2.putText(flo_color, 'EPE = ' + str(np.round(epe[valid].numpy().mean(), 2)), (50, 50),  cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=2, color=(0, 0, 255), thickness=3)
                flo_gt_color = flow_viz.flow_to_image(flo_gt_.permute(1, 2, 0).data.numpy(), convert_to_bgr=True)
                
                cv2.imwrite(osp.join(path, str(extra_info[0] + '_' + str(extra_info[1]) + '.jpg')), flo_color)
                cv2.imwrite(osp.join(path, str(extra_info[0] + '_' + str(extra_info[1]) + '_gt.jpg')), flo_gt_color)
            
            epe_nonocc = epe[(occ == 1) & valid]
            epe_flat = epe[(flat > 0) & valid ]
            epe_line = epe[(flat == 0) & valid ]
            epe_occ = epe[(occ == 0) & valid]
            epe_ds10 = epe[(mag <= 10.0) & valid]
            epe_ds1050 = epe[(mag > 10.0) & (mag <= 50.0) & valid]
            epe_ds50 = epe[(mag > 50.0) & valid]

            epe_list.append(epe[valid].view(-1).numpy())
            epe_flat_list.append(epe_flat.view(-1).numpy())
            epe_line_list.append(epe_line.view(-1).numpy())
            epe_nonocc_list.append(epe_nonocc.view(-1).numpy())
            epe_occ_list.append(epe_occ.view(-1).numpy())
            epe_ds10_list.append(epe_ds10.view(-1).numpy())
            epe_ds1050_list.append(epe_ds1050.view(-1).numpy())
            epe_ds50_list.append(epe_ds50.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe_flat_all = np.concatenate(epe_flat_list)
        epe_line_all = np.concatenate(epe_line_list)
        epe_nonocc_all = np.concatenate(epe_nonocc_list)
        epe_occ_all = np.concatenate(epe_occ_list)
        epe_ds10_all = np.concatenate(epe_ds10_list)
        epe_ds1050_all = np.concatenate(epe_ds1050_list)
        epe_ds50_all = np.concatenate(epe_ds50_list)

        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        # print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        print("Test (%s) EPE: %f, Occ: %f, Non-Occ: %f, Line: %f, Flat: %f, s<10: %f, s10-50: %f, s>50: %f" % (dstype, epe, np.mean(epe_occ_all), np.mean(epe_nonocc_all), np.mean(epe_line_all), np.mean(epe_flat_all), np.mean(epe_ds10_all), np.mean(epe_ds1050_all), np.mean(epe_ds50_all)))
        results[dstype] = epe

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    
    parser.add_argument('--num_scales', default=2, type=int,
                        help='basic gmflow model uses a single 1/8 feature, the refinement uses 1/4 feature')
    # parser.add_argument('--num_heads', default=1, type=int,
    #                 help='number of heads in attention and aggregation')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--upsample_factor', default=4, type=int)
    parser.add_argument('--num_head', default=1, type=int)

    parser.add_argument('--ffn_dim_expansion', default=4, type=int)
    parser.add_argument('--attention_type', default='swin', type=str)
    parser.add_argument('--num_transformer_layers', default=6, type=int)
    parser.add_argument('--feature_channels', default=128, type=int)

    args = parser.parse_args()

    model = GMFlow(feature_channels=args.feature_channels,
                num_scales=args.num_scales,
                upsample_factor=args.upsample_factor,
                num_head=args.num_head,
                attention_type=args.attention_type,
                ffn_dim_expansion=args.ffn_dim_expansion,
                num_transformer_layers=args.num_transformer_layers,
                )
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model))
    
    

    

    model.cuda()
    model.eval()

    # create_sintel_submission(model.module, warm_start=True)
    # create_kitti_submission(model.module)
    print('Testing {}'.format(args.model))
    if not osp.exists('visualize/' + args.model.replace('/', '_')):
        os.mkdir('visualize/' + args.model.replace('/', '_'))
    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model.module)

        elif args.dataset == 'sintel':
            validate_sintel(model.module)

        elif args.dataset == 'anime':
            validate_anime(model.module, path='visualize/' + args.model.replace('/', '_'))

        elif args.dataset == 'kitti':
            validate_kitti(model.module)


