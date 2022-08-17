""" This script handling the training process. """
import os
import time
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from datasets import fetch_dataloader
import random
from utils.log import Logger

from torch.optim import *
import warnings
from tqdm import tqdm
import itertools
import pdb
import numpy as np
import models
import datetime
import sys

import matplotlib.cm as cm
from models.utils import make_matching_seg_plot

warnings.filterwarnings('ignore')

# a, b, c, d = check_data_distribution('/mnt/lustre/lisiyao1/dance/dance2/DanceRevolution/data/aistpp_train')

import matplotlib.pyplot as plt
import pdb

class SGM():
    def __init__(self, args):
        self.config = args
        torch.backends.cudnn.benchmark = True
        self._build()

    def train(self):
        opt = self.config
        print(opt)

        # store viz results
        # eval_output_dir = Path(self.expdir)
        # eval_output_dir.mkdir(exist_ok=True, parents=True)

        # print('Will write visualization images to',
        #     'directory \"{}\"'.format(eval_output_dir))

        # load training data
        
        model = self.model

        # if torch.cuda.is_available():
        #     model.cuda() # make sure it trains on GPU
        # else:
        #     print("### CUDA not available ###")
            # return
        optimizer = self.optimizer
        schedular = self.schedular
        mean_loss = []
        log = Logger(self.config, self.expdir)
        updates = 0
        
        # set seed
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        # print(opt.seed)
        # start training
        for epoch in range(1, opt.epoch+1):
            np.random.seed(opt.seed + epoch)
            train_loader = self.train_loader
            log.set_progress(epoch, len(train_loader))
            batch_loss = 0
            batch_acc = 0 
            batch_area_acc = 0
            batch_valid_acc = 0
            batch_iter = 0
            model.train()
            for i, pred in enumerate(train_loader):
                # print(pred['file_name'])
                data = model(pred)
                # print(len(pred['all_matches'][0]), 'wula!       \n')
                # if len(pred['all_matches'][0]) == 1:
                #     continue
                for k, v in pred.items():
                    pred[k] = v[0]
                pred = {**pred, **data}

                # if pred['skip_train'] == True: # image has no keypoint
                #     continue
                # process loss
                # TODO Accumulate loss!

                if not pred['skip_train']:
                    loss = pred['loss'] / opt.batch_size
                    batch_loss += loss.item()
                    batch_acc += pred['accuracy'] 
                    batch_area_acc += pred['area_accuracy'] 
                    batch_valid_acc += pred['valid_accuracy'] 
                    loss.backward()
                    batch_iter += 1
                else:
                    print('Skip!')

                # pdb.set_trace()
                if ((i + 1) % opt.batch_size == 0) or (i + 1 == len(train_loader)):
                    optimizer.step()
                    optimizer.zero_grad()
                    batch_iter = 1 if batch_iter == 0 else batch_iter               
                    stats = {
                        'updates': updates,
                        'loss': batch_loss,
                        'accuracy': batch_acc / batch_iter,
                        'area_accuracy':batch_area_acc / batch_iter,
                        'valid_accuracy': batch_valid_acc / batch_iter
                    }
                    log.update(stats)
                    updates += 1
                    batch_loss = 0
                    batch_acc = 0 
                    batch_area_acc = 0
                    batch_valid_acc = 0
                    batch_iter = 0
                # for name, params in model.named_parameters():
                #     print('-->name:, ', name, '-->grad mean', params.grad.mean())

            # save checkpoint 
            if epoch % opt.save_per_epochs == 0 or epoch == 1:
                checkpoint = {
                    'model': model.state_dict(),
                    'config': opt,
                    'epoch': epoch
                }

                filename = os.path.join(self.ckptdir, f'epoch_{epoch}.pt')
                torch.save(checkpoint, filename)
                
            # validate
            if epoch % opt.test_freq == 0:

                if not os.path.exists(os.path.join(self.visdir, 'epoch' + str(epoch))):
                    os.mkdir(os.path.join(self.visdir, 'epoch' + str(epoch)))
                eval_output_dir = os.path.join(self.visdir, 'epoch' + str(epoch))    
                
                test_loader = self.test_loader

                with torch.no_grad():
                    # Visualize the matches.
                    mean_acc = []
                    mean_valid_acc = []
                    model.eval()
                    for i_eval, pred in enumerate(tqdm(test_loader, desc='Generating Dance Poses')):
                        data = model(pred)
                        for k, v in pred.items():
                            pred[k] = v[0]
                            pred = {**pred, **data}
                        image0, image1 = (pred['image0'].cpu().numpy()[0] + 1)*255.*0.5, (pred['image1'].cpu().numpy()[0] + 1)*255.*0.5
                        seg0, seg1 = pred['segment0'].data.cpu().numpy()[0], pred['segment1'].data.cpu().numpy()[0]
                        kpts0, kpts1 = pred['center_points0'].cpu().numpy(), pred['center_points1'].cpu().numpy()
                        # matches = pred['matches0'].cpu().detach().numpy()
                        matches, conf = pred['matches0'].long().cpu().detach().numpy(), pred['matching_scores0'].cpu().detach().numpy()
                
                    # pred['matching_scores0'].cpu().detach().numpy()
                    # image0 = read_image_modified(image0, opt.resize, opt.resize_float)
                    # image1 = read_image_modified(image1, opt.resize, opt.resize_float)
                        valid = matches > -1
                        # print(kpts0.shape, valid.shape)
                        mkpts0 = kpts0[valid]
                        mkpts1 = kpts1[matches[valid]]
                        mconf = conf[valid]
                        viz_path = eval_output_dir + '/{}_matches.{}'.format(pred['file_name'].replace('/', '_'), '.jpg')
                        viz_path_gt = eval_output_dir + '/{}_matches.{}'.format(pred['file_name'].replace('/', '_'), '_gt.jpg')
                        color = cm.jet(mconf)
                        stem = pred['file_name'].replace('/', '_')
                        text = []
                        mean_acc.append(pred['accuracy'])
                        mean_valid_acc.append(pred['valid_accuracy'])

                        make_matching_seg_plot(seg0, seg1, matches, viz_path, np.round(pred['accuracy'], 2))
                    log.log_eval({
                        'updates': opt.epoch,
                        'Accuracy': np.mean(mean_acc),
                        'Valid Accuracy': np.mean(mean_valid_acc),
                        })
                    print('Epoch [{}/{}]], Acc.: {:.4f}, Valid Acc.{:.4f}' 
                        .format(epoch, opt.epoch, np.mean(mean_acc), np.mean(mean_valid_acc)) )
                    sys.stdout.flush()
                        # make_matching_plot(
                        #     image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                        #     text, viz_path, stem, stem, True,
                        #     True, False, 'Matches')
                        

            self.schedular.step()
            # for param_group in self.optimizer.param_groups:
            #     print(param_group['lr'])

                    # process checkpoint for every 5e3 images
                    # if (i+1) % 5e3 == 0:
                    #     model_out_path = "model_epoch_{}.pth".format(epoch)
                    #     torch.save(superglue, model_out_path)
                    #     print ('Epoch [{}/{}], Step [{}/{}], Checkpoint saved to {}' 
                    #         .format(epoch, opt.epoch, i+1, len(train_loader), model_out_path)) 

            # save checkpoint when an epoch finishes
            # epoch_loss /= len(train_loader)
            # model_out_path = "model_epoch_{}.pth".format(epoch)
            # torch.save(superglue, model_out_path)
            # print("Epoch [{}/{}] done. Epoch Loss {}. Checkpoint saved to {}"
            #     .format(epoch, opt.epoch, epoch_loss, model_out_path))
            


    def eval(self):
        log = Logger(self.config, self.expdir)
        with torch.no_grad():
            model = self.model.eval()
            config = self.config
            epoch_tested = self.config.testing.ckpt_epoch
            ckpt_path = os.path.join(self.ckptdir, f"epoch_{epoch_tested}.pt")
            # self.device = torch.device('cuda' if config.cuda else 'cpu')
            print("Evaluation...")
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint['model'])
            model.eval()

            if not os.path.exists(os.path.join(self.evaldir, 'epoch' + str(epoch_tested))):
                os.mkdir(os.path.join(self.evaldir, 'epoch' + str(epoch_tested)))
            eval_output_dir = os.path.join(self.evaldir, 'epoch' + str(epoch_tested))    
                
            test_loader = self.test_loader
            print(len(test_loader))
            mean_acc = []
            mean_valid_acc = []
            mean_invalid_acc = []
            mean_large300_acc = []

            for i_eval, pred in enumerate(tqdm(test_loader, desc='Generating Dance Poses')):
                # if i_eval == 34:
                #     continue
                data = model(pred)
                for k, v in pred.items():
                    pred[k] = v[0]
                    pred = {**pred, **data}
                image0, image1 = (pred['image0'].cpu().numpy()[0] + 1)*255.*0.5, (pred['image1'].cpu().numpy()[0] + 1)*255.*0.5
                seg0, seg1 = pred['segment0'].data.cpu().numpy()[0], pred['segment1'].data.cpu().numpy()[0]
                kpts0, kpts1 = pred['center_points0'].cpu().numpy(), pred['center_points1'].cpu().numpy()
                # matches = pred['matches0'].cpu().detach().numpy()
                matches, conf = pred['matches0'].long().cpu().detach().numpy(), pred['matching_scores0'].cpu().detach().numpy()
                matches_gt = pred['all_matches'].long().cpu().detach().numpy()
            # pred['matching_scores0'].cpu().detach().numpy()
            # image0 = read_image_modified(image0, opt.resize, opt.resize_float)
            # image1 = read_image_modified(image1, opt.resize, opt.resize_float)
                valid = matches > -1
                # print(kpts0.shape, valid.shape)
                mkpts0 = kpts0[valid]
                mkpts1 = kpts1[matches[valid]]
                mconf = conf[valid]
                viz_path = eval_output_dir + '/{}_matches.{}'.format(pred['file_name'].replace('/', '_'), '.jpg')
                viz_path_gt = eval_output_dir + '/{}_matches.{}'.format(pred['file_name'].replace('/', '_'), '_gt.jpg')
                
                color = cm.jet(mconf)
                stem = pred['file_name'].replace('/', '_')
                text = []
                
                mean_acc.append(pred['accuracy'])
                mean_valid_acc.append(pred['valid_accuracy'])
                if pred['invalid_accuracy'] is not None:
                    mean_invalid_acc.append(pred['invalid_accuracy'])
                if seg1.max() > 300:
                    mean_large300_acc.append(pred['accuracy'])

                # make_matching_plot(
                #     image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                #     text, viz_path, stem, stem, True,
                #     True, False, 'Matches')
                make_matching_seg_plot(seg0, seg1, matches, viz_path, np.round(pred['accuracy'], 2))
                make_matching_seg_plot(seg0, seg1, matches_gt, viz_path_gt)

            log.log_eval({
                'updates': self.config.testing.ckpt_epoch,
                'Accuracy': np.mean(mean_acc),
                'Valid Accuracy': np.mean(mean_valid_acc),
                'Invalid Accuracy': np.mean(mean_invalid_acc),
                '>300 Accuracy': np.mean(mean_large300_acc),
                })
                # print ('Epoch [{}/{}]], Acc.: {:.4f}, Valid Acc.{:.4f}' 
                #     .format(epoch, opt.epoch, np.mean(mean_acc), np.mean(mean_valid_acc)) )
            sys.stdout.flush()

    def _build(self):
        config = self.config
        self.start_epoch = 0
        self._dir_setting()
        self._build_model()
        if not(hasattr(config, 'need_not_train_data') and config.need_not_train_data):
            self._build_train_loader()
        if not(hasattr(config, 'need_not_test_data') and config.need_not_train_data):      
            self._build_test_loader()
        self._build_optimizer()

    def _build_model(self):
        """ Define Model """
        config = self.config 
        if hasattr(config.model, 'name'):
            print(f'Experiment Using {config.model.name}')
            model_class = getattr(models, config.model.name)
            model = model_class(config.model)
        else:
            raise NotImplementedError("Wrong Model Selection")
        
        model = nn.DataParallel(model)
        self.model = model.cuda()

    def _build_train_loader(self):
        config = self.config
        self.train_loader = fetch_dataloader(config.data.train, type='training')

    def _build_test_loader(self):
        config = self.config
        self.test_loader = fetch_dataloader(config.data.test, type='test')

    def _build_optimizer(self):
        #model = nn.DataParallel(model).to(device)
        config = self.config.optimizer
        try:
            optim = getattr(torch.optim, config.type)
        except Exception:
            raise NotImplementedError('not implemented optim method ' + config.type)

        self.optimizer = optim(itertools.chain(self.model.module.parameters(),
                                             ),
                                             **config.kwargs)
        self.schedular = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, **config.schedular_kwargs)

    def _dir_setting(self):
        data = self.config.data
        self.expname = self.config.expname
        self.experiment_dir = os.path.join(".", "experiments")
        self.expdir = os.path.join(self.experiment_dir, self.expname)

        if not os.path.exists(self.expdir):
            os.mkdir(self.expdir)

        self.visdir = os.path.join(self.expdir, "vis")  # -- imgs, videos, jsons
        if not os.path.exists(self.visdir):
            os.mkdir(self.visdir)

        self.ckptdir = os.path.join(self.expdir, "ckpt")
        if not os.path.exists(self.ckptdir):
            os.mkdir(self.ckptdir)

        self.evaldir = os.path.join(self.expdir, "eval")
        if not os.path.exists(self.evaldir):
            os.mkdir(self.evaldir)

        

        # self.ckptdir = os.path.join(self.expdir, "ckpt")
        # if not os.path.exists(self.ckptdir):
        #     os.mkdir(self.ckptdir)



        




