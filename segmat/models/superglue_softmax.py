# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#  Edited: Siyao Li 
# 
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
# from seg_desc import seg_descriptor
from torch_scatter import scatter as super_pixel_pooling
import argparse

def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                # layers.append(nn.BatchNorm1d(channels[i]))
                layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*width, one*height, one*height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    # print(kpts.size(), center[:, None, :].size(), scaling[:, None, :].size())
    return (kpts - center[:, None, :]) / scaling[:, None, :]

class ThreeLayerEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, enc_dim):
        super().__init__()
        # input must be 3 channel (r, g, b)
        self.layer1 = nn.Conv2d(3, enc_dim//4, 7, padding=3)
        self.non_linear1 = nn.ReLU()
        self.layer2 = nn.Conv2d(enc_dim//4, enc_dim//2, 3, padding=1)
        self.non_linear2 = nn.ReLU()
        self.layer3 = nn.Conv2d(enc_dim//2, enc_dim, 3, padding=1)

        self.norm1 = nn.InstanceNorm2d(enc_dim//4)
        self.norm2 = nn.InstanceNorm2d(enc_dim//2)
        self.norm3 = nn.InstanceNorm2d(enc_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)

    def forward(self, img):
        x = self.non_linear1(self.norm1(self.layer1(img)))
        x = self.non_linear2(self.norm2(self.layer2(x)))
        x = self.norm3(self.layer3(x))
        # x = self.non_linear1(self.layer1(img))
        # x = self.non_linear2(self.layer2(x))
        # x = self.layer3(x)
        return x


class SegmentDescriptor(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, enc_dim):
        super().__init__()
        self.encoder = ThreeLayerEncoder(enc_dim)
        # self.super_pixel_pooling = 
        # use scatter
        # nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, img, seg):
        x = self.encoder(img)
        n, c, h, w = x.size()
        assert((h, w) == img.size()[2:4])
        return super_pixel_pooling(x.view(n, c, -1), seg.view(-1).long(), reduce='mean')
        # here return size is [1]xCx|Seg|


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([4] + layers + [feature_dim])
        # for m in self.encoder.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         nn.init.constant_(m.bias, 0.0)
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts):
        inputs = kpts.transpose(1, 2)
        # print(inputs.size(), 'wula!')
        x = self.encoder(inputs)
        # print(x.size())
        return x


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, prob = attention(query, key, value)
        self.prob.append(prob)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0, desc1):
        for layer, name in zip(self.layers, self.names):
            layer.attn.prob = []
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    # pad additional scores for unmatcheed (to -1)
    # alpha is the learned threshold
    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z

def transport(scores, alpha):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    # pad additional scores for unmatcheed (to -1)
    # alpha is the learned threshold
    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    return couplings



def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class SuperGlueSoftMax(nn.Module):
    """SuperGlue feature matching middle-end

    Given two sets of keypoints and locations, we determine the
    correspondences by:
      1. Keypoint Encoding (normalization + visual feature and location fusion)
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold

    The correspondence ids use -1 to indicate non-matching points.

    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
    Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763

    """
    # default_config = {
    #     'descriptor_dim': 128,
    #     'weights': 'indoor',
    #     'keypoint_encoder': [32, 64, 128],
    #     'GNN_layers': ['self', 'cross'] * 9,
    #     'sinkhorn_iterations': 100,
    #     'match_threshold': 0.2,
    # }

    def __init__(self, config=None):
        super().__init__()

        default_config = argparse.Namespace()
        default_config.descriptor_dim = 128
        # default_config.weights = 
        default_config.keypoint_encoder = [32, 64, 128]
        default_config.GNN_layers = ['self', 'cross'] * 9
        default_config.sinkhorn_iterations = 100
        default_config.match_threshold = 0.2
        # self.config = {**self.default_config, **config}

        if config is None:
            self.config = default_config
        else:
            self.config = config   
            self.config.GNN_layers = ['self', 'cross'] * self.config.GNN_layer_num
            # print('WULA!', self.config.GNN_layer_num)

        self.kenc = KeypointEncoder(
            self.config.descriptor_dim, self.config.keypoint_encoder)

        self.gnn = AttentionalGNN(
            self.config.descriptor_dim, self.config.GNN_layers)

        self.final_proj = nn.Conv1d(
            self.config.descriptor_dim, self.config.descriptor_dim,
            kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)
        self.segment_desc = SegmentDescriptor(self.config.descriptor_dim)

        # assert self.config.weights in ['indoor', 'outdoor']
        # path = Path(__file__).parent
        # path = path / 'weights/superglue_{}.pth'.format(self.config.weights)
        # self.load_state_dict(torch.load(path))
        # print('Loaded SuperGlue model (\"{}\" weights)'.format(
        #     self.config.weights))

    def forward(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        # print(data['segment0'].size())
        # desc0, desc1 = data['descriptors0'].float()(), data['descriptors1'].float()()
        desc0, desc1 = self.segment_desc(data['image0'], data['segment0']), self.segment_desc(data['image1'], data['segment1'])
        # print(desc0.size())
        kpts0, kpts1 = data['keypoints0'].float(), data['keypoints1'].float()

        # desc0 = desc0.transpose(0,1)
        # desc1 = desc1.transpose(0,1)
        # kpts0 = torch.reshape(kpts0, (1, -1, 2))
        # kpts1 = torch.reshape(kpts1, (1, -1, 2))

        
    
        if kpts0.shape[1] < 2 or kpts1.shape[1] < 2:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            # print(data['file_name'])
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int)[0],
                # 'matches1': kpts1.new_full(shape1, -1, dtype=torch.int)[0],
                'matching_scores0': kpts0.new_zeros(shape0)[0],
                # 'matching_scores1': kpts1.new_zeros(shape1)[0],
                'skip_train': True
            }

        file_name = data['file_name']
        all_matches = data['all_matches'] if 'all_matches' in data else None# shape = (1, K1)
        # .permute(1,2,0) # shape=torch.Size([1, 87,])
        
        # positional embedding
        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
        kpts1 = normalize_keypoints(kpts1, data['image1'].shape)

        # Keypoint MLP encoder.
        # print(data['file_name'])
        # print(kpts0.size())
    
        pos0 = self.kenc(kpts0)
        pos1 = self.kenc(kpts1)
        # print(desc0.size(), pos0.size())
        desc0 = desc0 + pos0
        desc1 = desc1 + pos1

        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1)

        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        # Compute matching descriptor distance.
        # print(mdesc0.size(), mdesc1.size())
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        # #print('here1!!', scores.size())

        # b k1 k2
        scores = scores / self.config.descriptor_dim**.5
        # print(scores.size())
        # Run the optimal transport.
        b, m, n = scores.size()
    
        # print(scores)
        scores = transport(scores, self.bin_score)

        weights = data['num0'].float().cuda()
        avg_w = weights.mean()
        all_matches_origin = all_matches.clone() if all_matches is not None else None

        # print(all_matches)
        if all_matches is not None:
            all_matches[all_matches == -1] = n
            loss = nn.functional.cross_entropy(scores[:, :-1, :].view(-1, n+1), all_matches.long().view(-1), reduction='mean')
            loss = loss.mean()

        # print(scores)
        # print(scores.sum())
        # print(scores.sum(1))
        # print(scores.sum(0))

        # Get the matches with score above "match_threshold".
        scores = nn.functional.softmax(scores, dim=2)
        # print(scores)
        max0, max1 = scores[:, :-1, :].max(2), scores[:, :, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        # mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        # mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        # zero = scores.new_tensor(0)
        mscores0 = max0.values
        # mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)[:, :-1]
        valid0 = indices0 < n
        valid1 = indices1 < m
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        # print(indices0)
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        # check if indexed correctly
        # #print(scores.size())

        # print(weights)

        
        
        # #print(scores.size())
        if all_matches is not None:
            return {
                'matches0': indices0[0], # use -1 for invalid match
                # 'matches1': indices1[0], # use -1 for invalid match
                'matching_scores0': mscores0[0],
                # 'matching_scores1': mscores1[0],
                'loss': loss,
                'skip_train': False,
                'accuracy': ((all_matches_origin[0] == indices0[0]).sum() / len(all_matches_origin[0])).item(),
                'area_accuracy': (torch.tensor([ (data['segment0'] == ii).sum() for ii in torch.arange(0, all_matches_origin[0].shape[0])[all_matches_origin[0] == indices0[0]]]).sum() / (weights.sum() * 1.0)).item(),
                'valid_accuracy': (((all_matches_origin[0] == indices0[0]) & (all_matches_origin[0] != -1)).sum() / (all_matches_origin[0] != -1).sum()).item(),
                'invalid_accuracy': (((all_matches_origin[0] == indices0[0]) & (all_matches_origin[0] == -1)).sum() / (all_matches_origin[0] == -1).sum()).item() if (all_matches_origin[0] == -1).sum() > 0 else None
            }
        else:
            return {
                'matches0': indices0[0], # use -1 for invalid match
                'matching_scores0': mscores0[0],
                'loss': -1,
                'skip_train': True,
                'accuracy': -1,
                'area_accuracy': -1,
                'valid_accuracy': -1,
            }


if __name__ == '__main__':
    from anime_seg_mat_dataset import fetch_dataloader
    args = argparse.Namespace()
    ss = SuperGlue()
    args.subset = 'trytry'
    args.batch_size = 1
    args.stage = 'anime'
    args.image_size = (368, 368)
    loader = fetch_dataloader(args)
    # #print(len(loader))
    for data in loader:
        # p1, p2, s1, s2, mi = data
        dict1 = data

        kp1 = dict1['keypoints0']
        kp2 = dict1['keypoints1']
        p1 = dict1['image0']
        p2 = dict1['image1']  
        s1 = dict1['segment0']
        s2 = dict1['segment1']
        # #print(s1)
        # #print(s1.type)
        mi = dict1['all_matches']
        fname = dict1['file_name']   
        # #print(mi.size())  
        # #print(mi)

        a = ss(data)
        #print(dict1['file_name'])
        # print(a['loss'])
        # a['loss'].backward()
        # print(a['matches0'].size())
        # print(a['accuracy'], a['valid_accuracy'])