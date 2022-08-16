import numpy as np
import random
import math
from PIL import Image

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
from torchvision.transforms import ColorJitter
import torch.nn.functional as F
from collections import Counter

class SegMatAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True):
        
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def spatial_transform(self, img1, img2, seg1, seg2, matching):
        # augmentation should contain crop, shift, scaling
        # when croped: seg index need to be changed to [0..Nc], lost item mismatching
        # when scaled: seg index need to be rescaled. if larger, doesn't matter, if smaller, index may vanish. delete it and reindex as above
        # when shift: doesn't matter
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht), 
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        
        x = np.random.rand()
        # print(x, x, x, x, x)
        if x < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            
            seg1 = cv2.resize(seg1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST)
            seg2 = cv2.resize(seg2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST)

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob: # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                seg1 = seg1[:, ::-1]
                seg2 = seg2[:, ::-1]

            if np.random.rand() < self.v_flip_prob: # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                seg1 = seg1[::-1, :]
                seg2 = seg2[::-1, :]

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
        x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])
        
        img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        seg1 = seg1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        seg2 = seg2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        seg1_id_hist = Counter(seg1.reshape(-1)).most_common()
        seg2_id_hist = Counter(seg2.reshape(-1)).most_common()

        # cc_5 = 0
        # for ii, jj in seg1_id_hist:
        #     if jj < 10:
        #         cc_5 += 1
        # print('# < 5 is ', cc_5)

        seg1_id = np.array([ii for (ii, _) in seg1_id_hist])
        seg2_id = np.array([ii for (ii, _) in seg2_id_hist])

        seg1_new = seg1.copy()
        seg2_new = seg2.copy()
        for ii in range(len(seg1_id)):
            seg1_new[seg1 == seg1_id[ii]] = ii 
        for ii in range(len(seg2_id)):
            seg2_new[seg2 == seg2_id[ii]] = ii

        matching_new = np.ones([len(seg1_id), ]) * -1
        for ii in range(len(seg1_id)):
            if matching[seg1_id[ii]] in seg2_id:
                matching_new[ii] = np.where(seg2_id == matching[seg1_id[ii]])[0][0] 

        return img1, img2, seg1_new, seg2_new, matching_new

    def __call__(self, img1, img2, seg1, seg2, matching):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, seg1, seg2, matching = self.spatial_transform(img1, img2, seg1, seg2, matching)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        seg1 = np.ascontiguousarray(seg1)
        seg2 = np.ascontiguousarray(seg2)
        matching = np.ascontiguousarray(matching)

        return img1, img2, seg1, seg2, matching
