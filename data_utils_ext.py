#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import albumentations as aug
from torch.utils.data import Dataset
import cv2
import logging
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import Resize


def get_default_augmentor():
    list_ops = [aug.Flip(always_apply=True), aug.RandomRotate90(always_apply=True)]
    augmentor = aug.Compose(list_ops)
    return augmentor


def convert_tensor_u8_to_fX(img, div_coef=1., dtype=torch.float32):
    return img.type(dtype) / div_coef


def x_preprocess(x, to_device=None, transpose_chn_last_to_first=True):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if to_device is not None:
        x = x.to(to_device)
    if transpose_chn_last_to_first:
        x = x.permute((0, 3, 1, 2))
    x = convert_tensor_u8_to_fX(x, div_coef=255.)
    return x


def read_rgb_img(path_img: str) -> np.ndarray:
    ret = cv2.imread(path_img)
    ret = cv2.cvtColor(ret, cv2.COLOR_BGR2RGB)
    return ret


def _get_random_seed():
    seed = int(time.time() * 100000) % 10000000 + os.getpid()
    return seed


def worker_init_fn_random(idx):
    seed_ = _get_random_seed() + idx
    torch.manual_seed(seed_)
    np.random.seed(seed_)
    logging.info('\t@@start-worker({}): @pid = [{}]'.format(idx, os.getpid()))



class DatasetExtTrn(Dataset):

    def __init__(self, path_idx: str, crop_lr: int,
                 scale: int, augmentator=get_default_augmentor(),
                 in_memory=False, upscale_lr=False, use_random_crop=True,
                 interpolation=cv2.INTER_CUBIC):
        self.path_idx = path_idx
        self.wdir = os.path.dirname(self.path_idx)
        self.scale = scale
        self.crop_lr = crop_lr
        self.crop_hr = self.crop_lr * self.scale
        self.augmentator = augmentator
        self.in_memory = in_memory
        self.upscale_lr = upscale_lr
        self.interpolation = interpolation
        self.use_random_crop = use_random_crop
        self.data_idx = None
        self.data = None

    def build(self):
        self.data_idx = pd.read_csv(self.path_idx)
        if self.in_memory:
            self.data = []
            pbar = tqdm(total=len(self.data_idx), desc='loading data into memory')
            for irow, row in self.data_idx.iterrows():
                self.data.append(self.load_sample(row))
                pbar.set_description('Processing {}'.format(row['path_hr']))
                pbar.update(irow)
            pbar.close()
        return self

    def random_crop(self, sample: dict) -> dict:
        img_lr, img_hr = sample['lr'], sample['hr']
        nr, nc = img_lr.shape[:2]
        rnd_r_lr = np.random.randint(0, nr - self.crop_lr - 1)
        rnd_c_lr = np.random.randint(0, nc - self.crop_lr - 1)
        rnd_r_hr = rnd_r_lr * self.scale
        rnd_c_hr = rnd_c_lr * self.scale
        img_lr = img_lr[rnd_r_lr:rnd_r_lr + self.crop_lr, rnd_c_lr:rnd_c_lr + self.crop_lr, ...].copy()
        img_hr = img_hr[rnd_r_hr:rnd_r_hr + self.crop_hr, rnd_c_hr:rnd_c_hr + self.crop_hr, ...].copy()
        return {
            'lr': img_lr,
            'hr': img_hr
        }

    def load_sample(self, row):
        path_lr = os.path.join(self.wdir, row['path_lr'])
        path_hr = os.path.join(self.wdir, row['path_hr'])
        path_lr = os.path.join(self.wdir, path_lr)
        path_hr = os.path.join(self.wdir, path_hr)
        img_lr = read_rgb_img(path_lr)
        img_hr = read_rgb_img(path_hr)
        if self.upscale_lr:
            siz_xy = img_hr.shape[:2][::-1]
            img_lr = cv2.resize(img_hr, siz_xy, interpolation=self.interpolation)
        return {'lr': img_lr, 'hr': img_hr}

    def augment_sample(self, aug, sample: dict) -> dict:
        tmp = {'image': sample['lr'], 'mask': sample['hr']}
        tmp = aug(**tmp)
        sample['lr'] = tmp['image']
        sample['hr'] = tmp['mask']
        return sample

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, idx):
        if self.in_memory:
            sample = self.data[idx]
        else:
            sample = self.load_sample(self.data_idx.iloc[idx])
        if self.use_random_crop:
            sample = self.random_crop(sample)
        if self.augmentator is not None:
            sample = self.augment_sample(self.augmentator, sample)
        return sample


def main_run():
    path_idx = '/home/ar/data/debug/high_resolution_data/idx-s2-x4.txt'
    scale = 4
    crop_lr = 32
    dataset = DatasetExtTrn(path_idx=path_idx, crop_lr=crop_lr, scale=scale, in_memory=True).build()
    for xi, x in enumerate(dataset):
        plt.subplot(1, 2, 1)
        plt.imshow(x['lr'])
        plt.title('lr')
        plt.subplot(1, 2, 2)
        plt.imshow(x['hr'])
        plt.title('hr')
        plt.show()

        print('-')


if __name__ == '__main__':

    main_run()

    print('-')
