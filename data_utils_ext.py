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
from albumentations import Compose
from torch.utils.data import Dataset
import cv2
import logging
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import Resize
from typing import Optional as O


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


def random_crop(sample: dict, scale: int, crop_lr: int) -> dict:
    crop_hr = crop_lr * scale
    img_lr, img_hr = sample['lr'], sample['hr']
    nr, nc = img_lr.shape[:2]
    rnd_r_lr = np.random.randint(0, nr - crop_lr - 1)
    rnd_c_lr = np.random.randint(0, nc - crop_lr - 1)
    rnd_r_hr = rnd_r_lr * scale
    rnd_c_hr = rnd_c_lr * scale
    img_lr = img_lr[rnd_r_lr:rnd_r_lr + crop_lr, rnd_c_lr:rnd_c_lr + crop_lr, ...].copy()
    img_hr = img_hr[rnd_r_hr:rnd_r_hr + crop_hr, rnd_c_hr:rnd_c_hr + crop_hr, ...].copy()
    ret = {
        'lr': img_lr,
        'hr': img_hr
    }
    if 'lr_up' in sample:
        img_lr_up = sample['lr_up'][rnd_r_hr:rnd_r_hr + crop_hr, rnd_c_hr:rnd_c_hr + crop_hr, ...].copy()
        ret['lr_up'] = img_lr_up
    return ret


def load_sample(wdir: str, row, upscale_lr: bool, interpolation) -> dict:
    path_lr = os.path.join(wdir, row['path_lr'])
    path_hr = os.path.join(wdir, row['path_hr'])
    path_lr = os.path.join(wdir, path_lr)
    path_hr = os.path.join(wdir, path_hr)
    img_lr = read_rgb_img(path_lr)
    img_hr = read_rgb_img(path_hr)
    ret = {'lr': img_lr, 'hr': img_hr}
    if upscale_lr:
        siz_xy = img_hr.shape[:2][::-1]
        img_lr_up = cv2.resize(img_lr, siz_xy, interpolation=interpolation)
        ret['lr_up'] = img_lr_up
    return ret


def load_idx_into_memory(data_idx: pd.DataFrame, wdir: str, upscale_lr, interpolation, scale_factor: int = None) -> list:
    data = []
    pbar = tqdm(total=len(data_idx), desc='loading data into memory')
    for irow, row in data_idx.iterrows():
        sample = load_sample(wdir, row, upscale_lr, interpolation=interpolation)
        if scale_factor is not None:
            sample = crop_sample_by_scale_factor(sample, scale_factor)
        data.append(sample)
        pbar.set_description('Processing {}'.format(row['path_hr']))
        pbar.update(irow)
    pbar.close()
    return data


def crop_img_by_scale_factor(img: np.ndarray, scale_factor: int = 2**5) -> np.ndarray:
    nrc = np.array(img.shape[:2])
    nrc_fix = (scale_factor * np.floor(nrc / scale_factor)).astype(np.int)
    ret = img[:nrc_fix[0], :nrc_fix[1], ...]
    return ret


def crop_sample_by_scale_factor(sample: dict, scale_factor: int) -> dict:
    ret = {k: crop_img_by_scale_factor(v, scale_factor) for k, v in sample.items()}
    return ret


class DatasetExtTrn(Dataset):

    def __init__(self, path_idx: str, crop_lr: int,
                 scale: int, augmentator: O[Compose] = get_default_augmentor(),
                 in_memory=False, upscale_lr=False, use_random_crop=True,
                 interpolation=cv2.INTER_CUBIC, crop_scale_factor: int = None):
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
        self.crop_scale_factor = crop_scale_factor

    def build(self):
        self.data_idx = pd.read_csv(self.path_idx)
        if self.in_memory:
            self.data = load_idx_into_memory(
                self.data_idx, wdir=self.wdir,
                upscale_lr=self.upscale_lr,
                interpolation=self.interpolation,
                scale_factor=self.crop_scale_factor
            )
        return self

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
            sample = load_sample(self.wdir, self.data_idx.iloc[idx],
                                 upscale_lr=self.upscale_lr,
                                 interpolation=self.interpolation)
            if self.crop_scale_factor is not None:
                sample = crop_sample_by_scale_factor(sample, scale_factor=self.crop_scale_factor)
        if self.use_random_crop:
            sample = random_crop(sample, scale=self.scale, crop_lr=self.crop_lr)
        if self.augmentator is not None:
            sample = self.augment_sample(self.augmentator, sample)
        return sample


class DatasetExtVal(DatasetExtTrn):

    def __init__(self, path_idx: str, crop_lr: int, scale: int,
                 in_memory=False, interpolation=cv2.INTER_CUBIC, crop_scale_factor: int = 2 ** 5):
        super().__init__(path_idx,
                         crop_lr, scale,
                         in_memory=in_memory,
                         upscale_lr=True,
                         interpolation=interpolation,
                         augmentator=None,
                         use_random_crop=False,
                         crop_scale_factor=crop_scale_factor)


def main_run():
    # path_idx = '/home/ar/data/debug/high_resolution_data/idx-s2-x4.txt'
    path_idx = '/home/ar/data/debug/high_resolution_data/idx-x4.txt'
    scale = 4
    crop_lr = 32
    # dataset_trn = DatasetExtTrn(path_idx=path_idx, crop_lr=crop_lr, scale=scale, in_memory=True).build()
    dataset_val = DatasetExtVal(path_idx=path_idx, crop_lr=crop_lr, scale=scale, in_memory=True).build()
    for xi, x in enumerate(dataset_val):
        num_plots = len(x)
        for ki, (k, v) in enumerate(x.items()):
            plt.subplot(1, num_plots, ki + 1)
            plt.imshow(v)
            plt.title('({}): {}'.format(ki, k))
        plt.show()
        print('-')


if __name__ == '__main__':

    main_run()

    print('-')
