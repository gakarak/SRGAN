#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import albumentations as aug
from torch.utils.data import Dataset
import cv2
import logging
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


class DatasetExtTrn(Dataset):

    def __init__(self, path_idx: str, crop_size_lr: int,
                 scale: int, augmentator=get_default_augmentor(),
                 in_memory=False, upscale_lr=False, interpolation=cv2.INTER_CUBIC):
        self.path_idx = path_idx
        self.wdir = os.path.dirname(self.path_idx)
        self.crop_size = crop_size_lr
        self.augmentator = augmentator
        self.in_memory = in_memory
        self.scale = scale
        self.upscale_lr = upscale_lr
        self.interpolation = interpolation
        self.data_idx = None
        self.data = None

    def build(self):
        self.data_idx = pd.read_csv(self.path_idx)
        return self

    def _load_sample(self, path_sample_lr: str, path_sample_hr: str):
        path_lr = os.path.join(self.wdir, path_sample_lr)
        path_hr = os.path.join(self.wdir, path_sample_hr)
        img_lr = read_rgb_img(path_lr)
        img_hr = read_rgb_img(path_hr)
        if self.upscale_lr:
            siz_xy = img_hr.shape[:2][::-1]
            img_lr = cv2.resize(img_hr, siz_xy, interpolation=self.interpolation)
        return img_lr, img_hr

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, item):
        pass



if __name__ == '__main__':
    pass
