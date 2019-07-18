import argparse

import os
import time
import numpy as np
import torch
import logging
from PIL import Image
from data_utils import get_device
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
import matplotlib.pyplot as plt
import s2_utils
from data_utils_ext import x_preprocess
import sys
from osgeo import gdal
from model_hr_inference import infer_pmap_by_grid_torch
from model import Generator


def crop_to_size_factor(img: np.ndarray, siz_factor: int) -> np.ndarray:
    siz_inp = img.shape[:2]
    siz_out = (siz_factor * np.floor(np.array(siz_inp) / siz_factor)).astype(np.int)
    ret = img[:siz_out[0], :siz_out[1], ...]
    return ret


def main_inference(path_image: str, path_image_out: str, path_model: str, upscale_factor: int, to_devide ='cuda:0'):
    if path_image_out is None:
        model_pref = os.path.basename(os.path.splitext(path_model)[0])
        # path_image_out = path_image + '_m{}_upscale_x{}.mbtiles'.format(model_pref, upscale_factor)
        path_image_out = path_image + '_m{}_upscale_x{}.jp2'.format(model_pref, upscale_factor)
    #
    model = Generator(upscale_factor).eval()
    model = model.to(to_devide)
    model.load_state_dict(torch.load(path_model))
    #
    image_lr0 = s2_utils._read_tif_nch(path_image)
    # image_lr0 = crop_to_size_factor(image_lr0, 2**5)
    image_lr0_batch = image_lr0
    t1 = time.time()
    print(':: start inference (inp-shape = {})'.format(image_lr0_batch.shape))
    pmap = infer_pmap_by_grid_torch(model, image_lr0_batch,
                                    pproc_function=x_preprocess,
                                    to_device=to_device,
                                    upscale_factor=upscale_factor,
                                    num_debug_prints=10,
                                    is_u8_norm=True)
    dt = time.time() - t1
    print('\t... done, dt ~ {:0.2f} (s)'.format(dt))
    print('export geo-data into: [{}]'.format(path_image_out))
    ds_pmap = s2_utils.clone_georef_ds(path_image, pmap)
    # gdal.Translate(path_image_out, ds_pmap, format="MBTiles")
    gdal.Translate(path_image_out, ds_pmap, format="JP2OpenJPEG")
    #JP2OpenJPEG
    print('\t\t... done, out={}'.format(path_image_out))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Test Single Image')
    parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
    parser.add_argument('--image', type=str, help='test S2 image')
    # parser.add_argument('--image_gt', type=str, help='test GT image')
    parser.add_argument('--model', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
    parser.add_argument('--out', default=None, type=str, help='output image path')
    parser.add_argument('--device', default='cuda:0', type=str, help='device, default "cuda:0"')
    args = parser.parse_args()
    #
    to_device = get_device(args.device)
    main_inference(
        path_image=args.image,
        path_image_out=args.out,
        path_model=args.model,
        upscale_factor=args.upscale_factor,
        to_devide=to_device
    )
