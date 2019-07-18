import argparse

import time
import numpy as np
import torch
from PIL import Image
from data_utils import get_device
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
import matplotlib.pyplot as plt
import s2_utils
from data_utils_ext import x_preprocess
import sys
from model_hr_inference import infer_pmap_by_grid_torch
from model import Generator


def crop_to_size_factor(img: np.ndarray, siz_factor: int) -> np.ndarray:
    siz_inp = img.shape[:2]
    siz_out = (siz_factor * np.floor(np.array(siz_inp) / siz_factor)).astype(np.int)
    ret = img[:siz_out[0], :siz_out[1], ...]
    return ret


def main_inference(path_image: str, path_image_out: str, path_model: str, upscale_factor: int, to_devide ='cuda:0'):
    if path_image_out is not None:
        path_image_out = path_image + '_upscale_x{}.tif'.format(upscale_factor)
    #
    model = Generator(upscale_factor).eval()
    model = model.to(to_devide)
    model.load_state_dict(torch.load(path_model))
    #
    image_lr0 = s2_utils._read_tif_nch(path_image)
    # image_lr0 = crop_to_size_factor(image_lr0, 2**5)
    # image_lr0_batch = np.expand_dims(image_lr0, axis=0).astype(np.float32) / 255.
    image_lr0_batch = image_lr0.astype(np.float32) / 255.
    # image_lr0_batch = image_lr0_batch.transpose((0, 2, 3, 1))
    # image_lr0_batch = x_preprocess(image_lr0_batch, to_device=to_device)
    # pmap = infer_pmap_by_grid_torch(model, image_lr0_batch, crop_size_pad=512)
    pmap = infer_pmap_by_grid_torch(model, image_lr0_batch, pproc_function=x_preprocess, to_device=to_device)
    #
    # image_lr = ToTensor()(image_lr0).unsqueeze(0).to(to_devide)
    # if TEST_MODE:
    #     image = image.cuda()
    start = time.time()
    out = model(image_lr)
    elapsed = (time.time() - start)
    print('cost' + str(elapsed) + 's')
    image_pr = np.array(ToPILImage()(out[0].data.cpu()))
    #
    plt.subplot(1, 3, 1)
    plt.imshow(image_lr0)
    plt.title('LR image')
    plt.subplot(1, 3, 2)
    plt.imshow(image_gt)
    # plt.imshow(image_gt[:image_pr.shape[0], :image_pr.shape[1], ...])
    plt.title('GT image')
    plt.subplot(1, 3, 3)
    plt.imshow(image_pr)
    plt.title('PR image')
    plt.show()

    print('-')

    # out_img.save('out_srf_' + str(upscale_factor) + '_' + path_image)


if __name__ == '__main__':
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
