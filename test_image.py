import argparse

import time
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
import matplotlib.pyplot as plt

from model import Generator

# if TEST_MODE:
#     model.cuda()
#     model.load_state_dict(torch.load('epochs/' + path_model))
# else:
#     model.load_state_dict(torch.load('epochs/' + path_model, map_location=lambda storage, loc: storage))


def crop_to_size_factor(img: np.ndarray, siz_factor: int) -> np.ndarray:
    siz_inp = img.shape[:2]
    siz_out = (siz_factor * np.floor(np.array(siz_inp) / siz_factor)).astype(np.int)
    ret = img[:siz_out[0], :siz_out[1], ...]
    return ret


def main_test(path_image: str, path_image_gt: str, path_model: str, upscale_factor: int, to_devide = 'cuda:0'):
    # upscale_factor = args.upscale_factor
    # TEST_MODE = True if args.test_mode == 'GPU' else False
    # path_image = args.image_name
    # path_model = args.model_name

    model = Generator(upscale_factor).eval()
    model = model.to(to_devide)
    model.load_state_dict(torch.load(path_model))
    #
    image_lr0 = np.array(Image.open(path_image))
    image_lr0 = crop_to_size_factor(image_lr0, 2**5)
    image_gt = np.array(Image.open(path_image_gt))
    #
    image_lr = ToTensor()(image_lr0).unsqueeze(0).to(to_devide)
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
    parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
    parser.add_argument('--image', type=str, help='test low resolution image name')
    parser.add_argument('--image_gt', type=str, help='test GT image')
    parser.add_argument('--model', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
    args = parser.parse_args()
    #
    main_test(
        path_image=args.image,
        path_image_gt=args.image_gt,
        path_model=args.model,
        upscale_factor=args.upscale_factor
    )
