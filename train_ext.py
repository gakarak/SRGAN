import argparse
import os
import cv2
import math
import time
from math import log10
import logging
import matplotlib.pyplot as plt

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
from torch.optim.optimizer import Optimizer

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from loss import GeneratorLoss
from model import Generator, Discriminator
from data_utils import get_device
from data_utils_ext import DatasetExtTrn, DatasetExtVal, x_preprocess, convert_tensor_u8_to_fX



def train_step(dataloader: DataLoader,
               netD: nn.Module, netG: nn.Module,
               optimizerD: Optimizer, optimizerG: Optimizer,
               generator_criterion_loss,
               idx_epoch, num_epochs, num_print: int = 5) -> dict:
    netG.train()
    netD.train()
    results = {'d_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
    batch_sizes = 0
    num_samples = len(dataloader)
    step_ = int(math.ceil(num_samples) / num_print)
    t1 = time.time()
    for idx_train, data_train in enumerate(dataloader):
        # (0) get lr/hr data
        data_lr, data_hr_target = data_train['lr'], data_train['hr']
        batch_size = data_lr.size(0)
        batch_sizes += batch_size
        # (1) Update D network: maximize D(x)-1-D(G(z))
        z = x_preprocess(data_lr, to_device=to_device)
        real_img = x_preprocess(data_hr_target, to_device=to_device)
        fake_img = netG(z)
        #
        netD.zero_grad()
        real_out = netD(real_img).mean()
        fake_out = netD(fake_img).mean()
        d_loss = 1 - real_out + fake_out
        d_loss.backward(retain_graph=True)
        optimizerD.step()
        # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
        netG.zero_grad()
        g_loss = generator_criterion_loss(fake_out, fake_img, real_img)
        g_loss.backward()
        optimizerG.step()
        fake_img = netG(z)
        fake_out = netD(fake_img).mean()
        # (3)
        g_loss = generator_criterion_loss(fake_out, fake_img, real_img)
        results['g_loss'] += float(g_loss) * batch_size
        d_loss = 1 - real_out + fake_out
        results['d_loss'] += float(d_loss) * batch_size
        results['d_score'] += float(real_out) * batch_size
        results['g_score'] += float(fake_out) * batch_size
        if (idx_train % step_) == 0:
            str_desc = ' * Loss_D: {:0.4f} Loss_G: {:0.4f} D(x): {:0.4f} D(G(z)): {:0.4f}'\
                .format(results['d_loss'], results['g_loss'], results['d_score'], results['g_score'])
            print('(TRN) [{}/{}] [{}/{}] -> {}'.format(idx_epoch, num_epochs, idx_train, num_samples, str_desc))
    dt = time.time() - t1
    results = {k: v/batch_sizes for k, v in results.items()}
    tmp_ = ', '.join(['{}: {:0.2f}'.format(k, v) for k, v in results.items()])
    print(' (TRAIN) ({}/{}) dt ~{:0.2f} (s), {}'.format(idx_epoch, num_epochs, dt, tmp_))
    return results


def validation_step(dataloader: DataLoader, netG: nn.Module,
                    out_dir: str, idx_epoch: int, num_epochs: int,
                    num_print: int = 5) -> dict:
    netG.eval()
    num_samples = len(dataloader)
    batch_sizes = 0
    step_ = int(math.ceil(num_samples) / num_print)
    t1 = time.time()
    with torch.no_grad():
        val_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
        val_images = []
        for idx_val, data_val in enumerate(dataloader):
            val_lr, val_hr_restore, val_hr = data_val['lr'], data_val['lr_up'], data_val['hr']
            batch_size = val_lr.size(0)
            batch_sizes += batch_size
            val_hr_restore = x_preprocess(val_hr_restore, to_device=None)
            lr = x_preprocess(val_lr, to_device=to_device)
            hr = x_preprocess(val_hr, to_device=to_device)
            sr = netG(lr)
            #
            batch_mse = float(((sr - hr) ** 2).mean())
            val_results['mse'] += batch_mse * batch_size
            batch_ssim = float(pytorch_ssim.ssim(sr, hr))  # .data[0]
            val_results['ssims'] += batch_ssim * batch_size
            val_results['psnr'] = 10 * log10(1 / (val_results['mse'] / batch_sizes))
            val_results['ssim'] = val_results['ssims'] / batch_sizes
            if (idx_val % step_) == 0:
                print('\t\t(VAL) [{}/{}] <- MSE/SSIM = {:0.3f}/{:0.3f}'.format(idx_val, num_samples, batch_mse, batch_ssim))
            val_images.extend([
                display_transform()(val_hr_restore.squeeze(0)),
                display_transform()(hr.data.cpu().squeeze(0)),
                display_transform()(sr.data.cpu().squeeze(0))
            ])
        val_images = torch.stack(val_images)
        val_images = torch.chunk(val_images, val_images.size(0) // 10)
        for idx_image, image in enumerate(val_images):  # val_save_bar:
            image = utils.make_grid(image, nrow=3, padding=5)
            utils.save_image(image,os.path.join(out_dir, 'epoch_%d_index_%d.png' % (idx_epoch, idx_image)), padding=5)
    dt = time.time() - t1
    val_results = {k: v / batch_sizes for k, v in val_results.items()}
    tmp_ = ', '.join(['{}: {:0.2f}'.format(k, v) for k, v in val_results.items()])
    print('\t\t(VALIDATION) ({}/{}) dt ~{:0.2f} (s), {}'.format(idx_epoch, num_epochs, dt, tmp_))
    return val_results


def export_results(current_results: dict, path_csv: str):
    df_add = pd.DataFrame(data={k: [v] for k, v in current_results.items()})
    if os.path.isfile(path_csv):
        df = pd.read_csv(path_csv)
        df = pd.concat([df, df_add], ignore_index=True)
    else:
        df = df_add
    print('\t\t... export data into CSV file [{}]'.format(path_csv))
    df.to_csv(path_csv, index=None)
    return os.path.isfile(path_csv)


def main_train(path_trn: str, path_val: str,
               crop_size: int, upscale_factor: int, num_epochs: int,
               num_workers: int, to_device: str = 'cuda:0',
               in_memory_trn: bool = False, in_memory_val: bool = False,
               batch_size: int = 64, step_val: int = 5):
    out_dir = path_trn + '_results_c{}_s{}'.format(crop_size, upscale_factor)
    out_dir_states = out_dir + '_states'
    out_dir_statistics = out_dir + '_staticstics'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_states, exist_ok=True)
    os.makedirs(out_dir_statistics, exist_ok=True)
    path_results_csv = os.path.join(out_dir, 'statistics_x{}_train_results.csv'.format(upscale_factor))
    #
    to_device = get_device(to_device)
    train_set = DatasetExtTrn(path_idx=path_trn,
                              crop_lr=crop_size,
                              scale=upscale_factor,
                              in_memory=in_memory_trn).build()
    val_set = DatasetExtVal(path_idx=path_val,
                            crop_lr=crop_size,
                            scale=upscale_factor,
                            in_memory=in_memory_val).build()
    #
    train_loader = DataLoader(dataset=train_set, num_workers=num_workers, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=num_workers, batch_size=1, shuffle=False)
    #
    netG = Generator(upscale_factor).to(to_device)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    netD = Discriminator().to(to_device)
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
    generator_criterion = GeneratorLoss().to(to_device)
    #
    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())
    # results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
    for epoch in range(1, num_epochs + 1):
        results_train = train_step(train_loader, netD, netG, optimizerD, optimizerG, generator_criterion, epoch, num_epochs)
        # FIXME: seperate function for epoch training
        if (epoch % step_val) == 0:
            results_validation = validation_step(val_loader, netG, out_dir, epoch, num_epochs)
            results_save = {**results_train, **results_validation}
            results_save['epoch'] = epoch
            export_results(results_save, path_results_csv)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Train Super Resolution Models')
    parser.add_argument('--trn', default=None, type=str, required=True, help='path to train dataset')
    parser.add_argument('--val', default=None, type=str, required=True, help='path to validation dataset')
    parser.add_argument('--crop_size', default=12, type=int, help='training images crop size')
    parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                        help='super resolution upscale factor')
    parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')
    parser.add_argument('--threads', default=1, type=int, help='#workers for parallel processing')
    parser.add_argument('--batch_size', default=32, type=int, help='batch-size')
    parser.add_argument('--device', default='cuda:0', type=str, help='device, default "cuda:0"')
    parser.add_argument('--in_memory_trn', action='store_true', help='Load train dataset into memory')
    parser.add_argument('--in_memory_val', action='store_true', help='Load validation dataset into memory')
    args = parser.parse_args()
    print('args:\n\t{}'.format(args))
    #
    to_device = get_device(args.device)
    main_train(
        path_trn=args.trn,
        path_val=args.val,
        crop_size=args.crop_size,
        upscale_factor=args.upscale_factor,
        num_epochs=args.num_epochs,
        num_workers=args.threads,
        batch_size=args.batch_size,
        to_device=to_device,
        in_memory_trn=args.in_memory_trn,
        in_memory_val=args.in_memory_val
    )



