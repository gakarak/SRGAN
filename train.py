import argparse
import os
import cv2
from math import log10
import matplotlib.pyplot as plt

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from loss import GeneratorLoss
from model import Generator, Discriminator
from data_utils import get_device


def train_step():


    print('-')




def main_train(path_trn: str, path_val: str,
               crop_size: int, upscale_factor: int, num_epochs: int,
               num_workers: int, to_device: str = 'cuda:0', batch_size: int = 64):
    to_device = get_device(to_device)
    train_set = TrainDatasetFromFolder(path_trn, crop_size=crop_size, upscale_factor=upscale_factor)
    val_set = ValDatasetFromFolder(path_val, upscale_factor=upscale_factor)
    # train_set = TrainDatasetFromFolder('data/VOC2012/train', crop_size=crop_size, upscale_factor=upscale_factor)
    # val_set = ValDatasetFromFolder('data/VOC2012/val', upscale_factor=upscale_factor)
    #
    train_loader = DataLoader(dataset=train_set, num_workers=num_workers, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=num_workers, batch_size=1, shuffle=False)

    netG = Generator(upscale_factor)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    netD = Discriminator()
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    generator_criterion = GeneratorLoss()

    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()

    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())

    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

    for epoch in range(1, num_epochs + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

        netG.train()
        netD.train()
        # FIXME: seperate function for epoch training
        for data, target in train_bar:
            g_update_first = True
            batch_size = data.size(0)
            #
            # img_hr = target.numpy().transpose((0, 2, 3, 1))[0]
            # img_lr = data.numpy().transpose((0, 2, 3, 1))[0]
            # img_lr_x4 = cv2.resize(img_lr, img_hr.shape[:2], interpolation=cv2.INTER_CUBIC)
            # #
            # plt.subplot(1, 3, 1)
            # plt.imshow(img_hr)
            # plt.subplot(1, 3, 2)
            # plt.imshow(img_lr)
            # plt.subplot(1, 3, 3)
            # plt.imshow(img_lr_x4)
            # plt.show()
            running_results['batch_sizes'] += batch_size

            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            # real_img = Variable(target)
            # if torch.cuda.is_available():
            #     real_img = real_img.cuda()
            # z = Variable(data)
            # if torch.cuda.is_available():
            #     z = z.cuda()
            z = data.to(to_device)
            real_img = target.to(to_device)
            fake_img = netG(z)

            netD.zero_grad()
            real_out = netD(real_img).mean()
            fake_out = netD(fake_img).mean()
            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph=True)
            optimizerD.step()

            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            netG.zero_grad()
            g_loss = generator_criterion(fake_out, fake_img, real_img)
            g_loss.backward()
            optimizerG.step()
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()

            g_loss = generator_criterion(fake_out, fake_img, real_img)
            running_results['g_loss'] += float(g_loss) * batch_size
            d_loss = 1 - real_out + fake_out
            running_results['d_loss'] += float(d_loss) * batch_size
            running_results['d_score'] += float(real_out) * batch_size
            running_results['g_score'] += float(fake_out) * batch_size

            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, num_epochs, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))

        netG.eval()
        #FIXME: seperate function for epoch validation
        with torch.no_grad():
            out_path = 'training_results/SRF_' + str(upscale_factor) + '/'
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            val_bar = tqdm(val_loader)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size
                # lr = Variable(val_lr, volatile=True)
                # hr = Variable(val_hr, volatile=True)
                # if torch.cuda.is_available():
                #     lr = lr.cuda()
                #     hr = hr.cuda()
                lr = val_lr.to(to_device)
                hr = val_hr.to(to_device)
                sr = netG(lr)

                batch_mse = ((sr - hr) ** 2).mean()
                valing_results['mse'] += float(batch_mse) * batch_size
                batch_ssim = float(pytorch_ssim.ssim(sr, hr)) #.data[0]
                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['psnr'] = 10 * log10(1 / (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['psnr'], valing_results['ssim']))

                val_images.extend(
                    [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                     display_transform()(sr.data.cpu().squeeze(0))])
            val_images = torch.stack(val_images)
            val_images = torch.chunk(val_images, val_images.size(0) // 15)
            val_save_bar = tqdm(val_images, desc='[saving training results]')
            index = 1
            for image in val_save_bar:
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                index += 1

        # save model parameters
        torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_%d.pth' % (upscale_factor, epoch))
        torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_%d.pth' % (upscale_factor, epoch))
        # save loss\scores\psnr\ssim
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])

        if epoch % 10 == 0 and epoch != 0:
            out_path = 'statistics/'
            data_frame = pd.DataFrame(
                data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                      'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            data_frame.to_csv(out_path + 'srf_' + str(upscale_factor) + '_train_results.csv', index_label='Epoch')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train Super Resolution Models')
    parser.add_argument('--trn', default=None, type=str, required=True, help='path to train dataset')
    parser.add_argument('--val', default=None, type=str, required=True, help='path to validation dataset')
    parser.add_argument('--crop_size', default=88, type=int, help='training images crop size')
    parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                        help='super resolution upscale factor')
    parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')
    parser.add_argument('--threads', default=1, type=int, help='#workers for parallel processing')
    parser.add_argument('--batch_size', default=64, type=int, help='batch-size')
    parser.add_argument('--device', default='cuda:0', type=str, help='device, default "cuda:0"')
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
        to_device=to_device
    )



