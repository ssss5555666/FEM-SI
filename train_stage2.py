import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import argparse
from dataloader.ME import *
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from vgg19 import *
import random
from dataloader import Voxall as DA
from models import *
#from models import  Twoscalemotion
from Loss import *
from models.Unet import Unet as flownet
from shiftViT.shiftvit import ShiftViT


import ssl
ssl._create_default_https_context = ssl._create_unverified_context






parser = argparse.ArgumentParser(description='FaceCycle')
parser.add_argument('--datapath', default='./dataloader/Voxall.txt',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=40,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default= r'',
                    help='load model')
parser.add_argument('--savemodel', default=r'',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument("--mode", default="train", choices=["train", "reconstruction", "animate"])
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(2)
torch.cuda.manual_seed(4)

save_image_fold = args.savemodel + 'imgs/'
if not os.path.isdir(save_image_fold):
    os.makedirs(save_image_fold)


def fast_collate(batch):
    imgs0 = [img[0] for img in batch]
    imgs1 = [img[1] for img in batch]

    w = imgs0[0].size[0]
    h = imgs0[0].size[1]

    tensor0 = torch.zeros((len(imgs0), 3, h, w), dtype=torch.uint8)
    tensor1 = torch.zeros((len(imgs1), 3, h, w), dtype=torch.uint8)

    for i, img in enumerate(imgs0):
        nump_array = np.asarray(img, dtype=np.uint8)
        tens = torch.from_numpy(nump_array)
        if (nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor0[i] += torch.from_numpy(nump_array)

    for i, img in enumerate(imgs1):
        nump_array = np.asarray(img, dtype=np.uint8)
        tens = torch.from_numpy(nump_array)
        if (nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor1[i] += torch.from_numpy(nump_array)

    return tensor0, tensor1


dataset = FramesDataset(is_train=(args.mode == 'train'))
dataset = DatasetRepeater(dataset, num_repeats=80)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2, drop_last=True)



def denorm(x):
    x[:, 0, :, :] = x[:, 0, :, :] * 0.229 + 0.485
    x[:, 1, :, :] = x[:, 1, :, :] * 0.224 + 0.456
    x[:, 2, :, :] = x[:, 2, :, :] * 0.225 + 0.406
    return x


def denorm_reto(x):
    y = x.clone()
    y[:, 0, :, :] = ((x[:, 2, :, :] * 0.229 + 0.485) * 255 - 91.4953)
    y[:, 1, :, :] = ((x[:, 1, :, :] * 0.224 + 0.456) * 255 - 103.8827)
    y[:, 2, :, :] = ((x[:, 0, :, :] * 0.225 + 0.406) * 255 - 131.0912)
    return y.contiguous()


device = torch.device('cuda')
vgg = Vgg19(requires_grad=False)

if torch.cuda.is_available():
    vgg = nn.DataParallel(vgg)
    vgg.to(device)
    vgg.eval()
feat_layers = ['r21', 'r31', 'r41']

flownet_exp = flownet(n_channels=2, n_classes=2, bilinear=False).cuda()
codegeneration = codegeneration().cuda()
exptoflow_exp = exptoflow().cuda()
Swap_Generator_exp = generator().cuda()
Swap_Generator_mag = generator().cuda()

if args.loadmodel is not None:
     state_dict = torch.load(args.loadmodel)
     Swap_Generator_exp.load_state_dict(state_dict['Swap_Generator_exp'])
     codegeneration.load_state_dict(state_dict['codegeneration'])
     exptoflow_exp.load_state_dict(state_dict['exptoflow_exp'])




optimizer = optim.Adam([{"params": Swap_Generator_mag.parameters()}, \
                        {"params":flownet_exp.parameters()}
                        ], lr=1e-5, betas=(0.5, 0.999))
#pytorch_total_params = sum(p.numel() for p in codegeneration.parameters())


def adjust_learning_rate(optimizer, epoch):
    lr = 5e-5
    if epoch >= 10 and epoch < 20:
        lr = 5e-5
    elif epoch >= 20 and epoch < 30:
        lr = 1e-5
    elif epoch >= 30 and epoch < 40:
        lr = 1e-5
    elif epoch >= 40:
        lr = 5e-6
    elif epoch == 0:
        lr = 1e-5

    print(epoch, lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    Bf, Cf, Hf, Wf = flo.size()
    ## down sample flow
    # scale = H/Hf
    flo = F.upsample(flo, size=(H, W), mode='bilinear', align_corners=True)  # resize flow to x
    flo = torch.clamp(flo, -1, 1)
    # flo = flo*scale # rescale flow depend on its size
    ##
    # mesh grid
    xs = np.linspace(-1, 1, W)
    xs = np.meshgrid(xs, xs)
    xs = np.stack(xs, 2)
    xs = torch.Tensor(xs).unsqueeze(0).repeat(B, 1, 1, 1).to(device)

    vgrid = Variable(xs, requires_grad=False) + flo.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)

    return output


def warp_flow(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    Bf, Cf, Hf, Wf = flo.size()
    ## down sample flow
    # scale = H/Hf
    # flo = F.upsample(flo, size = (H,W),mode='bilinear', align_corners=True)  # resize flow to x
    # flo = torch.clamp(flo,-1,1)        #
    # flo = flo*scale # rescale flow depend on its size
    ##

    # mesh grid
    xs = np.linspace(-1, 1, W)
    xs = np.meshgrid(xs, xs)
    xs = np.stack(xs, 2)
    xs = torch.Tensor(xs).unsqueeze(0).repeat(B, 1, 1, 1).cuda()

    vgrid = Variable(xs, requires_grad=False) + flo.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    output = torch.clamp(output, -1, 1)
    return output




def forwardloss(x, batch):
    optimizer.zero_grad()#优化器
    #mid_original = x['mid_original'].cuda()
    apex_aligned = x['apex_aligned'].cuda()
    mid_aligned = x['mid_aligned'].cuda()
    onset_aligned = x['onset_aligned'].cuda()
    expcode = codegeneration(mid_aligned)

    flow_full, backflow_full = exptoflow_exp(expcode)

    flow_mag = flownet_exp(backflow_full)#学习放大光流
    rec_apex_aligned_face = Swap_Generator_mag(mid_aligned, flow_mag)# 中间帧 + flowmag = 顶针
    rec_mid_aligned = Swap_Generator_mag(mid_aligned, None)


    # photometric loss
    pixel_loss = F.l1_loss(rec_apex_aligned_face,apex_aligned) + F.l1_loss(rec_mid_aligned,mid_aligned)# F.l1_loss(rec_mid_aligned_face,mid_aligned)




    # percetual loss
    perc_full = torch.cat([apex_aligned,mid_aligned], dim=0)#真值图像：对齐中间帧，原图中间帧，对齐初始帧，原图中间帧
    perc_full = perc_full.clone()
    rec_full = torch.cat([rec_apex_aligned_face, rec_mid_aligned], dim=0)
    im_feat = vgg(perc_full, feat_layers)
    rec_feat = vgg(rec_full, feat_layers)

    pec_loss0 = perceptual_loss(rec_feat[0], im_feat[0])
    pec_loss1 = perceptual_loss(rec_feat[1], im_feat[1])
    pec_loss2 = perceptual_loss(rec_feat[2], im_feat[2])
   # pec_loss3 = perceptual_loss(rec_feat[3], im_feat[3])

    perc_loss = pec_loss0 + pec_loss1 + pec_loss2 #+ pec_loss3

    # SSIM loss
    s_loss = ssim_loss(rec_apex_aligned_face, apex_aligned) + ssim_loss(rec_mid_aligned, mid_aligned)# ssim_loss(rec_mid_aligned_face,mid_aligned)

    loss = perc_loss + pixel_loss + 2.0 * s_loss
    # print(loss)
    # torch.autograd.set_detect_anomaly = True

    loss.backward()

    optimizer.step()

    if batch % 100 == 0:
        print('iter %d percetual loss: %.2f pixel_loss: %.2f ssim loss: %.2f ' % (
        batch, perc_loss, pixel_loss, s_loss))

    if batch % 100 == 0:
        save_image(torch.cat((mid_aligned.data \
                                  , apex_aligned.data \
                                  , rec_apex_aligned_face.data \
                                  ), 0), os.path.join(save_image_fold, '{}_{}_decodeflow3.png'.format(epoch, int(batch / 100))))




if __name__ == '__main__':

    exptoflow_exp.eval() ##()
    flownet_exp.train()
    Swap_Generator_exp.eval()
    Swap_Generator_mag.train()
    codegeneration.eval()

    for epoch in range(0, 40):
        adjust_learning_rate(optimizer, epoch)
        batch_idx = 0
        for x in dataloader:
            forwardloss(x, batch_idx)
            batch_idx += 1

        # SAVE
        if not os.path.isdir(args.savemodel):
            os.makedirs(args.savemodel)
        # model.module.state_dict() for nn.dataparallel
        savefilename = args.savemodel + 'ExpCode_' + str(epoch) + '.tar'
        torch.save({'codegeneration': codegeneration.state_dict(),
                    'exptoflow_exp': exptoflow_exp.state_dict(),
                    'flownet_exp': flownet_exp.state_dict(),
                    'Swap_Generator_exp': Swap_Generator_exp.state_dict(),
                    'Swap_Generator_mag': Swap_Generator_mag.state_dict(),
                    }, savefilename)




