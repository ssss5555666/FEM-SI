import time
import os
from skimage.transform import resize
import imageio
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
from Loss import *
from models.Unet import Unet as flownet
from shiftViT.shiftvit import ShiftViT

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
parser = argparse.ArgumentParser(description='FaceCycle')
parser.add_argument('--loadmodel', default=r'',#checkpoint
                    help='load model')
parser.add_argument('--savemodel', default=r'',#save image
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument("--source_image", default=r'', help="path to source image folder")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(2)
torch.cuda.manual_seed(4)

save_image_fold = args.savemodel
if not os.path.isdir(save_image_fold):
    os.makedirs(save_image_fold)


def denorm(x):
    x[:, 0, :, :] = x[:, 0, :, :] * 0.229 + 0.485
    x[:, 1, :, :] = x[:, 1, :, :] * 0.224 + 0.456
    x[:, 2, :, :] = x[:, 2, :, :] * 0.225 + 0.406
    return x.clamp(0, 1)


def denorm_reto(x):
    x[:, 0, :, :] = ((x[:, 0, :, :] * 0.229 + 0.485) - 0.5) * 0.5
    x[:, 1, :, :] = ((x[:, 1, :, :] * 0.224 + 0.456) - 0.5) * 0.5
    x[:, 2, :, :] = ((x[:, 2, :, :] * 0.225 + 0.406) - 0.5) * 0.5
    return x.clamp(0, 1)


device = torch.device('cuda')

codegeneration = codegeneration().cuda()
#Swap_Norm = normalizer().cuda()
#motion_exp = Twoscalemotion().cuda()
#codegeneration = ShiftViT().cuda()
#exptoflow_pose = exptoflow().cuda()
exptoflow_exp = exptoflow().cuda()
flownet_exp = flownet(n_channels=2, n_classes=2, bilinear=False).cuda()
Swap_Generator_mag = generator().cuda()
Swap_Generator_exp = generator().cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    flownet_exp.load_state_dict(state_dict['flownet_exp'])
    Swap_Generator_mag.load_state_dict(state_dict['Swap_Generator_mag'])
    codegeneration.load_state_dict(state_dict['codegeneration'])
    Swap_Generator_exp.load_state_dict(state_dict['Swap_Generator_exp'])
    exptoflow_exp.load_state_dict(state_dict['exptoflow_exp'])
    #motion_exp.load_state_dict(state_dict['motion_exp'])



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

    mid_aligned = x
    expcode = codegeneration(mid_aligned)
    flow_full, backflow_full = exptoflow_exp(expcode)
    backflow_mag = flownet_exp(backflow_full)#学习放大光流

    rec_onset_aligned_face = Swap_Generator_exp(mid_aligned, flow_full)  # 生成的中间帧对齐脸 - 表情信息 = 生成的初始帧对齐脸 loss3
    rec_apex_aligned_face = Swap_Generator_mag(mid_aligned, backflow_mag)  # 中间帧 + flowmag = 顶针
    # rec_mid_aligned = Swap_Generator_exp(mid_aligned, None)
   # rec_onset_aligned = Swap_Generator_mag(rec_apex_aligned_face, flow_mag)

    #rec_apex_aligned_face = Swap_Generator_mag(mid_aligned, flow_mag)  # 生成的初始帧对齐脸 + 表情信息 = 重建的中间帧对齐脸 ；loss4

    expcode1 = codegeneration(rec_apex_aligned_face)
    flow_full1, backflow_full1 = exptoflow_exp(expcode1)
    backflow_mag1 = flownet_exp(backflow_full1)  # 学习放大光流

    #rec_onset_aligned_face1 = Swap_Generator_exp(rec_apex_aligned_face, flow_full1)
    rec_apex_aligned_face1 = Swap_Generator_mag(mid_aligned, backflow_mag + backflow_mag1)  # 生成的初始帧对齐脸 + 表情信息 = 重建的中间帧对齐脸 ；loss4

    expcode2 = codegeneration(rec_apex_aligned_face1)
    flow_full2, backflow_full2 = exptoflow_exp(expcode2)
    backflow_mag2 = flownet_exp(backflow_full2)  # 学习放大光流

    # rec_onset_aligned_face1 = Swap_Generator_exp(rec_apex_aligned_face, flow_full1)
    rec_apex_aligned_face2 = Swap_Generator_mag(mid_aligned, backflow_mag + backflow_mag1+ backflow_mag2)

    #
    # save_image(torch.cat((mid_aligned.data \
    #                           , rec_apex_aligned_face.data \
    #                           #, rec_onset_aligned_face.data \
    #                           , rec_apex_aligned_face1.data \
    #                           , rec_apex_aligned_face2.data
    #                       ), 0), os.path.join(save_image_fold, 'translation'+ str(batch) + '.png'))
    save_image(rec_onset_aligned_face.data, os.path.join(save_image_fold, str(batch) + 'A1' + '.png'))
    save_image(mid_aligned.data, os.path.join(save_image_fold,  str(batch) + 'A2' +'.png'))
    save_image(rec_apex_aligned_face.data,os.path.join(save_image_fold, str(batch) +'B'+ '.png'))
    save_image(rec_apex_aligned_face1.data, os.path.join(save_image_fold,  str(batch) +'C' + '.png'))
    save_image(rec_apex_aligned_face2.data, os.path.join(save_image_fold,  str(batch) + 'D' +'.png'))

    #save_image(rec_apex_aligned_face.data , os.path.join(save_image_fold, 'translation' + str(batch) + '.png'))





if __name__ == '__main__':
    # for epoch in range(0, 40):
    #     adjust_learning_rate(optimizer, epoch)
    #     # prefetcher = data_prefetcher(TrainImgLoader)
    #
    #     # input0, input1 = prefetcher.next()
    #     batch_idx = 0
    #     for x in dataloader:  # 遍历数据集
    #         # x=x.cuda()
    #         # while input0 is not None:
    #         #     input0, input1 = input0.cuda(), input1.cuda()
        exptoflow_exp.eval()##
        flownet_exp.eval()
        #motion_exp.eval()
        codegeneration.eval()
        Swap_Generator_mag.eval()
        Swap_Generator_exp.eval()
        #exptoflow_pose.eval()
        source_image = sorted(os.listdir(args.source_image), key=lambda x: x)
        for i, image_one in enumerate(source_image):
            image_onest = os.path.join(args.source_image, image_one)
            # mid_original = [r'H:\Song\FaceCycle-master\Imgs\id2.jpg']
            image = imageio.imread(image_onest)
            # image = Image.open(mid_original[0]).convert('RGB')
            source_image = resize(image, (112, 112))[..., :3]
            source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).cuda()
            forwardloss(source, i)
            # batch_idx += 1
                # input0, input1 = prefetcher.next()

            # # SAVE
            # if not os.path.isdir(args.savemodel):
            #     os.makedirs(args.savemodel)
            # # model.module.state_dict() for nn.dataparallel
            # savefilename = args.savemodel + 'ExpCode_' + str(epoch) + '.tar'
            # torch.save({'codegeneration': codegeneration.state_dict(),
            #             'exptoflow': exptoflow.state_dict(),
            #             'Swap_Generator': Swap_Generator.state_dict(),
            #             'flownet_exp':flownet_exp.state_dict(),
            #             'flownet_pose':flownet_pose.state_dict()
            #             }, savefilename)




