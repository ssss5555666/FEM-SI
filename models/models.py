import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import math
from .submodule import *
from torchvision.utils import save_image
from timm.models.layers import trunc_normal_, DropPath


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class codegeneration(torch.nn.Module):
    def __init__(self):
        super(codegeneration, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3, bias=True),
                                   nn.LeakyReLU(negative_slope=0.1),
                                   nn.Conv2d(64, 64, 3, 1, 1, bias=True),
                                   nn.LeakyReLU(negative_slope=0.1))

        self.layer1 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.1),
                                    selfattention(64),
                                    nn.Conv2d(64, 64, 3, 1, 1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.1))  # 64

        self.layer2_1 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1, bias=True),
                                      nn.LeakyReLU(negative_slope=0.1),
                                      selfattention(128),
                                      nn.Conv2d(128, 128, 3, 1, 1, bias=True),
                                      nn.LeakyReLU(negative_slope=0.1), )  # 64

        self.resblock1 = Block(128, 128)
        self.resblock2 = Block(128, 128)

        self.layer2_2 = nn.Sequential(nn.Conv2d(128, 128, 3, 2, 1, bias=True),
                                      nn.LeakyReLU(negative_slope=0.1),
                                      nn.Conv2d(128, 128, 3, 1, 1, bias=True),
                                      nn.LeakyReLU(negative_slope=0.1), )  # 64

        self.layer3_1 = nn.Sequential(nn.Conv2d(128, 256, 3, 2, 1, bias=True),
                                      nn.LeakyReLU(negative_slope=0.1),
                                      selfattention(256),
                                      nn.Conv2d(256, 256, 3, 1, 1, bias=True),
                                      nn.LeakyReLU(negative_slope=0.1), )  # 64

        self.layer3_2 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1, bias=True),  # stride 2 for 128x128
                                      nn.LeakyReLU(negative_slope=0.1),
                                      nn.Conv2d(256, 128, 3, 1, 1, bias=True),
                                      nn.LeakyReLU(negative_slope=0.1))  # 64

        self.expresscode = nn.Sequential(  # nn.Linear(6272,3136),
            # nn.LeakyReLU(negative_slope=0.1),
            # nn.Linear(3136, 2048),
            # nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(6272, 2048),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(2048, 1024))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        # encoder
        out_1 = self.conv1(x)
        out_1 = self.layer1(out_1)
        out_2 = self.layer2_1(out_1)
        out_2 = self.resblock1(out_2)
        out_2 = self.resblock2(out_2)
        out_2 = self.layer2_2(out_2)
        out_3 = self.layer3_1(out_2)
        out_3 = self.layer3_2(out_3)
        # print(out_3.shape)
        out_3 = out_3.contiguous().view(x.size()[0], -1)
        # print(out_3.shape,'out3')
        expcode = self.expresscode(out_3)

        expcode = expcode.view(x.size()[0], -1, 1, 1)  # B 256 1 1
        # print(expcode.shape)
        expcode = torch.tanh(expcode)
        # print(expcode.shape)
        return expcode


class exptoflow(torch.nn.Module):
    def __init__(self):
        super(exptoflow, self).__init__()

        self.motiongen1 = nn.Sequential(nn.Conv2d(1024, 16384, 1, 1, 0),
                                        nn.PixelShuffle(4),
                                        nn.LeakyReLU(negative_slope=0.1),
                                        selfattention(1024),
                                        nn.Conv2d(1024, 1024, 3, 1, 1),
                                        nn.LeakyReLU(negative_slope=0.1), )  # to 4*4

        self.motiongen2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                        nn.Conv2d(1024, 512, 3, 1, 1),
                                        nn.LeakyReLU(negative_slope=0.1),
                                        selfattention(512),
                                        nn.Conv2d(512, 512, 3, 1, 1),
                                        nn.LeakyReLU(negative_slope=0.1), )  # to 8*8

        self.resblock1 = Block(512, 512)
        self.resblock2 = Block(512, 512)
        # self.resblock3 = BasicBlockNormal(128,128)
        # self.resblock4 = BasicBlockNormal(128, 128)

        self.motiongen3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                        nn.Conv2d(512, 256, 3, 1, 1),
                                        nn.LeakyReLU(negative_slope=0.1),
                                        selfattention(256),
                                        nn.Conv2d(256, 256, 1, 1, 0, bias=False),#7.28最新改动把bias=F 删除
                                        nn.LeakyReLU(negative_slope=0.1))  # 16*16

        self.resblock3 = Block(256, 256)
        self.resblock4 = Block(256, 256)

        # self.motiongen4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #                              nn.Conv2d(256, 128, 3, 1, 1),
        #                              nn.LeakyReLU(negative_slope=0.1),
        #                              selfattention(128),
        #                              nn.Conv2d(128, 128, 1, 1, 0, bias=False),
        #                              nn.LeakyReLU(negative_slope=0.1))  # 32*32
        #
        # self.resblock5 = BasicBlockNormal(128, 128)
        # self.resblock6 = BasicBlockNormal(128, 128)

        self.toflow5 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                     nn.Conv2d(256, 128, 3, 1, 1),
                                     nn.LeakyReLU(negative_slope=0.1),
                                     selfattention(128),
                                     # nn.Conv2d(128, 64, 3, 1, 1),
                                     nn.LeakyReLU(negative_slope=0.1),
                                     nn.Conv2d(128, 64, 1, 1, 0, bias=False))  # 32*32
        #
        # self.resblock5 = Block(64, 64)
        # self.resblock6 = Block(64, 64)
        #
        # self.toflow6 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #                              nn.Conv2d(64, 32, 3, 1, 1),
        #                              nn.LeakyReLU(negative_slope=0.1),
        #                              #selfattention(128),
        #                              # nn.Conv2d(128, 64, 3, 1, 1),
        #                              nn.LeakyReLU(negative_slope=0.1),
        #                              nn.Conv2d(32, 16, 1, 1, 0, bias=False))  # 64*64

        # self.normact = F.softmax()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.toflow5:
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.05)

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()  # 16 64 64
        Bf, Cf, Hf, Wf = flo.size()  # 16 64 64
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

    def gaussian2kp(self, heatmap):
        """
        Extract the mean and from a heatmap
        """
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1)
        grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)
        value = (heatmap * grid).sum(dim=(2, 3))
        # kp = {'value': value}

        return value

    def forward(self, expcode):

        motion = self.motiongen1(expcode)
        motion = self.motiongen2(motion)
        motion = self.resblock1(motion)
        motion = self.resblock2(motion)
        motion = self.motiongen3(motion)
        motion = self.resblock3(motion)
        motion = self.resblock4(motion)
        # motion = self.motiongen4(motion)
        # motion = self.resblock5(motion)
        # motion = self.resblock6(motion)
        # motion = self.motiongen4(motion)
        # motion = self.resblock5(motion)
        # motion = self.resblock6(motion)
        # motion = self.resblock3(motion)
        # motion = self.resblock4(motion)
        flow = self.toflow5(motion)  # 64 32 32
        final_flow_shape = flow.shape
        flow = flow.view(final_flow_shape[0], final_flow_shape[1], -1)  # 64 1024
        flow = F.softmax(flow / 0.1, dim=2)  # 64 1024
        flow = flow.view(*final_flow_shape)  # 64 32 32
        flow = self.gaussian2kp(flow)  # 64 2
        # print('flow',flow.shape)
        flow = flow.permute(0, 2, 1)  # 2 64
        # identity_grid = make_coordinate_grid((32, 32), type=flow.type())#32 32 2
        # print(identity_grid.shape)
        # identity_grid = identity_grid.permute(2,0,1)
        # print(identity_grid.shape)
        # identity_grid = identity_grid.view(flow.shape[0],flow.shape[1], 8, 8 )
        flow = flow.view(flow.shape[0], flow.shape[1], 8, 8)
        # flow = flow.view(flow.shape[0],2,32 ,32)# 2 32 32
        backflow = self.warp(flow.clone(), flow) * -1.0

        return flow, backflow


from torchvision import models


class generator(torch.nn.Module):
    def __init__(self, is_exp=False):
        super(generator, self).__init__()
        self.is_exp = is_exp
        vgg_pretrained_cnn = models.vgg19(pretrained=True).features
        # self.vgg_pretrained_classifer = models.vgg19(pretrained=False).classifier

        self.conv1_1 = nn.Sequential(vgg_pretrained_cnn[0], vgg_pretrained_cnn[1])
        self.conv1_2 = nn.Sequential(vgg_pretrained_cnn[2], vgg_pretrained_cnn[3])

        self.conv2_1 = nn.Sequential(vgg_pretrained_cnn[5], vgg_pretrained_cnn[6])
        self.conv2_2 = nn.Sequential(vgg_pretrained_cnn[7], vgg_pretrained_cnn[8])

        self.conv3_1 = nn.Sequential(vgg_pretrained_cnn[10], vgg_pretrained_cnn[11])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

        self.inplanes = 256
        self.redconv = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1, 1), nn.ReLU())

        self.cnn = self._make_layer(BasicBlockNormal, 256, 8, stride=1)  # 2

        self.up = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True),
                                nn.ReLU(),
                                nn.Conv2d(128, 128, 3, 1, 1, bias=True),
                                nn.ReLU())
        self.up2 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True),
                                 nn.ReLU(),
                                 nn.Conv2d(128, 128, 3, 1, 1, bias=True),
                                 nn.ReLU())

        self.torgb = nn.Conv2d(128, 3, 3, 1, 1)

        self.noise_encoding1 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1, bias=True),
                                             nn.ReLU(),
                                             nn.Conv2d(128, 128, 3, 1, 1, bias=True),
                                             nn.ReLU())
        self.noise_encoding0 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1, bias=True),
                                             nn.ReLU(),
                                             nn.Conv2d(256, 256, 3, 1, 1, bias=True),
                                             nn.ReLU())

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, 0),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        Bf, Cf, Hf, Wf = flo.size()

        ## down sample flow
        # scale = H/Hf
        flo = nn.functional.interpolate(flo, size=(H, W), mode='bilinear', align_corners=True)  # resize flow to x
        # occmap = F.upsample(occmap, size = (H,W),mode='bilinear')
        # flo = flo*scale # rescale flow depend on its size
        ##

        # mesh grid
        xs = np.linspace(-1, 1, W)
        xs = np.meshgrid(xs, xs)
        xs = np.stack(xs, 2)
        xs = torch.Tensor(xs).unsqueeze(0).repeat(B, 1, 1, 1).cuda()

        vgrid = Variable(xs, requires_grad=False) + flo.permute(0, 2, 3, 1)

        output = nn.functional.grid_sample(x, vgrid, align_corners=True)

        return output

    @staticmethod
    def denorm(x):
        x = x.clone()
        x[:, 0, :, :] = (x[:, 0, :, :] - 0.485) / 0.229
        x[:, 1, :, :] = (x[:, 1, :, :] - 0.456) / 0.224
        x[:, 2, :, :] = (x[:, 2, :, :] - 0.406) / 0.225
        return x

    def forward(self, x, flow=None):
        # x = self.denorm(x)
        # with torch.no_grad():

        feat = self.conv1_1(x)  # 64 112 112
        feat = self.conv1_2(feat)  # 64 112 112
        feat = F.max_pool2d(feat, kernel_size=2, stride=2, padding=0, dilation=1)  # 64 56 56
        feat2 = self.conv2_1(feat)  # 128 56 56
        feat = self.conv2_2(feat2)  # 128 56 56
        feat = F.max_pool2d(feat, kernel_size=2, stride=2, padding=0, dilation=1)  # 128 28 28
        feat = self.conv3_1(feat)  # 256 28 28
        # print(feat.shape,feat2.shape)

        if flow is not None:
            global_face = F.adaptive_avg_pool2d(feat, 1)  # 256 1 1
            global_face1 = F.adaptive_avg_pool2d(feat2, 1)  # 128 1 1
            batch, _, height, width = feat.shape
            noise = feat.new_empty(batch, 256, height, width).normal_()  # 256 28 28
            face_res0 = self.noise_encoding0(noise + global_face)  # 256 28 28
            batch2, _, height2, width2 = feat2.shape
            noise2 = feat.new_empty(batch2, 128, height2, width2).normal_()  # 128 56 56
            face_res1 = self.noise_encoding1(noise2 + global_face1)  # 128 56 56

            deform_feat = self.warp(feat, flow) + face_res0  # 256 28 28
            deform_feat2 = self.warp(feat2, flow) + face_res1  # 128 56 56
        else:
            deform_feat = feat
            deform_feat2 = feat2

        deform_feat = self.redconv(deform_feat)  # 256 28 28
        out0 = self.cnn(deform_feat)  # 256 28 28
        out = self.up(out0)  # 128 56 56
        out1 = self.up2(torch.cat([deform_feat2, out], dim=1))  # 128 112 112
        face = self.torgb(out1)  # 3 112 112

        return face


class normalizer(torch.nn.Module):
    def __init__(self, is_exp=False):
        super(normalizer, self).__init__()
        self.is_exp = is_exp
        vgg_pretrained_cnn = models.vgg19(pretrained=True).features
        # self.vgg_pretrained_classifer = models.vgg19(pretrained=False).classifier

        self.conv1_1 = nn.Sequential(vgg_pretrained_cnn[0], vgg_pretrained_cnn[1])
        self.conv1_2 = nn.Sequential(vgg_pretrained_cnn[2], vgg_pretrained_cnn[3])

        self.conv2_1 = nn.Sequential(vgg_pretrained_cnn[5], vgg_pretrained_cnn[6])
        self.conv2_2 = nn.Sequential(vgg_pretrained_cnn[7], vgg_pretrained_cnn[8])

        self.conv3_1 = nn.Sequential(vgg_pretrained_cnn[10], vgg_pretrained_cnn[11])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

        self.inplanes = 256
        self.denorm = SPADE(256, 256)
        self.renorm = SPADE(256, 256)

        self.cnn = self._make_layer(BasicBlockNormal, 256, 6, stride=1)  # 2
        self.cnn3 = self._make_layer(BasicBlockNormal, 256, 1, stride=1)  # 2

        self.up = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True),
                                nn.ReLU(),
                                nn.Conv2d(128, 128, 3, 1, 1, bias=True),
                                nn.ReLU(),
                                nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True),
                                nn.ReLU())

        self.torgb = nn.Conv2d(128, 3, 1, 1, 0)

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, 0),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, denorm=True, code=None):
        # x = self.denorm(x)
        # with torch.no_grad():

        feat = self.conv1_1(x)
        feat = self.conv1_2(feat)
        feat = F.max_pool2d(feat, kernel_size=2, stride=2, padding=0, dilation=1)
        feat = self.conv2_1(feat)
        feat = self.conv2_2(feat)
        feat = F.max_pool2d(feat, kernel_size=2, stride=2, padding=0, dilation=1)
        feat = self.conv3_1(feat)

        if denorm:
            deform_feat = self.denorm(feat, code)
        else:
            deform_feat = self.renorm(feat, code)

        out = self.cnn(deform_feat)
        out = self.cnn3(out)
        out = self.up(out)
        face = self.torgb(out)

        return face

