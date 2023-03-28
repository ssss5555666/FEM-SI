
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=int(in_dim // 2), kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=int(in_dim // 2), kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B * C * W * H)
            returns :
                out : self attention value + input feature
                attention: B * N * N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # B X (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out

class Unet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.blok1 = Block(64,64)
        self.blok2 = Block(64,64)
        self.down1 = Down(32, 64) #32
        self.down2 = Down(64, 128) #16
        # self.down3 = Down(128, 256) #8
        factor = 2 if bilinear else 1
        #self.down4 = Down(256, 512 // factor) #4
        # self.up1 = Up(512, 256 // factor, bilinear) #8
        # self.up2 = Up(256, 128 // factor, bilinear) #16
        self.up3 = Up(128, 64 // factor, bilinear) #32
        self.up4 = Up(64, 32, bilinear) #64
        self.outc = OutConv(32, n_classes)
        self.attn1 = Self_Attn(64)
        self.attn2 = Self_Attn(32)

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
    def forward(self, x):
        #mf = mf.unsqueeze(2).unsqueeze(2)
        #print(mf.shape,x.shape)
        #print(mf[0])
        #x = x.mul(mf)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.blok1(x2)
        x3 = self.down2(x2)
        #print(x3.shape)
        # mf = mf.sum()
        # mf = mf / 3#3 is batch size
        # #print(mf.shape,mf,'55,')
        # mf1 = mf.repeat(1, x1.size(1), x1.size(2), x1.size(3))
        # x1 = x1 * mf1
        # mf2 = mf.repeat(1, x2.size(1), x2.size(2), x2.size(3))
        # x2 = x2 * mf2
        # #x3 = x3 * mf
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        # x = self.up2(x4, x3)
        x = self.up3(x3, x2)
        x = self.attn1(x)
        x = self.blok1(x)
        x = self.up4(x, x1)
        x = self.attn2(x)
        logits = self.outc(x)
        return logits



class Unet_mag(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(Unet_mag, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 256)
        self.blok1 = Block(512,512)
        self.blok2 = Block(256,256)
        # self.down1 = Down(32, 64) #32
        # self.down2 = Down(64, 128) #16
        # self.down3 = Down(128, 256) #8
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor) #4
        self.up1 = Up(512, 256 // factor, bilinear) #8
        # self.up2 = Up(256, 128 // factor, bilinear) #16
        # self.up3 = Up(128, 64 // factor, bilinear) #32
        # self.up4 = Up(64, 32, bilinear) #64
        self.outc = OutConv(256, n_classes)
        # self.attn1 = Self_Attn(64)
        # self.attn2 = Self_Attn(32)

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
    def forward(self, x):
        #mf = mf.unsqueeze(2).unsqueeze(2)
        #print(mf.shape,x.shape)
        #print(mf[0])
        #x = x.mul(mf)
        x1 = self.inc(x)
        x2 = self.down4(x1)
        x2 = self.blok1(x2)
        # x3 = self.down2(x2)
        #print(x3.shape)
        # mf = mf.sum()
        # mf = mf / 3#3 is batch size
        # #print(mf.shape,mf,'55,')
        # mf1 = mf.repeat(1, x1.size(1), x1.size(2), x1.size(3))
        # x1 = x1 * mf1
        # mf2 = mf.repeat(1, x2.size(1), x2.size(2), x2.size(3))
        # x2 = x2 * mf2
        # #x3 = x3 * mf
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        # x = self.up2(x4, x3)
        x = self.up1(x2, x1)
        x = self.blok2(x)
        # x = self.up4(x, x1)
        # x = self.attn2(x)
        logits = self.outc(x)
        return logits