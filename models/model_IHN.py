from .model_G_F import G_Unet_add_input
from .padding_same_conv import Conv2d
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torch.nn import init
import functools


class ResidualBlock(nn.Module):
    def __init__(self, i, out_channels, name):
        super(ResidualBlock, self).__init__()
        self.residual = nn.Sequential()
        self.residual.add_module(name +"/c1/%s" % i, Conv2d(out_channels, out_channels, kernel_size=3, stride=1))
        self.residual.add_module("relu1", nn.ReLU())
        self.residual.add_module(name + "/c2/%s" % i, Conv2d(out_channels, out_channels, kernel_size=3, stride=1))
    def forward(self, x):
        return x + self.residual(x)

class IHNet(nn.Module):
    def __init__(self, out_channels=1, z=None, is_train=False, reuse=True):
        super(IHNet, self).__init__()
        # ------------------------------ Content Aggregation ------------------------------
        self.Unet = G_Unet_add_input(input_nc=1, output_nc=1, nz=8, use_dense=True)

        # ------------------------------ Detail Generation ------------------------------
        self.n64s1 = nn.Sequential()
        self.n64s1.add_module("n64s1/c", Conv2d(1 + out_channels, 64, kernel_size=3, stride=1))

        self.dn64s1 = nn.Sequential(*[ResidualBlock(i, 64, "dn64s1") for i in range(8)])

        self.n256s1 = nn.Sequential()
        self.n256s1.add_module("n256s1/2", Conv2d(64, 256, kernel_size=3, stride=1))

        self.out = nn.Sequential()
        self.out.add_module("out", Conv2d(256, out_channels, kernel_size=1, stride=1))

    def forward(self, inputs, z=None):
        n = self.Unet(inputs, z=z)
        content_map = n

        fussion = torch.cat([content_map, inputs], dim=1)
        n = self.n64s1(fussion) # (64, 256, 256)

        n = self.dn64s1(n)

        n = self.n256s1(n)
        n = self.out(n)

        detail_map = n

        assert content_map.shape == detail_map.shape
        output_map = content_map + detail_map
        
        return content_map, detail_map, output_map

class Mixture(nn.Module):
    def __init__(self, out_channels, z=None, is_train=False, reuse=False):
        super(Mixture, self).__init__()
        self.PRL = PRLNet(out_channels)
        self.Unet = G_Unet_add_input(input_nc=1, output_nc=1, nz=8, use_dense=True)
    def forward(self, inputs, z=None):
        content_map = self.Unet(inputs, z=z)
        return self.PRL(inputs, content_map)

class VGG19(nn.Module):
    """docstring for VGG19"""
    def __init__(self):
        super(VGG19, self).__init__()

        vgg_features = models.vgg19(pretrained=True).features
        self.vgg19_relu4_4 = nn.Sequential()

        for x in range(28):
            self.vgg19_relu4_4.add_module(str(x), vgg_features[x])

    def forward(self, img):
        return self.vgg19_relu4_4(img)
