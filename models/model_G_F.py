import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from torchvision import models
import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageFilter

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=4, kernel_size=3):
        super(Encoder, self).__init__()

        # Initial convolution block       
        enc = [   nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, 64, 7),
                  nn.InstanceNorm2d(64),
                  nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            enc += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            enc += [ResidualBlock(in_features)]

        self.enc = nn.Sequential(*enc)

        self.Avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(16384,8)
        self.fcVar = nn.Linear(16384,8)

    def forward(self, c_img):
        feat = self.enc(c_img)
        feat = self.Avgpool(feat).view(feat.size(0), -1)
        return self.fc(feat), self.fcVar(feat)

class Halftone(nn.Module):
    """docstring for Halftone"""
    def __init__(self, kernel_size = 3, img_size = 256):
        super(Halftone, self).__init__()
        self.kernel_size = kernel_size

        BW_filter = torch.from_numpy(np.array([[1/3]])).unsqueeze(0)
        self.ToBW = nn.Conv2d(3, 1, kernel_size = 1, stride = 1, padding = 0, bias=False)
        self.ToBW.weight.data = BW_filter.float().unsqueeze(0).expand(1, 3, 1, 1)

        Mean_filter = torch.from_numpy(np.ones([kernel_size, kernel_size])/(kernel_size**2)).unsqueeze(0)
        self.Mean = nn.Conv2d(1, 1, kernel_size = kernel_size, stride = kernel_size, padding = int((kernel_size-1)/2), bias=False)
        self.Mean.weight.data = Mean_filter.float().unsqueeze(0).expand(1, 1, kernel_size, kernel_size)

        # Gaussian_filter = np.array([[1, 4, 7, 4, 1],
        #                            [4, 16, 26, 16, 4],
        #                            [7, 26, 41, 26, 7],
        #                            [4, 16, 26, 16, 4],
        #                            [1, 4, 7, 4, 1]])
        Gaussian_filter = np.array([[1,2,1],
                                    [2,4,2],
                                    [1,2,1]])
        # Gaussian_filter = Gaussian_filter/(np.sum(Gaussian_filter)*(self.kernel_size-1)/2)
        Gaussian_filter = Gaussian_filter/1.5
        Gaussian_filter_torch = torch.from_numpy(Gaussian_filter).unsqueeze(0)
        self.upsample = nn.ConvTranspose2d(1, 1, kernel_size = kernel_size, stride = kernel_size, padding = int((kernel_size-1)/2), bias=False)
        self.upsample.weight.data = Gaussian_filter_torch.float().unsqueeze(0).expand(1, 1, kernel_size, kernel_size)

        self.activation = nn.Tanh()

    def forward(self, input, eps=1e-3):
        bw_ph = self.ToBW(input)
        output = self.Mean(bw_ph)
        output = output*(self.kernel_size-1)/2
        output = self.upsample(output)
        output = self.activation(output)

        return bw_ph, output

class G_Unet_add_input(nn.Module):
    def __init__(self, input_nc=2, output_nc=3, nz=8, num_downs=7, ngf=64,
                 norm_layer=functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True),
                 nl_layer=functools.partial(nn.ReLU, inplace=True),
                 use_dropout=True, upsample='bilinear', use_dense=False):
        super(G_Unet_add_input, self).__init__()
        self.nz = nz
        max_nchn = 8
        # construct unet structure
        unet_block = UnetBlock(ngf * max_nchn, ngf * max_nchn, ngf * max_nchn,
                               innermost=True, use_dense=use_dense, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        for i in range(num_downs - 5):
            unet_block = UnetBlock(ngf * max_nchn, ngf * max_nchn, ngf * max_nchn, unet_block,
                                   norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout, upsample=upsample)
        unet_block = UnetBlock(ngf * 4, ngf * 4, ngf * max_nchn, unet_block,
                               norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock(ngf * 2, ngf * 2, ngf * 4, unet_block,
                               norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock(ngf, ngf, ngf * 2, unet_block,
                               norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock(input_nc + nz, output_nc, ngf, unet_block,
                               outermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)

        self.Unet = unet_block

    def forward(self, x, edge=None, z=None):
        if self.nz > 0:
            if len(z.size()) != len(x.size()):
                z_img = z.view(z.size(0), z.size(1), 1, 1).expand(
                    z.size(0), z.size(1), x.size(2), x.size(3))
            else:
                z_img = z
            x_with_z = torch.cat([x, z_img], 1)
        else:
            x_with_z = x  # no z

        if edge is not None:
            x_with_z = torch.cat([x_with_z, edge], 1)

        return self.Unet(x_with_z)

class UnetBlock(nn.Module):
    def __init__(self, input_nc, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, use_dense=False,
                 norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic', padding_type='zero'):
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        self.use_dense = use_dense
        self.innermost = innermost
        p = 0
        downconv = []
        if padding_type == 'reflect':
            downconv += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            downconv += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)
        downconv += [nn.Conv2d(input_nc, inner_nc,
                               kernel_size=4, stride=2, padding=p)]
        # downsample is different from upsample
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc) if norm_layer is not None else None
        uprelu = nl_layer()
        upnorm = norm_layer(outer_nc) if norm_layer is not None else None

        if outermost:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = downconv
            up = [uprelu] + upconv + [nn.Tanh()]
            model = down + [submodule] + up

            self.model = nn.Sequential(*model)
        elif innermost:
            upconv = upsampleLayer(
                inner_nc, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            up = [uprelu] + upconv
            if upnorm is not None:
                up += [upnorm]
            if use_dense:
                self.down = nn.Sequential(*down)
                self.block_1 = DenseBlock(3, inner_nc, int(inner_nc/2), BottleneckBlock, 0.5)
                self.trans_1 = TransitionBlock(int(inner_nc+3*int(inner_nc/2)), inner_nc, dropRate=0.5)
                self.up = nn.Sequential(*up)
            else:
                model = down + up
                self.model = nn.Sequential(*model)
        else:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            if downnorm is not None:
                down += [downnorm]
            up = [uprelu] + upconv
            if upnorm is not None:
                up += [upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

            self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        elif self.innermost and self.use_dense:
            feat = self.down(x)
            feat = self.block_1(feat)
            feat = self.trans_1(feat)
            return torch.cat([self.up(feat), x], 1)
        else:
            return torch.cat([self.model(x), x], 1)

def upsampleLayer(inplanes, outplanes, upsample='basic', padding_type='zero'):
    # padding_type = 'zero'
    if upsample == 'basic':
        upconv = [nn.ConvTranspose2d(
            inplanes, outplanes, kernel_size=4, stride=2, padding=1)]
    elif upsample == 'bilinear':
        upconv = [nn.Upsample(scale_factor=2, mode='bilinear'),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0)]
    else:
        raise NotImplementedError(
            'upsample layer [%s] not implemented' % upsample)
    return upconv

class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)
    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)

        return out
        
class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class D(nn.Module):
    def __init__(self, input_nc=3, input_nz=8, ndf=64, n_layers=3,
                 norm_layer=nn.BatchNorm2d, num_D=2):
        super(D, self).__init__()
        # st()
        self.num_D = num_D
        if num_D == 1:
            layers = self.get_layers(input_nc, input_nz, ndf, n_layers, norm_layer)
            self.model = nn.Sequential(*layers)
        else:
            layers = self.get_layers(input_nc, input_nz, ndf, n_layers, norm_layer)
            self.add_module("model_0", nn.Sequential(*layers))
            self.down = nn.AvgPool2d(3, stride=2, padding=[
                                     1, 1], count_include_pad=False)
            for i in range(1, num_D):
                ndf_i = int(round(ndf / (2**i)))
                layers = self.get_layers(input_nc, input_nz, ndf_i, n_layers, norm_layer)
                self.add_module("model_%d" % i, nn.Sequential(*layers))

    def get_layers(self, input_nc, input_nz, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc + input_nz, ndf, kernel_size=kw,
                              stride=2, padding=padw), nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1,
                               kernel_size=kw, stride=1, padding=padw)]

        return sequence

    def forward(self, img, z=None):
        if z is not None:
            z_img = z.view(z.size(0), z.size(1), 1, 1).expand(
                    z.size(0), z.size(1), img.size(2), img.size(3))
            input = torch.cat([img, z_img], 1)
        else:
            input = img

        if self.num_D == 1:
            return self.model(input)
        result = []
        down = input
        for i in range(self.num_D):
            model = getattr(self, "model_%d" % i)
            result.append(model(down))
            if i != self.num_D - 1:
                down = self.down(down)
        return result

class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, predictions, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor list) - - tpyically the prediction output from a discriminator; supports multi Ds.
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        all_losses = []
        for prediction in predictions:
            if self.gan_mode in ['lsgan', 'vanilla']:
                target_tensor = self.get_target_tensor(prediction, target_is_real)
                loss = self.loss(prediction, target_tensor)
            elif self.gan_mode == 'wgangp':
                if target_is_real:
                    loss = -prediction.mean()
                else:
                    loss = prediction.mean()
            all_losses.append(loss)
        total_loss = sum(all_losses)
        return total_loss

class VGG19(nn.Module):
    """docstring for VGG19"""
    def __init__(self):
        super(VGG19, self).__init__()

        vgg_features = models.vgg19(pretrained=True).features
        self.vgg19_maxpool_1 = nn.Sequential()
        self.vgg19_relu4_1 = nn.Sequential()

        for x in range(5):
            self.vgg19_maxpool_1.add_module(str(x), vgg_features[x])
        for x in range(5,21):
            self.vgg19_relu4_1.add_module(str(x), vgg_features[x])

    def low_level_feat(self, img):
        return self.vgg19_maxpool_1(img)

    def forward(self, img):
        return self.vgg19_relu4_1(self.vgg19_maxpool_1(img))

class Hist_Model(nn.Module):
    """docstring for Hist_Model"""
    def __init__(self, class_n):
        super(Hist_Model, self).__init__()
        self.hist_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU())

        self.hist_fc = nn.Sequential(
            nn.Linear(65536, class_n),
            nn.Softmax())
        
    def forward(self, img):
        hist_feat = self.hist_conv(img)
        hist_feat = hist_feat.view(hist_feat.size(0),-1)
        return self.hist_fc(hist_feat)

def conv3x3(in_planes, out_planes, dilation = 1, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=int(dilation*(3-1)/2), dilation=dilation, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, dilation = 1, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes,dilation, stride)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #         # weight_init.xavier_normal()
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        # out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class MultipleBasicBlock(nn.Module):

    def __init__(self,input_feature,
                 block, num_blocks,
                 intermediate_feature = 64, dense = True):
        super(MultipleBasicBlock, self).__init__()
        self.dense = dense
        self.num_block = num_blocks
        self.intermediate_feature = intermediate_feature

        self.block1= nn.Sequential(*[
            nn.Conv2d(input_feature, intermediate_feature,
                      kernel_size=7, stride=1, padding=3, bias=True),
            nn.ReLU(inplace=True)
        ])

        # for i in range(1, num_blocks):
        self.block2 = block(intermediate_feature, intermediate_feature, dilation = 1) if num_blocks>=2 else None
        self.block3 = block(intermediate_feature, intermediate_feature, dilation = 1) if num_blocks>=3 else None
        self.block4 = block(intermediate_feature, intermediate_feature, dilation = 1) if num_blocks>=4 else None
        self.block5 = nn.Sequential(*[nn.Conv2d(intermediate_feature, 3 , (3, 3), 1, (1, 1))])

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def forward(self, img, img_inv):
        x = torch.cat([img, img_inv], 1)
        x = self.block1(x)
        x = self.block2(x) if self.num_block>=2 else x
        x = self.block3(x) if self.num_block>=3 else x
        x = self.block4(x) if self.num_block== 4 else x
        x = self.block5(x)
        return x

def MultipleBasicBlock_4(input_feature,intermediate_feature = 64):
    model = MultipleBasicBlock(input_feature,
                               BasicBlock,4 ,
                               intermediate_feature)
    return model