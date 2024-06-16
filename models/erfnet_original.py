# ERFNet full model definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.transforms as T


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1):
        super(ConvBlock, self).__init__()
        self.strides = strides
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
        )

        self.conv11 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, padding=0)

    def forward(self, x):
        out1 = self.block(x)
        out2 = self.conv11(x)
        out = out1 + out2
        return out


class DownsamplerBlock(nn.Module):
    '''Downsampling by concatenating parallel output of
    3x3 conv(stride=2) & max-pooling'''

    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput - ninput,
                              (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class non_bottleneck_1d(nn.Module):
    '''Factorized residual layer
    dilation can gather more context (a.k.a. atrous conv)'''

    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        # factorize 3x3 conv to (3x1) x (1x3) x (3x1) x (1x3)
        # non-linearity can be added to each decomposed 1D filters
        self.conv3x1_1 = nn.Conv2d(
            chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(
            chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(
            1 * dilated, 0), bias=True, dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(
            0, 1 * dilated), bias=True, dilation=(1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = F.relu(output)  # non-linearity

        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)

        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.relu(output + input)  # +input = identity (residual connection)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 16)  # (N, 16, 256, 256)

        self.layers1 = nn.ModuleList()
        self.layers1.append(DownsamplerBlock(16, 64))  # (N, 64, 128, 128)

        for x in range(0, 5):  # 5 times (no dilation)
            self.layers1.append(non_bottleneck_1d(64, 0.03, 1))  # (N, 64, 128, 128)

        self.layers2 = nn.ModuleList()
        self.layers2.append(DownsamplerBlock(64, 128))  # (N, 128, 64, 64)
        for x in range(0, 2):  # 2 times (with dilation)
            self.layers2.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers2.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers2.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers2.append(non_bottleneck_1d(128, 0.3, 16))  # (N, 128, 64, 64)

    def forward(self, input):
        #feats = {}
        '''input: (N, 3, 512, 512)'''
        output = self.initial_block(input)  # (N, 16, 256, 256)
        #feats['layer0'] = output

        for layer in self.layers1:
            output = layer(output)  # (N, 64, 128, 128)
        #feats['layer1'] = output

        for layer in self.layers2:
            output = layer(output)  # (N, 128, 64, 64)
        #feats['layer2'] = output

        return output #, feats


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        # do not use max-unpooling operation
        self.conv = nn.ConvTranspose2d(
            ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class Decoder(nn.Module):
    '''small (not symmetric) decoder that upsamples encoder's output
    by fine-tuning the details'''

    def __init__(self, num_classes=3):
        super().__init__()

        self.layers1 = nn.ModuleList()
        self.layers1.append(UpsamplerBlock(128, 64))
        self.layers1.append(non_bottleneck_1d(64, 0, 1))
        self.layers1.append(non_bottleneck_1d(64, 0, 1))

        self.layers2 = nn.ModuleList()
        self.layers2.append(UpsamplerBlock(64, 16))
        self.layers2.append(non_bottleneck_1d(16, 0, 1))
        self.layers2.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d(
            16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input  # (N, 128, 64, 64)
        #feats = {}
        for layer in self.layers1:
            output = layer(output)  # (N, 64, 128, 128)
        #feat = output
        #feats['layer1'] = output

        for layer in self.layers2:
            output = layer(output)  # (N, 16, 256, 256)
        #feats['layer2'] = output
        feat = output

        output = self.output_conv(output)  # (N, 2, 512, 512)

        return output, feat
