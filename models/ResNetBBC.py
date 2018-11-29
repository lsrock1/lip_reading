# Adapted from TorchVision's ResNet to use a custom frontend and backend.
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
#
# BSD 3-Clause License
#
# Copyright (c) Soumith Chintala 2016,
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch.nn as nn
from .RCA import RCAttention
from .Cbam import CBAM
import math
import torch.nn.functional as F
import torch
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, attention=False, dropout=0.2):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        if attention and attention.startswith('cbam'):
            self.attn = CBAM(planes, inplanes, stride, dropout=dropout)
        elif attention and attention.startswith('se'):
            self.attn = CBAM(planes, inplanes, stride, no_spatial=True, dropout=dropout)
        elif attention and attention.startswith('tcbam'):
            self.attn = CBAM(planes, inplanes, stride, no_temporal=False, dropout=dropout)
        else:
            self.attn = None

    def forward(self, x, att=None):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.attn:
            out, att = self.attn(out, att)

        out += residual
        out = self.relu(out)

        return out, att


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, attention=False, dropout=0.2):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        if attention and attention.startswith('cbam'):
            self.attn = CBAM(planes*4, inplanes, stride, dropout=dropout)
        elif attention and attention.startswith('se'):
            self.attn = CBAM(planes*4, inplanes, stride, no_spatial=True, dropout=dropout)
        elif attention and attention.startswith('tcbam'):
            self.attn = CBAM(planes*4, inplanes, stride, no_temporal=False, dropout=dropout)
        else:
            self.attn = None

    def forward(self, x, att=None):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.attn:
            out, att = self.attn(out, att)

        out += residual
        out = self.relu(out)

        return out, att

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, attention=False, dropout=0.2, temporal=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.attn = attention
        self.dropout = dropout
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.bn2 = nn.BatchNorm1d(num_classes)

        if temporal:
            self.r1 = nn.Sequential(
                TemporalUnflat(),
                nn.Conv3d(64, 64,
                        kernel_size=(5,1,1), stride=1, padding=(1,0,0), bias=False),
                nn.BatchNorm3d(64),
                TemporalFlat()
            )
            self.r2 = nn.Sequential(
                TemporalUnflat(),
                nn.Conv3d(128, 128,
                        kernel_size=(5,1,1), stride=1, padding=(1,0,0), bias=False),
                nn.BatchNorm3d(128),
                TemporalFlat()
            )
            self.r3 = nn.Sequential(
                TemporalUnflat(),
                nn.Conv3d(256, 256,
                        kernel_size=(5,1,1), stride=1, padding=(1,0,0), bias=False),
                nn.BatchNorm3d(256),
                TemporalFlat()
            )
        else:
            self.r1, self.r2, self.r3 = None, None, None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, attention=self.attn, dropout=self.dropout))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, attention=self.attn, dropout=self.dropout))

        return AS(*layers)

    def forward(self, x, landmark=False):
        x, attn = self.layer1(x, landmark if self.attn and self.attn.endswith('lmk') else False)

        if self.r1:
            x = self.r1(x)
        x, attn = self.layer2(x, attn if self.attn and self.attn.endswith('lmk') else False)
        
        if self.r2:
            x = self.r2(x)
        x, attn = self.layer3(x, attn if self.attn and self.attn.endswith('lmk') else False)
        
        if self.r3:
            x = self.r3(x)
        x, _ = self.layer4(x, attn if self.attn and self.attn.endswith('lmk') else False)
        del _
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn2(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

class ResNetBBC(nn.Module):
    def __init__(self, options):
        super(ResNetBBC, self).__init__()
        self.batch_size = options["input"]["batch_size"]
        self.resnetModel = resnet34(False, num_classes=options["model"]["input_dim"], attention=options['model']['attention'], dropout=options['model']['attention_dropout'], temporal=options['model']['temporal'])
        self.input_dim = options['model']['input_dim']
        
    def forward(self, x, landmark=False):
        x = x.transpose(1, 2).contiguous().view(-1, 64, 28, 28)
        x = self.resnetModel(x, landmark)
        x = x.view(self.batch_size, -1, self.input_dim)
        return x


class AS(nn.Sequential):
    def forward(self, input, landmark=None):
        for module in self._modules.values():
            input, landmark = module(input, landmark)
        return input, landmark


class TemporalUnflat(nn.Module):
    def forward(self, x):
        bs, c, h, w = x.size()
        bs = int(bs/29)
        return x.view(bs, 29, c, h, w).transpose(1, 2).contiguous()

class TemporalFlat(nn.Module):
    def forward(self, x):
        bs, c, _, h, w = x.size()
        return x.transpose(1, 2).contiguous().view(-1, c, h, w)
