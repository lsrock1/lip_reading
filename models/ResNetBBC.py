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

    def __init__(self, inplanes, planes, stride=1, downsample=None, attention=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        if attention and attention.startswith('cbam'):
            self.attn = CBAM(planes, inplanes, stride)
        elif attention and attention.startswith('se'):
            self.attn = CBAM(planes, inplanes, stride, no_spatial=True)
        elif attention and attention.startswith('tcbam'):
            self.attn = CBAM(planes, inplanes, stride, no_temporal=False)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None, attention=False):
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
            self.attn = CBAM(planes*4, inplanes, stride)
        elif attention and attention.startswith('se'):
            self.attn = CBAM(planes*4, inplanes, stride, no_spatial=True)
        elif attention and attention.startswith('tcbam'):
            self.attn = CBAM(planes*4, inplanes, stride, no_temporal=False)
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

    def __init__(self, block, layers, num_classes=1000, attention=False, fpn=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.fpn = fpn
        self.attn = attention
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        if self.fpn:
            dim = int(512*block.expansion/4)
            self.fpn1 = nn.Linear(64, dim)
            self.fpn2 = nn.Linear(64*block.expansion, dim)
            self.fpn3 = nn.Linear(128*block.expansion, dim)
            self.fpn4 = nn.Linear(256*block.expansion, dim)
            self.fpn5 = nn.Linear(512*block.expansion, dim)
            self.last = nn.Linear(5*dim, num_classes)
        else:
            self.avgpool = nn.AvgPool2d(4, stride=1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.bn2 = nn.BatchNorm1d(num_classes)
        
        if attention == 'bcbam':
            self.r1 = CBAM(64*block.expansion, 64, 1)
            self.r2 = CBAM(128*block.expansion, 64*block.expansion, 2)
            self.r3 = CBAM(256*block.expansion, 128*block.expansion, 2)
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
        layers.append(block(self.inplanes, planes, stride, downsample, attention=self.attn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, attention=self.attn))

        return AS(*layers)

    def forward(self, x, landmark=False):
        if self.fpn:
            fpn = [F.relu(self.fpn1(F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)))]
        x, attn = self.layer1(x, landmark if self.attn and self.attn.endswith('lmk') else False)
        if self.fpn:
            fpn.append(F.relu(self.fpn2(F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1))))
        if self.r1:
            x, landmark = self.r1(x, landmark)

        x, attn = self.layer2(x, attn if self.attn and self.attn.endswith('lmk') else False)
        if self.fpn:
            fpn.append(F.relu(self.fpn3(F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1))))
        if self.r2:
            x, landmark = self.r2(x, landmark)

        x, attn = self.layer3(x, attn if self.attn and self.attn.endswith('lmk') else False)
        if self.fpn:
            fpn.append(F.relu(self.fpn4(F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1))))
        if self.r3:
            x, _ = self.r3(x, landmark)
            del _
        x, _ = self.layer4(x, attn if self.attn and self.attn.endswith('lmk') else False)
        if self.fpn:
            fpn.append(F.relu(self.fpn5(F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1))))
        del _
        if self.fpn:
            x = self.last(torch.cat(fpn, dim=1))
        else:
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
        self.resnetModel = resnet34(False, num_classes=options["model"]["input_dim"], attention=options['model']['attention'], fpn=options['model']['fpn'])
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