import torch
import torch.nn as nn
import torch.nn.functional as F

import re

from .ConvFrontend import ConvFrontend
from .ResNetBBC import ResNetBBC
from .LSTMBackend import LSTMBackend
from .ConvBackend import ConvBackend
from .Densenet import Densenet

class LipRead(nn.Module):
    def __init__(self, options):
        super(LipRead, self).__init__()
        self.landmarkloss = options['training']['landmarkloss']
        self.frontend = ConvFrontend(options)
        if options['model']['front'] == 'DENSENET':
            self.model = Densenet(options)
            self.resnet = None
        else:
            self.resnet = ResNetBBC(options)
            self.model = None
        self.lstm = LSTMBackend(options)
        if options['model']['seperate'] == 'cord':
            self.embedding = nn.Sequential(
                nn.Linear(40, 64),
                nn.BatchNorm1d(29),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.BatchNorm1d(29),
                nn.ReLU()
            )
        else:
            self.embedding = None

        if options['model']['seperate'] == 'attention':
            self.attn = nn.ModuleList(
                [nn.Conv2d(29, 29, kernel_size=2, stride=2) for i in range(3)]
            )
            self.up = nn.ConvTranspose2d(29, 29, kernel_size=2, stride=2)
        else:
            self.attn = None
        #function to initialize the weights and biases of each module. Matches the
        #classname with a regular expression to determine the type of the module, then
        #initializes the weights for it.
        def weights_init(m):
            classname = m.__class__.__name__
            if re.search("Conv[123]d", classname):
                nn.init.xavier_uniform_(m.weight)
            elif re.search("BatchNorm[123]d", classname):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0)
            elif re.search("Linear", classname):
                m.bias.data.fill_(0)

        #Apply weight initialization to every module in the model.
        self.apply(weights_init)

    def forward(self, x, landmark=None):
        if self.attn:
            x = self.attention(x, landmark, x)
        if self.embedding:
            landmark = self.embedding(landmark.view(landmark.size(0), 29, -1))
        if self.model:
            x = self.model(self.frontend(x))
        else:
            x = self.resnet(self.frontend(x))
        if self.landmarkloss and self.training:
            x, dot = x
            x = self.lstm(x)
            return x, dot
        if self.embedding:
            x = torch.cat([x, landmark], dim=2)
        x = self.lstm(x)
        return x

    def loss(self):
        return self.lstm.loss

    def validator_function(self):
        return self.lstm.validator

    def attention(self, query, key, value):
        bs, c, length, h, _ = query.size()
        query = self.attn[0](query.squeeze(1)).view(bs, length, -1, 1)
        key = self.attn[1](key.squeeze(1)).view(bs, length, -1, 1).transpose(-2, -1)
        value = self.attn[2](value.squeeze(1)).view(bs, length, -1, 1)
        attn = F.softmax(torch.matmul(query, key), dim=-1)
        del query, key
        return self.up(torch.matmul(attn, v).view(bs, 1, length, int(h/2), int(h/2)).contiguous())

