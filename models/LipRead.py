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
            self.zip1 = nn.ModuleList(
                [nn.Sequential(nn.Linear(112, 56), nn.ReLU(), nn.Linear(56, 1)) for i in range(2)] +
                [nn.Conv2d(29, 29, kernel_size=1)]
            )
            self.zip2 = nn.ModuleList(
                [nn.Sequential(nn.Linear(112, 56), nn.ReLU(), nn.Linear(56, 1)) for i in range(2)] +
                [nn.Conv2d(29, 29, kernel_size=1)]
            )
        else:
            self.zip1 = None
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
        if self.zip1:
            x = self.attention(x, landmark)
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

    def attention(self, query, key, value=None):
        bs, c, length, h, _ = query.size()
        # bs 29 112 112
        # height
        # zip -> bs 29 112 (width -> 1)
        query = query.squeeze(1)
        key = key.squeeze(1)
        attn = F.softmax(
            torch.matmul(
                self.zip1[0](query),
                self.zip1[1](key).transpose(-2, -1)),
                dim=-1)
        result = torch.matmul(attn, self.zip1[2](query))
        # zip -> bs 29 112 (height -> 1)
        # attn bs 29 112(q) 112(k)
        attn = F.softmax(
            torch.matmul(
                self.zip2[0](query.transpose(-2, -1)),
                self.zip2[1](key.transpose(-2, -1)).transpose(-2, -1)),
                dim=-1)
        return query.unsqueeze(1), torch.matmul(self.zip2[2](result), attn.transpose(-2, -1)).view(bs, 1, length, h, h)

