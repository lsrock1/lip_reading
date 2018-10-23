import torch
import torch.nn as nn
import torch.nn.functional as F

import re

from .ConvFrontend import ConvFrontend
from .ResNetBBC import ResNetBBC
from .LSTMBackend import LSTMBackend
from .ConvBackend import ConvBackend

class LipRead(nn.Module):
    def __init__(self, options):
        super(LipRead, self).__init__()
        self.frontend = ConvFrontend(options)
        self.resnet = ResNetBBC(options)
        self.lstm = LSTMBackend(options)

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

    def forward(self, input):
        output = self.lstm(self.resnet(self.frontend(input)))

        return output

    def loss(self):
        return self.lstm.loss

    def validator_function(self):
        return self.lstm.validator
