import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .ResNetBBC import RCAttention


class ConvFrontend(nn.Module):
    def __init__(self, options):
        super(ConvFrontend, self).__init__()
        dim = 1
        if options['model']['landmark']:
            dim += 1
        if options['model']['seperate'] == 'attention':
            dim -= 1
            self.attn = RCAttention(64, 1, 4, 8, 2)
        else:
            self.attn = None
        self.conv = nn.Conv3d(dim, 64, (5,7,7), stride=(1,2,2), padding=(2,3,3))
        self.norm = nn.BatchNorm3d(64)
        self.pool = nn.MaxPool3d((1,3,3), stride=(1,2,2), padding=(0,1,1))

    def forward(self, input, landmark=None):
        #return self.conv(input)
        # [32, 64, 29, 28, 28]
        output = self.pool(F.relu(self.norm(self.conv(input))))
        if self.attn:
            output, attn = self.attn(
                output.transpose(1, 2).contiguous().view(-1, 64, 28, 28), 
                landmark.view(-1, 112, 112).unsqueeze(1))
            return output, attn
        else:
            return output