import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .RCA import RCAttention
from .Cbam import CBAM
from .CoordConv import AddCoords


class ConvFrontend(nn.Module):
    def __init__(self, options):
        super(ConvFrontend, self).__init__()
        self.attention = options['model']['attention']
        self.coord = options['model']['coord']
        dim = 1
        if options['input']['landmark'] and not options['input']['landmark_seperate']:
            dim += 1
        if self.coord:
            dim += 2
            self.addCoord = AddCoords()
        if self.attention and self.attention.startswith('tcbam'):
            self.attn = CBAM(64, dim, 4, 8, 2, no_spatial=True, no_temporal=False)
        else:
            self.attn = None
        self.conv = nn.Conv3d(dim, 64, (5,7,7), stride=(1,2,2), padding=(2,3,3), bias=False)
        self.norm = nn.BatchNorm3d(64)
        self.pool = nn.MaxPool3d((1,3,3), stride=(1,2,2), padding=(0,1,1))

    def forward(self, x):
        #return self.conv(input)
        # [32, 64, 29, 28, 28]
        if self.coord:
            x = self.addCoord(x)
        x = self.pool(F.relu(self.norm(self.conv(x))))
        print(type(x))
        if self.attn:
            x = self.attn(
                x.transpose(1, 2).contiguous().view(-1, 64, 28, 28))
        return x