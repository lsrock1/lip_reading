import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .RCA import RCAttention
from .Cbam import CBAM


class ConvFrontend(nn.Module):
    def __init__(self, options):
        super(ConvFrontend, self).__init__()
        self.attention = options['model']['attention']
        if self.attention and self.attention.startswith('cbam'):
            self.attn = CBAM(64, 1, 4, 8, 2, dropout=options['model']['attention_dropout'])
        elif self.attention and self.attention.startswith('se'):
            self.attn = CBAM(64, 1, 4, 8, 2, no_spatial=True, dropout=options['model']['attention_dropout'])
        elif self.attention and self.attention.startswith('tcbam'):
            self.attn = CBAM(64, 1, 4, 8, 2, no_temporal=False, dropout=options['model']['attention_dropout'])
        else:
            self.attn = None
        self.conv = nn.Conv3d(1, 64, (5,7,7), stride=(1,2,2), padding=(2,3,3))
        self.norm = nn.BatchNorm3d(64)
        self.pool = nn.MaxPool3d((1,3,3), stride=(1,2,2), padding=(0,1,1))

    def forward(self, input, landmark=False):
        #return self.conv(input)
        # [32, 64, 29, 28, 28]
        output = self.pool(F.relu(self.norm(self.conv(input))))
        if self.attn:
            if self.attention and self.attention.endswith('lmk'):
                landmark = landmark.view(-1, 112, 112).unsqueeze(1)
            output, landmark = self.attn(
                output.transpose(1, 2).contiguous().view(-1, 64, 28, 28), landmark)
        return output, landmark