import torch
from torch import nn
import torch.nn.functional as F


class RCAttention(nn.Module):
    def __init__(self, channel, inchannel, stride, kernel_size=3, padding=1):
        super(RCAttention, self).__init__()
        self.attn = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=1), 
                nn.ReLU()
                ) for i in range(4)]
        )
        self.zip = nn.Sequential(
            nn.Conv2d(inchannel, channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

    def resize(self, x):
        return self.zip(x)

    def forward(self, x, att):
        bs, c, h, _ = x.size()
        # bs 29 112 112
        # height
        # zip -> bs 29 112 112
        # query: bs 29 112(h) 112(w) * bs 29 112(w) 112(h) ->
        # bs 29 112(h) 112(energy[h]) * bs 29 112(h) 112(w)
        query = self.attn[0](x)
        key = self.attn[1](att)
        value = self.attn[2](x)
        attn1 = F.softmax(
            torch.matmul(
                query,
                key.transpose(-2, -1)),
                dim=-1)
        # zip -> bs 29 112 112
        # key: bs 29 112(w) 112(h) * query: bs 29 112(h) 112(w)
        # bs 29 112(h) 112(w) * bs 29 112(energy[w]) 112(w)
        attn2 = F.softmax(
            torch.matmul(
                key.transpose(-2, -1),
                query),
                dim=-1)
        return self.attn[3](torch.matmul(torch.matmul(attn1, value), attn2)).view(bs, -1, h, h)