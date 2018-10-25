import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class ConvFrontend(nn.Module):
    def __init__(self, options):
        super(ConvFrontend, self).__init__()
        dim = 1
        if options['model']['coord']:
            dim += 2
        if options['model']['landmark']:
            dim +=1
        self.conv = nn.Conv3d(dim, 64, (5,7,7), stride=(1,2,2), padding=(2,3,3))
        self.norm = nn.BatchNorm3d(64)
        self.pool = nn.MaxPool3d((1,3,3), stride=(1,2,2), padding=(0,1,1))
        if options['model']['coord']:
            self.coord = AddCoords()
        else:
            self.coord = False

    def forward(self, input):
        #return self.conv(input)
        if self.coord:
            input = self.coord(input)
        output = self.pool(F.relu(self.norm(self.conv(input))))
        return output


class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret