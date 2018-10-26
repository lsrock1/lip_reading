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
    def __init__(self):
        super().__init__()
        self.is_time_coord = False

    def forward(self, x):
        bs, _, t_dim, y_dim, x_dim = x.size()
        x_append = torch.arange(x_dim).view(1, 1, 1, 1, -1).expand(bs, -1, t_dim, y_dim, -1)
        y_append = torch.arange(y_dim).view(1, 1, 1, -1, 1).expand(bs, -1, t_dim, -1, x_dim)
        x_append = (x_append / (x_dim-1)) * 2 - 1
        y_append = (y_append / (y_dim-1)) * 2 - 1
        t_append = []
        if self.is_time_coord:
            t_append = torch.arange(t_dim).view(1, 1, -1, 1, 1).expand(bs, -1, -1, y_dim, x_dim)
            t_append = (t_append / (t_dim-1)) * 2 - 1
            t_append = [t_append]
        
        if torch.cuda.is_available():
            x_append = x_append.float().cuda()
            y_append = y_append.float().cuda()
            if t_append:
                t_append = [t_append.float().cuda()]

        return torch.cat([x , x_append, y_append] + t_append, dim=1)