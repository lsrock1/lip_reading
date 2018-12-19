import torch
import torch.nn as nn


class AddCoords(nn.Module):
    def __init__(self):
        super().__init__()
        self.is_time_coord = False
        self.x_append = None
        self.y_append = None
        self.t_append = None

    def forward(self, x):
        if not self.x_append:
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
            self.x_append = x_append
            self.y_append = y_append
            self.t_append = t_append
            return torch.cat([x , x_append, y_append] + t_append, dim=1)
        else:
            if self.t_append:
                t_append = self.t_append.clone()
            else:
                t_append = []
            return torch.cat([x, self.x_append.clone(), self.y_append.clone()] + t_append, dim=1)
        