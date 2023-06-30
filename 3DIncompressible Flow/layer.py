import math
import torch
from torch import nn

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
 
    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x

class Respart(nn.Module):
    def __init__(self,inum,n):
        super(Respart, self).__init__()
        self.linear=nn.Sequential(
            #nn.LayerNorm(inum),
            #nn.Tanh(),
            Swish(),
            nn.Linear(inum,n),
            #nn.LayerNorm(n),
            #nn.Tanh(),
            Swish(),
            nn.Linear(n, n),
        )
        self.short_cut=nn.Sequential()
        if inum!=n:
            self.short_cut.add_module('cut',nn.Linear(inum,n))
 
    def forward(self, x):
        out = self.linear(x)
        out=out+self.short_cut(x)
        return out

class Rescovjump(nn.Module):
    def __init__(self,inum,n):
        super(Respart, self).__init__()
        self.linear=nn.Sequential(
            nn.Conv1d(in_channels=inum, out_channels=n, kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Conv1d(in_channels=inum, out_channels=n, kernel_size=3, padding=1),
        )
        self.Swish=Swish()
 
    def forward(self, x):
        out = self.linear(x)
        out=out+x
        out=self.Swish(out)
        return out