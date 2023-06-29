import os
import math
import torch
import layer
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from layer import Swish
from layer import Respart

class UDate(nn.Module):
    def __init__(self):
        super(UDate, self).__init__()
        self.sub=subnet(3,1,5,128)
    def forward(self,x,y,t):
        xyt=torch.cat([x,y,t],-1)
        u=self.sub(xyt)
        return u

class VDate(nn.Module):
    def __init__(self):
        super(VDate, self).__init__()
        self.sub=subnet(3,1,5,128)
    def forward(self,x,y,t):
        xyt=torch.cat([x,y,t],-1)
        v=self.sub(xyt)
        return v
    
class PDate(nn.Module):
    def __init__(self):
        super(PDate, self).__init__()
        self.sub=subnet(3,1,5,128)
    def forward(self,x,y,t):
        xyt=torch.cat([x,y,t],-1)
        p=self.sub(xyt)
        return p

class subnet(nn.Module):
    def __init__(self,input,output,layernum,partnum):
        super(subnet, self).__init__()
        self.fc0=nn.Linear(input,partnum)
        self.line=nn.Sequential()
        for i in range(layernum):
            #self.line.add_module('layer_{}'.format(i),Respart(partnum,partnum))
            self.line.add_module('layer_{}'.format(i),nn.Linear(partnum,partnum))
            self.line.add_module('swish_{}'.format(i),Swish())
        self.fc1=nn.Linear(partnum,output)
    def forward(self, x):
        x=self.fc0(x)
        x=Swish()(x)
        x=self.line(x)
        x=self.fc1(x)
        return x


