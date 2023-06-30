import torch 
import math 
from SPnet import UDate
from SPnet import VDate
from SPnet import PDate
import torch.optim as optim
from tool import gradients
from tool import sech

loss1,loss2,loss3,loss4,loss5,loss6,loss7,loss8,loss9 = torch.nn.MSELoss(),torch.nn.MSELoss(),torch.nn.MSELoss(),torch.nn.MSELoss(),\
torch.nn.MSELoss(),torch.nn.MSELoss(),torch.nn.MSELoss(),torch.nn.MSELoss(),torch.nn.MSELoss()

def pre(n,Tmodel,options):
    optT = optim.Adam([
    {'params': Tmodel.parameters(),'lr':0.000001},
    ],)
    for step in range(n):
        #一次进入样本数
        optT.zero_grad()

        l1=l_T(Tmodel,options)

        l1.backward()
        optT.step()
        if step==-1:
            break

def l_T(Tmodel,options):
    x=options['x']
    y=options['y']
    t=options['t']
    m=options['m']
    n=options['n']
    #输入初始场 正太分布 位置（22.5，2.5） 尺度1m
    with torch.no_grad():
        d2=(x[0,...]-22.5)**2+(y[0,...]-2.5)**2
        T0=10*torch.exp(-d2/2)/math.sqrt(2*math.pi)

    T=Tmodel(x[0,...],y[0,...],t[0,...])
    l1=loss1(T,T0)
    return l1
