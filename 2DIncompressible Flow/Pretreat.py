import torch  
from SPnet import UDate
from SPnet import VDate
from SPnet import PDate
import torch.optim as optim
from tool import gradients
from tool import sech

loss1,loss2,loss3,loss4,loss5,loss6,loss7,loss8,loss9 = torch.nn.MSELoss(),torch.nn.MSELoss(),torch.nn.MSELoss(),torch.nn.MSELoss(),\
torch.nn.MSELoss(),torch.nn.MSELoss(),torch.nn.MSELoss(),torch.nn.MSELoss(),torch.nn.MSELoss()

def pre(n,Umodel,Vmodel,Pmodel,options):
    optU = optim.Adam([
    {'params': Umodel.parameters(),'lr':0.001},
    ],)
    optV = optim.Adam([
    {'params': Vmodel.parameters(),'lr':0.001},
    ],)
    optP = optim.Adam([
    {'params': Pmodel.parameters(),'lr':0.001},
    ],)
    step=0
    for i in range(n):
        #一次进入样本数
        optU.zero_grad()
        optV.zero_grad()
        optP.zero_grad()

        l1=l_U(Umodel,options)
        l2=l_V(Vmodel,options)
        l3=l_P(Pmodel,options)

        l1.backward()
        l2.backward()
        l3.backward()
        optU.step()
        optV.step()
        optP.step()
        if step==1:
            break

def l_U(Umodel,options):
    xyt=options['xyt']
    x=options['x']
    y=options['y']
    t=options['t']
    M=options['m']
    N=options['n']


    u=Umodel(x,y,t)
    #输入初始
    with torch.no_grad():
        u0=torch.clone(u)
        u0=-torch.cos(x)*torch.sin(y)
        #u0[0,...]=-torch.cos(x[0,:])*torch.sin(y[0,:])
        #u0[:,0:M,0],u0[:,(N-1)*M:N*M,0]=-torch.cos(x[:,0:M,0])*torch.sin(y[:,0:M,0])*torch.exp(-2*t[:,0:M,0]),\
        #                                -torch.cos(x[:,(N-1)*M:N*M,0])*torch.sin(y[:,(N-1)*M:N*M,0])*torch.exp(-2*t[:,(N-1)*M:N*M,0])
    
    l1=loss1(u,u0)
    return l1

def l_V(Vmodel,options):
    xyt=options['xyt']
    x=options['x']
    y=options['y']
    t=options['t']
    M=options['m']
    N=options['n']

    v=Vmodel(x,y,t)
    #输入初始
    with torch.no_grad():
        v0=torch.clone(v)
        v0=torch.sin(x)*torch.cos(y)
        #v0[0,...]=torch.sin(x[0,:])*torch.cos(y[0,:])
        #v0[:,0:(N-1)*M+1:M,0],v0[:,M-1:N*M:M,0]=torch.sin(x[:,0:(N-1)*M+1:M,0])*torch.cos(y[:,0:(N-1)*M+1:M,0])*torch.exp(-2*t[:,0:(N-1)*M+1:M,0]),\
        #                                        torch.sin(x[:,M-1:N*M:M,0])*torch.cos(y[:,M-1:N*M:M,0])*torch.exp(-2*t[:,M-1:N*M:M,0])
    
    l2=loss2(v,v0)
    return l2
    
def l_P(Pmodel,options):
    xyt=options['xyt']
    x=options['x']
    y=options['y']
    t=options['t']

    p=Pmodel(x,y,t)
    #输入初始孤立波
    with torch.no_grad():
         p0=torch.clone(p)
         p0=-0.25*(torch.cos(2*x)+torch.cos(2*y))
         #p0[0,...]=-0.25*(torch.cos(2*x[0,:])+torch.cos(2*y[0,:]))

    
    l3=loss3(p,p0)
    return l3
