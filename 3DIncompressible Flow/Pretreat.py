import torch  
import math
import tool
from SPnet import UDate
from SPnet import VDate
from SPnet import PDate
import torch.optim as optim
from tool import gradients
from tool import sech

loss1,loss2,loss3,loss4,loss5,loss6,loss7,loss8,loss9 = torch.nn.MSELoss(),torch.nn.MSELoss(),torch.nn.MSELoss(),torch.nn.MSELoss(),\
torch.nn.MSELoss(),torch.nn.MSELoss(),torch.nn.MSELoss(),torch.nn.MSELoss(),torch.nn.MSELoss()

A,D=tool.A,tool.D

def pre(n,Umodel,Vmodel,Wmodel,Pmodel,options):
    optU = optim.Adam([
    {'params': Umodel.parameters(),'lr':0.000001},
    ],)
    optV = optim.Adam([
    {'params': Vmodel.parameters(),'lr':0.000001},
    ],)
    optW = optim.Adam([
    {'params': Wmodel.parameters(),'lr':0.000001},
    ],)
    optP = optim.Adam([
    {'params': Pmodel.parameters(),'lr':0.000001},
    ],)
    step=0
    for step in range(n):
        #一次进入样本数
        optU.zero_grad()
        optV.zero_grad()
        optW.zero_grad()
        optP.zero_grad()

        l1=l_U(Umodel,options)
        l2=l_V(Vmodel,options)
        l3=l_W(Wmodel,options)
        l4=l_P(Pmodel,options)

        l1.backward()
        l2.backward()
        l3.backward()
        l4.backward()
        optU.step()
        optV.step()
        optW.step()
        optP.step()
        if step==-1:
            break
#正确的结果不应该除2 除2是为了使结果不准
def l_U(Umodel,options):
    x=options['x']
    y=options['y']
    z=options['z']
    t=options['t']
    M=options['m']
    N=options['n']
    L=options['l']


    u=Umodel(x,y,z,t)
    #输入初始
    with torch.no_grad():
        u0=torch.clone(u)
        u0=-A*(torch.exp(A*x)*torch.sin(A*y+D*z)+torch.exp(A*z)*torch.cos(A*x+D*y))*torch.exp(-D**2*t/2)
        #u0[0,...]=-torch.cos(x[0,:])*torch.sin(y[0,:])
        #u0[:,0:M,0],u0[:,(N-1)*M:N*M,0]=-torch.cos(x[:,0:M,0])*torch.sin(y[:,0:M,0])*torch.exp(-2*t[:,0:M,0]),\
        #                                -torch.cos(x[:,(N-1)*M:N*M,0])*torch.sin(y[:,(N-1)*M:N*M,0])*torch.exp(-2*t[:,(N-1)*M:N*M,0])
    
    l1=loss1(u,u0)
    return l1

def l_V(Vmodel,options):
    x=options['x']
    y=options['y']
    z=options['z']
    t=options['t']
    M=options['m']
    N=options['n']
    L=options['l']


    v=Vmodel(x,y,z,t)
    #输入初始
    with torch.no_grad():
        v0=torch.clone(v)
        v0=-A*(torch.exp(A*y)*torch.sin(A*z+D*x)+torch.exp(A*x)*torch.cos(A*y+D*z))*torch.exp(-D**2*t/2)
        #u0[0,...]=-torch.cos(x[0,:])*torch.sin(y[0,:])
        #u0[:,0:M,0],u0[:,(N-1)*M:N*M,0]=-torch.cos(x[:,0:M,0])*torch.sin(y[:,0:M,0])*torch.exp(-2*t[:,0:M,0]),\
        #                                -torch.cos(x[:,(N-1)*M:N*M,0])*torch.sin(y[:,(N-1)*M:N*M,0])*torch.exp(-2*t[:,(N-1)*M:N*M,0])
    
    l2=loss1(v,v0)
    return l2

def l_W(Wmodel,options):
    x=options['x']
    y=options['y']
    z=options['z']
    t=options['t']
    M=options['m']
    N=options['n']
    L=options['l']


    w=Wmodel(x,y,z,t)
    #输入初始
    with torch.no_grad():
        w0=torch.clone(w)
        w0=-A*(torch.exp(A*z)*torch.sin(A*x+D*y)+torch.exp(A*y)*torch.cos(A*z+D*x))*torch.exp(-D**2*t/2)        #u0[0,...]=-torch.cos(x[0,:])*torch.sin(y[0,:])
        #u0[:,0:M,0],u0[:,(N-1)*M:N*M,0]=-torch.cos(x[:,0:M,0])*torch.sin(y[:,0:M,0])*torch.exp(-2*t[:,0:M,0]),\
        #                                -torch.cos(x[:,(N-1)*M:N*M,0])*torch.sin(y[:,(N-1)*M:N*M,0])*torch.exp(-2*t[:,(N-1)*M:N*M,0])
    
    l3=loss3(w,w0)
    return l3

def l_P(Pmodel,options):
    x=options['x']
    y=options['y']
    z=options['z']
    t=options['t']

    p=Pmodel(x,y,z,t)
    #输入初始孤立波
    with torch.no_grad():
         p0=torch.clone(p)
         p0=-A**2/2*(torch.exp(2*A*x)+torch.exp(2*A*y)+torch.exp(2*A*z)
                     +2*torch.sin(A*x+D*y)*torch.cos(A*z+D*x)*torch.exp(A*(y+z))
                     +2*torch.sin(A*y+D*z)*torch.cos(A*x+D*y)*torch.exp(A*(z+x))
                     +2*torch.sin(A*z+D*x)*torch.cos(A*y+D*z)*torch.exp(A*(x+y))
                     )*torch.exp(-2*D**2*t/2)
         #p0[0,...]=-0.25*(torch.cos(2*x[0,:])+torch.cos(2*y[0,:]))

    
    l4=loss4(p,p0)
    return l4
