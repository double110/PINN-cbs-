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
    {'params': Umodel.parameters(),'lr':0.0001},
    ],)
    optV = optim.Adam([
    {'params': Vmodel.parameters(),'lr':0.0001},
    ],)
    optP = optim.Adam([
    {'params': Pmodel.parameters(),'lr':0.000001},
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
    a=options['a']
    aph=options['apha']
    g=options['g']
    H=options['H']
    xyt=options['xyt']
    x=options['x']
    y=options['y']
    t=options['t']
    m=options['m']
    n=options['n']
    #输入初始孤立波
    with torch.no_grad():
        h=a*sech(1/2*((3*a)**0.5)*(x-1/aph + t))**2
        u0=-(H+1/2*a)*h/(h+(x+ t)*aph)
        u0[:,0:m,0],u0[:,(n-1)*m:n*m,0]=0,0

    u=Umodel(x,y,t)
    l1=loss1(u,u0)
    return l1

def l_V(Vmodel,options):
    a=options['a']
    aph=options['apha']
    g=options['g']
    H=options['H']
    xyt=options['xyt']
    x=options['x']
    y=options['y']
    t=options['t']
    #输入初始孤立波
    Vture=torch.zeros(y.shape)

    v=Vmodel(x,y,t)
    l2=loss2(v,Vture)
    return l2
    
def l_P(Pmodel,options):
    a=options['a']
    aph=options['apha']
    g=options['g']
    H=options['H']
    xyt=options['xyt']
    x=options['x']
    y=options['y']
    t=options['t']
    #输入初始孤立波
    with torch.no_grad():
         h0=a*sech(1/2*((3*a)**0.5)*(x-1/aph + t))**2+H

    h=Pmodel(x,y,t)
    l3=loss3(h,h0)
    return l3
