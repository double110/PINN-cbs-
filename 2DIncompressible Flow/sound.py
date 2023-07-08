import os
import torch
import math
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.cuda.amp import autocast as autocast # 注意！！！
#import SPnet
from SPnet import UDate
from SPnet import VDate
from SPnet import PDate
from tool import rand_xy
from tool import rand_t
from tool import sech
from tool import meandepth
from Pretreat import pre
from loss import dUV_loss
from loss import P_loss
from loss import dP_loss
from loss import U_loss
from loss import V_loss

torch.set_default_tensor_type(torch.FloatTensor)
#torch.set_default_tensor_type(torch.DoubleTensor)
torch.backends.cuda.matmul.allow_tf32 = False  # 禁止矩阵乘法使用tf32
torch.backends.cudnn.allow_tf32 = False        # 禁止卷积使用tf32
torch.set_printoptions(8)

#让我们检查一下 torch.cuda是否可用，否则我们继续使用CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


#设置
options ={ # defalt 
    'preset':1,
    'loadmodel':1,
    'X':2*math.pi,
    'Y':2*math.pi,
    'dx':2*math.pi/40,
    'dy':2*math.pi/40,
    'T':0.5,
    'dt':0.025,
    }

#模型
Umodel = UDate().to(device)
Vmodel = VDate().to(device)
Pmodel = PDate().to(device)

#输入
options['n']=round(options['X']/options['dx'])+1
options['m']=round(options['Y']/options['dy'])+1
options['k']=round(options['T']/options['dt'])+1
options['xy']=rand_xy(options['dx'],options['dy'],options['X'],options['Y']).to(device)
options['t']=rand_t(options['dt'],options['T']).repeat(1,options['n']*options['m'],1).to(device)
options['xyt']=torch.cat([options['xy'].repeat(options['k'],1,1),options['t']],dim=-1)
options['x'],options['y'],options['t']=options['xyt'].split(1,dim=-1)
options['tdt']=options['t']+options['dt']
#预处理
if options['preset']==0 :

    #Umodel.load_state_dict(torch.load('./preUmodel.pth'))
    #Vmodel.load_state_dict(torch.load('./preVmodel.pth'))
    #Pmodel.load_state_dict(torch.load('./prePmodel.pth'))
    #Umodel.load_state_dict(torch.load('./Umodel.pth'))
    #Vmodel.load_state_dict(torch.load('./Vmodel.pth'))
    #Pmodel.load_state_dict(torch.load('./Pmodel.pth'))

    pre(200000,Umodel,Vmodel,Pmodel,options)
    torch.save(Umodel.state_dict(), './preUmodel.pth')
    torch.save(Vmodel.state_dict(), './preVmodel.pth')
    torch.save(Pmodel.state_dict(), './prePmodel.pth')

    #保存参数
else:
    Umodel.load_state_dict(torch.load('./preUmodel.pth'))
    Vmodel.load_state_dict(torch.load('./preVmodel.pth'))
    Pmodel.load_state_dict(torch.load('./prePmodel.pth'))

#训练
if options['loadmodel']!=0 :
    Umodel.load_state_dict(torch.load('./Umodel.pth'))
    Vmodel.load_state_dict(torch.load('./Vmodel.pth'))
    Pmodel.load_state_dict(torch.load('./Pmodel.pth'))

optU = optim.Adam([
    {'params': Umodel.parameters(),'lr':0.0000001},
    ],)
optV = optim.Adam([
    {'params': Vmodel.parameters(),'lr':0.0000001},
    ],)
optP = optim.Adam([
    {'params': Pmodel.parameters(),'lr':0.000001},
    ],)

#scheduler = optim.lr_scheduler.StepLR(opt, step_size=30000, gamma=0.1)
lsum=0
xyt=options['xyt']
dt =options['dt']
tdt=options['tdt']
x=options['x'].detach()
x.requires_grad=True
y=options['y'].detach()
y.requires_grad=True
t=options['t'].detach()
K=options['k']
N=options['n']
M=options['m']
#约束条件 0时刻浪高和速度
with torch.no_grad():
    p=-0.25*(torch.cos(2*x[0,:])+torch.cos(2*y[0,:]))
    u=-torch.cos(x[0,:])*torch.sin(y[0,:])
    v=torch.sin(x[0,:])*torch.cos(y[0,:])
for step in range(10000):
    #options['xy']=torch.rand(N*M,2).to(device)*2*math.pi
    #options['t']=rand_t(options['dt'],options['T']).repeat(1,options['n']*options['m'],1).to(device)
    #options['xyt']=torch.cat([options['xy'].repeat(options['k'],1,1),options['t']],dim=-1).detach()
    #x,y,t=options['xyt'].split(1,dim=-1)
    #x.requires_grad=True
    #y.requires_grad=True
    #with torch.no_grad():
    #    p=-0.25*(torch.cos(2*x[0,:])+torch.cos(2*y[0,:]))
    #    u=-torch.cos(x[0,:])*torch.sin(y[0,:])
    #    v=torch.sin(x[0,:])*torch.cos(y[0,:])
    #step1 
    #为了节省内存 分段计算
    ub,vb,pb,dUdx,dVdy,dUa,dVa,dUadx,dVady,dUc,dVc,dUcdx,dVcdy,ddpdd,dpdx,dpdy,ukddpddx,ukddpddy=torch.zeros(x.shape,device=device),torch.zeros(x.shape,device=device),\
    torch.zeros(x.shape,device=device), torch.zeros(x.shape,device=device),torch.zeros(x.shape,device=device),torch.zeros(x.shape,device=device),\
    torch.zeros(x.shape,device=device),torch.zeros(x.shape,device=device) ,torch.zeros(x.shape,device=device),torch.zeros(x.shape,device=device),\
    torch.zeros(x.shape,device=device),torch.zeros(x.shape,device=device) ,torch.zeros(x.shape,device=device),torch.zeros(x.shape,device=device),\
    torch.zeros(x.shape,device=device) ,torch.zeros(x.shape,device=device),torch.zeros(x.shape,device=device),torch.zeros(x.shape,device=device),
    #for k in range(K):
    #    for j in range(M):
    #        ub[k,j*N:(j+1)*N],vb[k,j*N:(j+1)*N],pb[k,j*N:(j+1)*N],dUdx[k,j*N:(j+1)*N],dVdy[k,j*N:(j+1)*N],\
    #        dUa[k,j*N:(j+1)*N],dVa[k,j*N:(j+1)*N],dUadx[k,j*N:(j+1)*N],dVady[k,j*N:(j+1)*N],\
    #        dUc[k,j*N:(j+1)*N],dVc[k,j*N:(j+1)*N],dUcdx[k,j*N:(j+1)*N],dVcdy[k,j*N:(j+1)*N],\
    #        ddpdd[k,j*N:(j+1)*N],dpdx[k,j*N:(j+1)*N],dpdy[k,j*N:(j+1)*N],ukddpddx[k,j*N:(j+1)*N],ukddpddy[k,j*N:(j+1)*N]\
    #        =dUV_loss(Umodel,Vmodel,Pmodel,x[k,j*N:(j+1)*N],y[k,j*N:(j+1)*N],t[k,j*N:(j+1)*N],dt)
    for k in range(K):
            ub[k],vb[k],pb[k],dUdx[k],dVdy[k],\
            dUa[k],dVa[k],dUadx[k],dVady[k],\
            dUc[k],dVc[k],dUcdx[k],dVcdy[k],\
            ddpdd[k],dpdx[k],dpdy[k],ukddpddx[k],ukddpddy[k]\
            =dUV_loss(Umodel,Vmodel,Pmodel,x[k],y[k],t[k],dt)
    #浪高与速度约束
    with torch.no_grad():
        p0,u0,v0=torch.clone(pb),torch.clone(ub),torch.clone(vb)
        p0[0,...]=p
        u0[0,...]=u
        v0[0,...]=v
        u0[:,0:M,0],u0[:,(N-1)*M:N*M,0]=-torch.cos(x[:,0:M,0])*torch.sin(y[:,0:M,0])*torch.exp(-2*t[:,0:M,0]),\
                                        -torch.cos(x[:,(N-1)*M:N*M,0])*torch.sin(y[:,(N-1)*M:N*M,0])*torch.exp(-2*t[:,(N-1)*M:N*M,0])
        u0[:,0:(N-1)*M+1:M,0],u0[:,M-1:N*M:M,0]=-torch.cos(x[:,0:(N-1)*M+1:M,0])*torch.sin(y[:,0:(N-1)*M+1:M,0])*torch.exp(-2*t[:,0:(N-1)*M+1:M,0]),\
                                                -torch.cos(x[:,M-1:N*M:M,0])*torch.sin(y[:,M-1:N*M:M,0])*torch.exp(-2*t[:,M-1:N*M:M,0])
        #ub[:,0:M,0],ub[20:K,(N-1)*M:N*M,0]=0,0
        v0[:,0:M,0],v0[:,(N-1)*M:N*M,0]=torch.sin(x[:,0:M,0])*torch.cos(y[:,0:M,0])*torch.exp(-2*t[:,0:M,0]),\
                                        torch.sin(x[:,(N-1)*M:N*M,0])*torch.cos(y[:,(N-1)*M:N*M,0])*torch.exp(-2*t[:,(N-1)*M:N*M,0])
        v0[:,0:(N-1)*M+1:M,0],v0[:,M-1:N*M:M,0]=torch.sin(x[:,0:(N-1)*M+1:M,0])*torch.cos(y[:,0:(N-1)*M+1:M,0])*torch.exp(-2*t[:,0:(N-1)*M+1:M,0]),\
                                                torch.sin(x[:,M-1:N*M:M,0])*torch.cos(y[:,M-1:N*M:M,0])*torch.exp(-2*t[:,M-1:N*M:M,0])
        #p0[:,0:M,0],p0[:,(N-1)*M:N*M,0]=-0.25*(torch.cos(2*x[:,0:M,0])+torch.cos(2*y[:,0:M,0]))*torch.exp(-4*t[:,0:M,0]),\
                                        #-0.25*(torch.cos(2*x[:,(N-1)*M:N*M,0])+torch.cos(2*y[:,(N-1)*M:N*M,0]))*torch.exp(-4*t[:,(N-1)*M:N*M,0])
        #p0[:,0:(N-1)*M+1:M,0],p0[:,M-1:N*M:M,0]=-0.25*(torch.cos(2*x[:,0:(N-1)*M+1:M,0])+torch.cos(2*y[:,0:(N-1)*M+1:M,0]))*torch.exp(-4*t[:,0:(N-1)*M+1:M,0]),\
                                                #-0.25*(torch.cos(2*x[:,M-1:N*M:M,0])+torch.cos(2*y[:,M-1:N*M:M,0]))*torch.exp(-4*t[:,M-1:N*M:M,0])
    #step2
    dUVadxy=dUdx+dVdy+(dUadx+dVady)/2
    dUVcdxy=dUdx+dVdy+(dUcdx+dVcdy)/2
    for i in range(10):
        optP.zero_grad()
        lsum=0
        for k in range(K):
            #with autocast():
            l=P_loss(Pmodel,x[k,...],y[k,...],t[k,...],dt,pb[k,...],p0[k,...],dUVadxy[k,...],dUVcdxy[k,...],ddpdd[k,...])
            with torch.no_grad():
                lsum=lsum+l
            l.backward()
        lsum=lsum/K
        optP.step()
    #for i in range(80):
    #    optP.zero_grad()
    #    l=P_loss(Pmodel,x,y,t,dt,pb,p0,dUVadxy,dUVcdxy,ddpdd)
    #    l.backward()
    #    optP.step()

    dpdxa,dpdya,dpdxc,dpdyc=torch.zeros(x.shape,device=device),torch.zeros(x.shape,device=device),torch.zeros(x.shape,device=device),torch.zeros(x.shape,device=device)
    #for k in range(K):
    #    for m in range(M):
    #        dpdxa[k,m*N:(m+1)*N],dpdya[k,m*N:(m+1)*N],dpdxc[k,m*N:(m+1)*N],dpdyc[k,m*N:(m+1)*N]=dP_loss(Pmodel,x[k,m*N:(m+1)*N],y[k,m*N:(m+1)*N],t[k,m*N:(m+1)*N],dt)
    for k in range(K):
        dpdxa[k],dpdya[k],dpdxc[k],dpdyc[k]=dP_loss(Pmodel,x[k],y[k],t[k],dt)
    dpdxa=(dpdx+dpdxa)/2
    dpdya=(dpdy+dpdya)/2
    dpdxc=(dpdx+dpdxc)/2
    dpdyc=(dpdy+dpdyc)/2

    #step3
    #for i in range(80):
    #    optU.zero_grad()
    #    lsum=0
    #    for k in range(K):
    #        l=U_loss(Umodel,x[k,...],y[k,...],t[k,...],dt,ub[k,...],u0[k,...],pb[k,...],dUa[k,...],dUc[k,...],dpdxa[k,...],dpdxc[k,...],ukddpddx[k,...])
    #        with torch.no_grad():
    #            lsum=lsum+l
    #        l.backward()
    #    lsum=lsum/K
    #    optU.step()
    for i in range(40):
        optU.zero_grad()
        l=U_loss(Umodel,x,y,t,dt,ub,u0,pb,dUa,dUc,dpdxa,dpdxc,ukddpddx)
        l.backward()
        optU.step()

    #for i in range(80):
    #    optV.zero_grad()
    #    lsum=0
    #    for k in range(K):
    #        l=V_loss(Vmodel,x[k,...],y[k,...],t[k,...],dt,vb[k,...],v0[k,...],pb[k,...],dVa[k,...],dVc[k,...],dpdya[k,...],dpdyc[k,...],ukddpddy[k,...])
    #        with torch.no_grad():
    #            lsum=lsum+l
    #        l.backward()
    #    lsum=lsum/K
    #    optV.step()
    for i in range(40):
        optV.zero_grad()
        l=V_loss(Vmodel,x,y,t,dt,vb,v0,pb,dVa,dVc,dpdya,dpdyc,ukddpddy)
        l.backward()
        optV.step()

    if step==-1:
        break


torch.save(Umodel.state_dict(), './Umodel.pth')
torch.save(Vmodel.state_dict(), './Vmodel.pth')
torch.save(Pmodel.state_dict(), './Pmodel.pth')

