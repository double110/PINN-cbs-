import os
import torch
import math
import tool
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.cuda.amp import autocast as autocast # 注意！！！
#import SPnet
from SPnet import UDate
from SPnet import VDate
from SPnet import WDate
from SPnet import PDate
from tool import rand_xyz
from tool import rand_t
from tool import sech
from tool import meandepth
from tool import exactU
from tool import exactV
from tool import exactW
from tool import exactP
from tool import exactBoun
#from tool import _initAD
from Pretreat import pre
from loss import dUVW_loss
from loss import P_loss
from loss import dP_loss
from loss import U_loss
from loss import V_loss
from loss import W_loss

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
    'X':2,
    'Y':2,
    'Z':2,
    'dx':2/10,
    'dy':2/10,
    'dz':2/10,
    'T':0.1,
    'dt':0.01,
    }

A,D=tool.A,tool.D

#模型
Umodel = UDate().to(device)
Vmodel = VDate().to(device)
Wmodel = WDate().to(device)
Pmodel = PDate().to(device)

#输入
options['n']=round(options['X']/options['dx'])+1
options['m']=round(options['Y']/options['dy'])+1
options['l']=round(options['Z']/options['dz'])+1
options['k']=round(options['T']/options['dt'])+1
options['xyz']=rand_xyz(options).unsqueeze_(dim=0).repeat(options['k'],1,1).to(device)
options['t']=rand_t(options['dt'],options['T']).repeat(1,options['n']*options['m']*options['l'],1).to(device)
options['x'],options['y'],options['z']=options['xyz'].split(1,dim=-1)
#预处理
if options['preset']==0 :

    #Umodel.load_state_dict(torch.load('./preUmodel.pth'))
    #Vmodel.load_state_dict(torch.load('./preVmodel.pth'))
    #Wmodel.load_state_dict(torch.load('./preWmodel.pth'))
    #Pmodel.load_state_dict(torch.load('./prePmodel.pth'))
    #Umodel.load_state_dict(torch.load('./Umodel.pth'))
    #Vmodel.load_state_dict(torch.load('./Vmodel.pth'))
    #Pmodel.load_state_dict(torch.load('./Pmodel.pth'))

    pre(20000,Umodel,Vmodel,Wmodel,Pmodel,options)
    torch.save(Umodel.state_dict(), './preUmodel.pth')
    torch.save(Vmodel.state_dict(), './preVmodel.pth')
    torch.save(Wmodel.state_dict(), './preWmodel.pth')
    torch.save(Pmodel.state_dict(), './prePmodel.pth')

    #保存参数
else:
    Umodel.load_state_dict(torch.load('./preUmodel.pth'))
    Vmodel.load_state_dict(torch.load('./preVmodel.pth'))
    Wmodel.load_state_dict(torch.load('./preWmodel.pth'))
    Pmodel.load_state_dict(torch.load('./prePmodel.pth'))

#训练
if options['loadmodel']!=0 :
    Umodel.load_state_dict(torch.load('./Umodel.pth'))
    Vmodel.load_state_dict(torch.load('./Vmodel.pth'))
    Wmodel.load_state_dict(torch.load('./Wmodel.pth'))
    Pmodel.load_state_dict(torch.load('./Pmodel.pth'))

optU = optim.Adam([
    {'params': Umodel.parameters(),'lr':0.0000001},
    ],)
optV = optim.Adam([
    {'params': Vmodel.parameters(),'lr':0.0000001},
    ],)
optW = optim.Adam([
    {'params': Wmodel.parameters(),'lr':0.0000001},
    ],)
optP = optim.Adam([
    {'params': Pmodel.parameters(),'lr':0.000001},
    ],)

#scheduler = optim.lr_scheduler.StepLR(opt, step_size=30000, gamma=0.1)
lsum=0
dt =options['dt']
x=options['x'].detach()
x.requires_grad=True
y=options['y'].detach()
y.requires_grad=True
z=options['z'].detach()
z.requires_grad=True
t=options['t'].detach()
K=options['k']
N=options['n']
M=options['m']
L=options['l']
#约束条件 0时刻浪高和速度
with torch.no_grad():
    p=-A**2/2*(torch.exp(2*A*x[0])+torch.exp(2*A*y[0])+torch.exp(2*A*z[0])
                     +2*torch.sin(A*x[0]+D*y[0])*torch.cos(A*z[0]+D*x[0])*torch.exp(A*(y[0]+z[0]))
                     +2*torch.sin(A*y[0]+D*z[0])*torch.cos(A*x[0]+D*y[0])*torch.exp(A*(z[0]+x[0]))
                     +2*torch.sin(A*z[0]+D*x[0])*torch.cos(A*y[0]+D*z[0])*torch.exp(A*(x[0]+y[0]))
                     )
    u=-A*(torch.exp(A*x[0])*torch.sin(A*y[0]+D*z[0])+torch.exp(A*z[0])*torch.cos(A*x[0]+D*y[0]))
    v=-A*(torch.exp(A*y[0])*torch.sin(A*z[0]+D*x[0])+torch.exp(A*x[0])*torch.cos(A*y[0]+D*z[0]))
    w=-A*(torch.exp(A*z[0])*torch.sin(A*x[0]+D*y[0])+torch.exp(A*y[0])*torch.cos(A*z[0]+D*x[0]))
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
    ub,vb,wb,pb,dUdx,dVdy,dWdz,dUa,dVa,dWa,dUadx,dVady,dWadz,dUc,dVc,dWc,dUcdx,dVcdy,dWcdz,ddpdd,dpdx,dpdy,dpdz,ukddpddx,ukddpddy,ukddpddz=\
    torch.zeros(x.shape,device=device),torch.zeros(x.shape,device=device),\
    torch.zeros(x.shape,device=device), torch.zeros(x.shape,device=device),torch.zeros(x.shape,device=device),torch.zeros(x.shape,device=device),\
    torch.zeros(x.shape,device=device),torch.zeros(x.shape,device=device) ,torch.zeros(x.shape,device=device),torch.zeros(x.shape,device=device),\
    torch.zeros(x.shape,device=device),torch.zeros(x.shape,device=device) ,torch.zeros(x.shape,device=device),torch.zeros(x.shape,device=device),\
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
            ub[k],vb[k],wb[k],pb[k],dUdx[k],dVdy[k],dWdz[k],\
            dUa[k],dVa[k],dWa[k],dUadx[k],dVady[k],dWadz[k],\
            dUc[k],dVc[k],dWc[k],dUcdx[k],dVcdy[k],dWcdz[k],\
            ddpdd[k],dpdx[k],dpdy[k],dpdz[k],ukddpddx[k],ukddpddy[k],ukddpddz[k]\
            =dUVW_loss(Umodel,Vmodel,Wmodel,Pmodel,x[k],y[k],z[k],t[k],dt)
    #浪高与速度约束
    with torch.no_grad():
        p0,u0,v0,w0=torch.clone(pb),torch.clone(ub),torch.clone(vb),torch.clone(wb)
        p0[0,...]=p
        u0[0,...]=u
        v0[0,...]=v
        w0[0,...]=w
        for k in range(L):
            for m in range (M): #yz
                bx1,bx2=k*N*M+m,k*N*M+m+(N-1)*M
                u0[:,bx1,0],v0[:,bx1,0],w0[:,bx1,0],p0[:,bx1,0]=exactBoun(x[:,bx1,0],y[:,bx1,0],z[:,bx1,0],t[:,bx1,0])
                u0[:,bx2,0],v0[:,bx2,0],w0[:,bx2,0],p0[:,bx2,0]=exactBoun(x[:,bx2,0],y[:,bx2,0],z[:,bx2,0],t[:,bx2,0])
            for n in range (N):#xz
                bx1,bx2=k*N*M+n*M,k*N*M+(M-1)+n*M
                u0[:,bx1,0],v0[:,bx1,0],w0[:,bx1,0],p0[:,bx1,0]=exactBoun(x[:,bx1,0],y[:,bx1,0],z[:,bx1,0],t[:,bx1,0])
                u0[:,bx2,0],v0[:,bx2,0],w0[:,bx2,0],p0[:,bx2,0]=exactBoun(x[:,bx2,0],y[:,bx2,0],z[:,bx2,0],t[:,bx2,0])
        for ud in range(N*M):#yz
             bx1,bx2=ud,ud+N*M*(L-1)
             u0[:,bx1,0],v0[:,bx1,0],w0[:,bx1,0],p0[:,bx1,0]=exactBoun(x[:,bx1,0],y[:,bx1,0],z[:,bx1,0],t[:,bx1,0])
             u0[:,bx2,0],v0[:,bx2,0],w0[:,bx2,0],p0[:,bx2,0]=exactBoun(x[:,bx2,0],y[:,bx2,0],z[:,bx2,0],t[:,bx2,0])



    #step2
    dUVWadxyz=dUdx+dVdy+dWdz+(dUadx+dVady+dWadz)/2
    dUVWcdxyz=dUdx+dVdy+dWdz+(dUcdx+dVcdy+dWcdz)/2
    for i in range(10):
        optP.zero_grad()
        lsum=0
        for k in range(K):
            #with autocast():
            l=P_loss(Pmodel,x[k,...],y[k,...],z[k,...],t[k,...],dt,pb[k,...],p0[k,...],dUVWadxyz[k,...],dUVWcdxyz[k,...],ddpdd[k,...])
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

    dpdxa,dpdya,dpdza,dpdxc,dpdyc,dpdzc=\
    torch.zeros(x.shape,device=device),torch.zeros(x.shape,device=device),torch.zeros(x.shape,device=device),\
    torch.zeros(x.shape,device=device),torch.zeros(x.shape,device=device),torch.zeros(x.shape,device=device)
    #for k in range(K):
    #    for m in range(M):
    #        dpdxa[k,m*N:(m+1)*N],dpdya[k,m*N:(m+1)*N],dpdxc[k,m*N:(m+1)*N],dpdyc[k,m*N:(m+1)*N]=dP_loss(Pmodel,x[k,m*N:(m+1)*N],y[k,m*N:(m+1)*N],t[k,m*N:(m+1)*N],dt)
    for k in range(K):
        dpdxa[k],dpdya[k],dpdza[k],dpdxc[k],dpdyc[k],dpdzc[k]=dP_loss(Pmodel,x[k],y[k],z[k],t[k],dt)
    dpdxa=(dpdx+dpdxa)/2
    dpdya=(dpdy+dpdya)/2
    dpdza=(dpdz+dpdza)/2
    dpdxc=(dpdx+dpdxc)/2
    dpdyc=(dpdy+dpdyc)/2
    dpdzc=(dpdz+dpdzc)/2

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
        lu=U_loss(Umodel,x,y,z,t,dt,ub,u0,pb,dUa,dUc,dpdxa,dpdxc,ukddpddx)
        lu.backward()
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
        lv=V_loss(Vmodel,x,y,z,t,dt,vb,v0,pb,dVa,dVc,dpdya,dpdyc,ukddpddy)
        lv.backward()
        optV.step()
    for i in range(40):
        optW.zero_grad()
        lw=W_loss(Wmodel,x,y,z,t,dt,wb,w0,pb,dWa,dWc,dpdza,dpdzc,ukddpddz)
        lw.backward()
        optW.step()

    if (step%1)==0:
        print(f"step: {step} ")
        print(f"lp: {lsum} ")
        print(f"lu: {lu} ")
        print(f"lv: {lv} ")
        print(f"lw: {lw} \n")
    if step==-1:
        break


torch.save(Umodel.state_dict(), './Umodel.pth')
torch.save(Vmodel.state_dict(), './Vmodel.pth')
torch.save(Wmodel.state_dict(), './Wmodel.pth')
torch.save(Pmodel.state_dict(), './Pmodel.pth')

sound=Sound(model,model2)
