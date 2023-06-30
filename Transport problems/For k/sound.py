import os
import torch
import math 
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
#import SPnet
from SPnet import UDate
from SPnet import VDate
from SPnet import PDate
from SPnet import TDate
from tool import rand_xy
from tool import rand_t
from tool import sech
from tool import meandepth
from Pretreat import pre
from loss import predUVh
from loss import dUVh_G
from loss import T_loss

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
    'kt':0.03767613,
    'X':5,
    'Y':5,
    'dx':0.2,
    'dy':0.2,
    'T':10,
    'dt':0.125,
    }

#模型
Umodel = UDate().to(device)
Vmodel = VDate().to(device)
Pmodel = PDate().to(device)
Tmodel = TDate().to(device)
Umodel.load_state_dict(torch.load('./Umodel.pth'))
Vmodel.load_state_dict(torch.load('./Vmodel.pth'))
Pmodel.load_state_dict(torch.load('./Pmodel.pth'))
Tmodelref = TDate().to(device)
Tmodelref.load_state_dict(torch.load('./Tmodelref.pth'))

#输入
options['n']=int(options['X']/options['dx']+1)
options['m']=int(options['Y']/options['dy']+1)
options['k']=int(options['T']/options['dt']+1)
options['xy']=rand_xy(options['dx'],options['dy'],options['X'],options['Y']).to(device)
options['t']=rand_t(options['dt'],options['T']).repeat(1,options['n']*options['m'],1).to(device)
options['xyt']=torch.cat([options['xy'].repeat(options['k'],1,1),options['t']],dim=-1)
options['x'],options['y'],options['t']=options['xyt'].split(1,dim=-1)
options['tdt']=options['t']+options['dt']
options['H']=meandepth(options['x'],options['y'],options['t'])


#预处理
if options['preset']==0 :
    Tmodel.load_state_dict(torch.load('./preTmodel.pth'))

    pre(1000,Tmodel,options)
    torch.save(Tmodel.state_dict(), './preTmodel.pth')
    
    #保存参数
else:
    Tmodel.load_state_dict(torch.load('./preTmodel.pth'))

#训练
if options['loadmodel']!=0 :
    Tmodel.load_state_dict(torch.load('./Tmodel.pth'))

optT = optim.Adam([
    {'params': Tmodel.parameters(),'lr':0.0000001},
    ],)

#设置kt为可学习变量
kt= torch.nn.Parameter(torch.tensor([options['kt']],device=device), requires_grad=True)
optkt = optim.Adam([
    {'params': kt,'lr':0.0001},
    ],)

#scheduler = optim.lr_scheduler.StepLR(opt, step_size=30000, gamma=0.1)
step=0
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
H=options['H']
#约束条件 0时刻T，tf时刻T
with torch.no_grad():
    d2=(x[0,...]-22.5)**2+(y[0,...]-2.5)**2
    Tini=Tmodelref(x[0,...],y[0,...],t[0,...])
    Ttf=Tmodelref(x[K-1,...],y[K-1,...],t[K-1,...])

u,v,h,ha,hc=torch.zeros(x.shape,device=device).repeat(1,1,5).split(1,dim=-1)
dUVh=torch.zeros(x.shape,device=device).repeat(1,1,15)
for k in range(K):
    dUVh[k],u[k],v[k],h[k],ha[k],hc[k] =predUVh(Umodel,Vmodel,Pmodel,x[k],y[k],t[k],dt)

for step in range(1000):
    #step1 
    optkt.zero_grad()
    #为了节省内存 分段计算
    Tb,dTa,dTc,ddhTdd,dhdx,dhdy=torch.zeros(x.shape,device=device),torch.zeros(x.shape,device=device),torch.zeros(x.shape,device=device),\
                                torch.zeros(x.shape,device=device),torch.zeros(x.shape,device=device),torch.zeros(x.shape,device=device)
    #for k in range(K):
    #    for j in range(M):
    #        Tb[k,j*N:(j+1)*N],h[k,j*N:(j+1)*N],dTa[k,j*N:(j+1)*N],dTc[k,j*N:(j+1)*N],ddhTdd[k,j*N:(j+1)*N],\
    #        dhdx[k,j*N:(j+1)*N],dhdy[k,j*N:(j+1)*N]\
    #        =dUVh_G(Umodel,Vmodel,Pmodel,Tmodel,x[k,j*N:(j+1)*N],y[k,j*N:(j+1)*N],t[k,j*N:(j+1)*N],dt)
    for k in range(K):
        Tb[k],h[k],dTa[k],dTc[k],ddhTdd[k],\
            dhdx[k],dhdy[k]\
            =dUVh_G(dUVh[k],u[k],v[k],h[k],Tmodel,x[k],y[k],t[k],dt)
    
    #T约束
    with torch.no_grad():
        T0=torch.clone(Tb)
        T0[0,...]=Tini
        T0[K-1,...]=Ttf

    #step1
    for i in range(20):
        optT.zero_grad()
        lsum=0
        for k in range(K):
            l=T_loss(options,Tmodel,x[k,...],y[k,...],t[k,...],dt,kt,Tb[k,...],T0[k,...],dTa[k,...],dTc[k,...],h[k,...],ha[k,...],hc[k,...],ddhTdd[k,...],dhdx[k,...],dhdy[k,...])
            with torch.no_grad():
                lsum=lsum+l
            l.backward()
        lsum=lsum/K
        optT.step()

    #for i in range(40):
    #    optT.zero_grad()
    #    lsum=0
    #    l=T_loss(Tmodel,x,y,t,dt,kt,Tb,T0,dTa,dTc,h,ddhTdd,dhdx,dhdy)
    #    l.backward()
    #    optT.step()

    #求kt
    optkt.step()

    if step==-1:
        break
    if (step%1)==0:
        print(f"step: {step} ")
        print(f"kt: {kt} ")
        print(f"lsum: {lsum} ")
        print(f"")

torch.save(Tmodel.state_dict(), './Tmodel.pth')

sound=Sound(model,model2)
