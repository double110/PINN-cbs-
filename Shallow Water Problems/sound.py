import os
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
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
    'a':0.1,
    'g':1.0,
    'apha':1/30,
    'X':40,
    'Y':5,
    'dx':0.25,
    'dy':0.25,
    'T':10,
    'dt':0.125,
    }

#模型
Umodel = UDate().to(device)
Vmodel = VDate().to(device)
Pmodel = PDate().to(device)

#输入
options['n']=int(options['X']//options['dx']+1)
options['m']=int(options['Y']//options['dy']+1)
options['k']=int(options['T']//options['dt']+1)
options['xy']=rand_xy(options['dx'],options['dy'],options['X'],options['Y']).to(device)
options['t']=rand_t(options['dt'],options['T']).repeat(1,options['n']*options['m'],1).to(device)
options['xyt']=torch.cat([options['xy'].repeat(options['k'],1,1),options['t']],dim=-1)
options['x'],options['y'],options['t']=options['xyt'].split(1,dim=-1)
options['tdt']=options['t']+options['dt']
options['H']=meandepth(options['x'],options['y'],options['t'])
#预处理
if options['preset']==0 :

    Umodel.load_state_dict(torch.load('./preUmodel.pth'))
    Vmodel.load_state_dict(torch.load('./preVmodel.pth'))
    Pmodel.load_state_dict(torch.load('./prePmodel.pth'))

    pre(1000,Umodel,Vmodel,Pmodel,options)
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
    {'params': Pmodel.parameters(),'lr':0.0000001},
    ],)

#scheduler = optim.lr_scheduler.StepLR(opt, step_size=30000, gamma=0.1)
lsum=0
a=options['a']
aph=options['apha']
g=options['g']
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
#约束条件 0时刻浪高和速度
with torch.no_grad():
    h=a*sech(1/2*((3*a)**0.5)*(x[0,...]-1/aph + 5))**2
    u=-(H[0,...]+1/2*a)*h/(h+x[0,...]*aph)
    h=h+H[0,...]
for step in range(1000):
    #step1 
    #为了节省内存 分段计算
    ub,vb,hb,dUdx,dVdy,dUa,dVa,dUadx,dVady,dUc,dVc,dUcdx,dVcdy,ddPdd,dPdx,dPdy,ukddPddx,ukddPddy=torch.zeros(x.shape,device=device),torch.zeros(x.shape,device=device),\
    torch.zeros(x.shape,device=device), torch.zeros(x.shape,device=device),torch.zeros(x.shape,device=device),torch.zeros(x.shape,device=device),\
    torch.zeros(x.shape,device=device),torch.zeros(x.shape,device=device) ,torch.zeros(x.shape,device=device),torch.zeros(x.shape,device=device),\
    torch.zeros(x.shape,device=device),torch.zeros(x.shape,device=device) ,torch.zeros(x.shape,device=device),torch.zeros(x.shape,device=device),\
    torch.zeros(x.shape,device=device) ,torch.zeros(x.shape,device=device),torch.zeros(x.shape,device=device),torch.zeros(x.shape,device=device),
    for k in range(K):
        for j in range(M):
            ub[k,j*N:(j+1)*N],vb[k,j*N:(j+1)*N],hb[k,j*N:(j+1)*N],dUdx[k,j*N:(j+1)*N],dVdy[k,j*N:(j+1)*N],\
            dUa[k,j*N:(j+1)*N],dVa[k,j*N:(j+1)*N],dUadx[k,j*N:(j+1)*N],dVady[k,j*N:(j+1)*N],\
            dUc[k,j*N:(j+1)*N],dVc[k,j*N:(j+1)*N],dUcdx[k,j*N:(j+1)*N],dVcdy[k,j*N:(j+1)*N],\
            ddPdd[k,j*N:(j+1)*N],dPdx[k,j*N:(j+1)*N],dPdy[k,j*N:(j+1)*N],ukddPddx[k,j*N:(j+1)*N],ukddPddy[k,j*N:(j+1)*N]\
            =dUV_loss(Umodel,Vmodel,Pmodel,x[k,j*N:(j+1)*N],y[k,j*N:(j+1)*N],t[k,j*N:(j+1)*N],dt)
    #浪高与速度约束
    with torch.no_grad():
        h0,u0,v0=torch.clone(hb),torch.clone(ub),torch.clone(vb)
        h0[40,...]=h
        u0[40,...]=u
        v0[40,...]=0
        u0[:,0:M,0],u0[:,(N-1)*M:N*M,0]=0,0
        #ub[:,0:M,0],ub[20:K,(N-1)*M:N*M,0]=0,0
        v0[:,0:(N-1)*M+1:M,0],v0[:,M-1:N*M:M,0]=0,0

    #step2
    dUVadxy=dUdx+dVdy+(dUadx+dVady)/2
    dUVcdxy=dUdx+dVdy+(dUcdx+dVcdy)/2
    for i in range(40):
        optP.zero_grad()
        lsum=0
        for k in range(K):
            l=P_loss(Pmodel,x[k,...],y[k,...],t[k,...],dt,hb[k,...],h0[k,...],dUVadxy[k,...],dUVcdxy[k,...],ddPdd[k,...])
            with torch.no_grad():
                lsum=lsum+l
            l.backward()
        lsum=lsum/K
        optP.step()

    dPdxa,dPdya,dPdxc,dPdyc=torch.zeros(x.shape,device=device),torch.zeros(x.shape,device=device),torch.zeros(x.shape,device=device),torch.zeros(x.shape,device=device)
    for k in range(K):
        for m in range(M):
            dPdxa[k,m*N:(m+1)*N],dPdya[k,m*N:(m+1)*N],dPdxc[k,m*N:(m+1)*N],dPdyc[k,m*N:(m+1)*N]=dP_loss(Pmodel,x[k,m*N:(m+1)*N],y[k,m*N:(m+1)*N],t[k,m*N:(m+1)*N],dt)
    dPdxa=(dPdx+dPdxa)/2
    dPdya=(dPdy+dPdya)/2
    dPdxc=(dPdx+dPdxc)/2
    dPdyc=(dPdy+dPdyc)/2

    #step3
    for i in range(40):
        optU.zero_grad()
        lsum=0
        for k in range(K):
            l=U_loss(Umodel,x[k,...],y[k,...],t[k,...],dt,ub[k,...],u0[k,...],hb[k,...],dUa[k,...],dUc[k,...],dPdxa[k,...],dPdxc[k,...],ukddPddx[k,...])
            with torch.no_grad():
                lsum=lsum+l
            l.backward()
        lsum=lsum/K
        optU.step()

    for i in range(40):
        optV.zero_grad()
        lsum=0
        for k in range(K):
            l=V_loss(Vmodel,x[k,...],y[k,...],t[k,...],dt,vb[k,...],v0[k,...],hb[k,...],dVa[k,...],dVc[k,...],dPdya[k,...],dPdyc[k,...],ukddPddy[k,...])
            with torch.no_grad():
                lsum=lsum+l
            l.backward()
        lsum=lsum/K
        optV.step()

    if step==-1:
        break


torch.save(Umodel.state_dict(), './Umodel.pth')
torch.save(Vmodel.state_dict(), './Vmodel.pth')
torch.save(Pmodel.state_dict(), './Pmodel.pth')


