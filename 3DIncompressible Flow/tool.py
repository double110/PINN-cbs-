import torch  
import math

A,D=math.pi/4,math.pi/2
def _initAD():  # 初始化
    global A,D
    A,D=math.pi/4,math.pi/2

# 定义区域及其上的采样  
def rand_xyz(op):
    #根据需要的采样间隔确定采样数
    x,y,z=op['X'],op['Y'],op['Z']
    dx,dy,dz=op['dx'],op['dy'],op['dz']
    n,m,l=op['n'],op['m'],op['l']
    #x为传播方向
    xyz= torch.zeros(n*m*l,3)
    for i in range(n):
        for j in range(m):
            for k in range(l):
                xyz[i*m+j+k*n*m,0]=i*dx-1
                xyz[i*m+j+k*n*m,1]=j*dy-1
                xyz[i*m+j+k*n*m,2]=k*dz-1
    return xyz

def rand_t(dt,T):
    k=int(T/dt+1)
    t=torch.zeros(k)
    for i in range(k):
        t[i]=i*dt
    t.unsqueeze_(dim=-1).unsqueeze_(dim=-1)
    return t

#定义梯度计算  
def gradients(u, x, order=1):  
    if order == 1:  
        return torch.autograd.grad(u, x, \
        grad_outputs=torch.ones_like(u),  \
        create_graph=True, )[0]
    else:  
        return gradients(gradients(u, x), x, \
        order=order - 1)  

#定义深度
def meandepth(x,y,t):
    H=x/40+0.5+0*y
    return H

#定义sech 2/(e**x+e**-x)
def sech(x):
    y=2/(torch.e**(x)+torch.e**(-x))
    return y

def exactU(x,y,z,t):
    U=-A*(torch.exp(A*x)*torch.sin(A*y+D*z)+torch.exp(A*z)*torch.cos(A*x+D*y))*torch.exp(-D**2*t)
    return U

def exactV(x,y,z,t):
    V=-A*(torch.exp(A*y)*torch.sin(A*z+D*x)+torch.exp(A*x)*torch.cos(A*y+D*z))*torch.exp(-D**2*t)
    return V

def exactW(x,y,z,t):
    W=-A*(torch.exp(A*z)*torch.sin(A*x+D*y)+torch.exp(A*y)*torch.cos(A*z+D*x))*torch.exp(-D**2*t)  
    return W

def exactP(x,y,z,t):
    P=-A**2/2*(torch.exp(2*A*x)+torch.exp(2*A*y)+torch.exp(2*A*z)
                     +2*torch.sin(A*x+D*y)*torch.cos(A*z+D*x)*torch.exp(A*(y+z))
                     +2*torch.sin(A*y+D*z)*torch.cos(A*x+D*y)*torch.exp(A*(z+x))
                     +2*torch.sin(A*z+D*x)*torch.cos(A*y+D*z)*torch.exp(A*(x+y))
                     )*torch.exp(-2*D**2*t)
    return P

def exactBoun(x,y,z,t):
    u=exactU(x,y,z,t)
    v=exactV(x,y,z,t)
    w=exactW(x,y,z,t)
    p=exactP(x,y,z,t)
    return u,v,w,p