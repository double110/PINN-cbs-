import torch  
import math

# 定义区域及其上的采样  
def rand_xy(dx,dy,x,y):
    #根据需要的采样间隔确定采样数
    n,m=round(x/dx)+1,round(y/dy)+1
    #x为传播方向
    xy= torch.zeros(n*m,2)
    for i in range(n):
        for j in range(m):
            xy[i*m+j,0]=i*dx
            xy[i*m+j,1]=j*dy
    return xy

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

