import gc 
import torch  
import math
from torch import nn
import torch.optim as optim
from tool import gradients
from tool import sech
from tool import meandepth

loss1,loss2,loss3,loss4,loss5,loss6,loss7,loss8,loss9 = torch.nn.MSELoss(),torch.nn.MSELoss(),torch.nn.MSELoss(),torch.nn.MSELoss(),\
torch.nn.MSELoss(),torch.nn.MSELoss(),torch.nn.MSELoss(),torch.nn.MSELoss(),torch.nn.MSELoss()
def predUVh(Umodel,Vmodel,Pmodel,x,y,t,dt):
    u=Umodel(x,y,t)
    v=Vmodel(x,y,t)
    h=Pmodel(x,y,t)
    ha,hc=Pmodel(x,y,t+dt),Pmodel(x,y,t-dt)
    ##各阶偏导
    dhdx,dhdy,dudx,dudy,dvdx,dvdy=gradients(h,x),gradients(h,y),gradients(u,x),gradients(u,y),gradients(v,x),gradients(v,y)
    with torch.no_grad():
        ddhddxx,ddhddxy,ddhddyy = gradients(dhdx,x),gradients(dhdx,y),gradients(dhdy,y)
        dduddxx,dduddxy,dduddyy = gradients(dudx,x),gradients(dudx,y),gradients(dudy,y)
        ddvddxx,ddvddxy,ddvddyy = gradients(dvdx,x),gradients(dvdx,y),gradients(dvdy,y)
    dUVh=  torch.cat([dhdx.detach(),dhdy.detach(),dudx.detach(),dudy.detach(),dvdx.detach(),dvdy.detach(),\
           ddhddxx.detach(),ddhddxy.detach(),ddhddyy.detach(),\
           dduddxx.detach(),dduddxy.detach(),dduddyy.detach(),\
           ddvddxx.detach(),ddvddxy.detach(),ddvddyy.detach()],
           dim=-1          )
    return dUVh\
           ,u.detach(),v.detach(),h.detach(),ha.detach(),hc.detach()

def dUVh_G(dUVh,u,v,h,Tmodel,x,y,t,dt):

    T=Tmodel(x,y,t)
    ##各阶偏导
    dhdx,dhdy,dudx,dudy,dvdx,dvdy,\
    ddhddxx,ddhddxy,ddhddyy,\
    dduddxx,dduddxy,dduddyy,\
    ddvddxx,ddvddxy,ddvddyy = dUVh.split(1,dim=-1)

    dTdx,dTdy=gradients(T,x),gradients(T,y)
    ddTddxx,ddTddxy,ddTddyy=gradients(dTdx,x),gradients(dTdx,y),gradients(dTdy,y)
    
    with torch.no_grad():
    #d(huT)/dx
        dhuTdx,dhvTdy=h*dudx*T+dhdx*u*T+h*u*dTdx,h*dvdy*T+dhdy*v*T+h*v*dTdy

    #u*d(duTdx+dvTdy)/dx v*d(duTdx+dvTdy)/dy
        uddhuTddxx=u*(dhdx*dudx*T+h*dduddxx*T+h*dudx*dTdx+ddhddxx*u*T+dhdx*dudx*T+dhdx*u*dTdx+dhdx*u*dTdx+h*dudx*dTdx+h*u*ddTddxx)
        uddhvTddyx=u*(dhdx*dvdy*T+h*ddvddxy*T+h*dvdy*dTdx+ddhddxy*v*T+dhdy*dvdx*T+dhdy*v*dTdx+dhdx*v*dTdy+h*dvdx*dTdy+h*v*ddTddxy)

        vddhuTddxy=v*(dhdy*dudx*T+h*dduddxy*T+h*dudx*dTdy+ddhddxy*u*T+dhdx*dudy*T+dhdx*u*dTdy+dhdy*u*dTdx+h*dudy*dTdx+h*u*ddTddxy)
        vddhvTddyy=v*(dhdy*dvdy*T+h*ddvddyy*T+h*dvdy*dTdy+ddhddyy*v*T+dhdy*dvdy*T+dhdy*v*dTdy+dhdy*v*dTdy+h*dvdy*dTdy+h*v*ddTddyy)
    #
    
        dTa=-dt*(dhuTdx+dhvTdy-dt/2*(uddhuTddxx+uddhvTddyx+vddhuTddxy+vddhvTddyy))
        dTc=+dt*(dhuTdx+dhvTdy+dt/2*(uddhuTddxx+uddhvTddyx+vddhuTddxy+vddhvTddyy))
        ddhTdd=dhdx*dTdx+h*ddTddxx+dhdy*dTdy+h*ddTddyy

    return T.detach(),h.detach(),dTa.detach(),dTc.detach(),ddhTdd.detach(),dhdx.detach(),dhdy.detach()


def T_loss(options,Tmodel,x,y,t,dt,k,T1,T0,dTa,dTc,h,ha,hc,ddhTdd,dhdx,dhdy):
    N=options['n']
    M=options['m']

    Tb=Tmodel(x,y,t)
    Ta=Tmodel(x,y,t+dt)
    Tc=Tmodel(x,y,t-dt)
    #偏导
    dTadx,dTady=gradients(Ta,x),gradients(Ta,y)
    dTcdx,dTcdy=gradients(Tc,x),gradients(Tc,y)
    with torch.no_grad():
        ddTaddxx,ddTaddyy=gradients(dTadx,x),gradients(dTady,y)
        ddTcddxx,ddTcddyy=gradients(dTcdx,x),gradients(dTcdy,y)
    #ddTaddxx,ddTaddyy=gradients(dTadx,x),gradients(dTady,y)
    #ddTcddxx,ddTcddyy=gradients(dTcdx,x),gradients(dTcdy,y)
    ddhkTadd=(k*ddhTdd+k*(dhdx*dTadx+h*ddTaddxx+dhdy*dTady+h*ddTaddyy))/2
    ddhkTcdd=(k*ddhTdd+k*(dhdx*dTcdx+h*ddTcddxx+dhdy*dTcdy+h*ddTcddyy))/2


    dTa=dTa+dt*ddhkTadd
    dTc=dTc-dt*ddhkTcdd
    
    #边界设定
    kbou=torch.detach(k)
    xb=torch.cat([x[0:M,:],x[(N-1)*M:N*M,:],x[0:(N-1)*M+1:M,:],x[M-1:N*M:M,:]],dim=-2)
    yb=torch.cat([y[0:M,:],y[(N-1)*M:N*M,:],y[0:(N-1)*M+1:M,:],y[M-1:N*M:M,:]],dim=-2)
    tb=torch.cat([t[0:M,:],t[(N-1)*M:N*M,:],t[0:(N-1)*M+1:M,:],t[M-1:N*M:M,:]],dim=-2)
    hbou=torch.cat([h[0:M,:],h[(N-1)*M:N*M,:],h[0:(N-1)*M+1:M,:],h[M-1:N*M:M,:]],dim=-2)
    Tbou=Tmodel(xb,yb,tb)
    
    dTboudx,dTboudy=gradients(Tbou,xb),gradients(Tbou,yb)
    ddTbouddxx,ddTbouddyy=gradients(dTboudx,xb),gradients(dTboudy,yb)
    dhboudx=torch.cat([dhdx[0:M,:],dhdx[(N-1)*M:N*M,:],dhdx[0:(N-1)*M+1:M,:],dhdx[M-1:N*M:M,:]],dim=-2)
    dhboudy=torch.cat([dhdy[0:M,:],dhdy[(N-1)*M:N*M,:],dhdy[0:(N-1)*M+1:M,:],dhdy[M-1:N*M:M,:]],dim=-2)
    ddhkTboudd=kbou*(dhboudx*dTboudx+hbou*ddTbouddxx+dhboudy*dTboudy+hbou*ddTbouddyy)
    #ddhkTaboudd=torch.cat([ddhkTadd[0:M,:],ddhkTadd[(N-1)*M:N*M,:],ddhkTadd[0:(N-1)*M+1:M,:],ddhkTadd[M-1:N*M:M,:]],dim=-2)
    #l4=loss4(ddhkTaboudd,torch.zeros(ddhkTaboudd.shape).to(ddhkTaboudd.device))
    l4=loss4(ddhkTboudd,torch.zeros(ddhkTboudd.shape).to(ddhkTboudd.device))
    ####
    
    l1=loss1(Tb,T0)/(dt**2)*10
    l2=loss2(Ta,(dTa+h*T0)/ha)/(dt**2)
    l3=loss3(Tc,(dTc+h*T0)/hc)/(dt**2)
    return l1+l2+l3+l4
