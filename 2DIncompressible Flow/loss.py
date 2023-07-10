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

def dUV_loss(Umodel,Vmodel,Pmodel,x,y,t,dt):

    ub=Umodel(x,y,t)
    vb=Vmodel(x,y,t)
    pb=Pmodel(x,y,t)

    ##各阶偏导
    dpdx,dpdy,dudx,dudy,dvdx,dvdy=gradients(pb,x),gradients(pb,y),gradients(ub,x),gradients(ub,y),gradients(vb,x),gradients(vb,y)
    ddpddxx,ddpddxy,ddpddyy = gradients(dpdx,x),gradients(dpdx,y),gradients(dpdy,y)
    dduddxx,dduddxy,dduddyy = gradients(dudx,x),gradients(dudx,y),gradients(dudy,y)
    ddvddxx,ddvddxy,ddvddyy = gradients(dvdx,x),gradients(dvdx,y),gradients(dvdy,y)
    

    duUdx,dvUdy = 2*ub*dudx,dvdy*ub+vb*dudy
    duVdx,dvVdy = dvdx*ub+vb*dudx,2*vb*dvdy

    #u*d(duUdx+dvUdy)/dx v*d(duUdx+dvUdy)/dy
    udduUddxx=ub*(2*dudx*dudx+2*ub*dduddxx)
    uddvUddyx=ub*(ddvddxy*ub+dvdy*dudx+dvdx*dudy+vb*dduddxy)
    vdduUddxy=vb*(2*dudy*dudx+2*ub*dduddxy)
    vddvUddyy=vb*(ddvddyy*ub+dvdy*dudy+dvdy*dudy+vb*dduddyy)
    
    duvUdxy=duUdx+dvUdy
    ukduvU=(udduUddxx+uddvUddyx+vdduUddxy+vddvUddyy)
    dduddxxyy=dduddxx+dduddyy
      
    with torch.no_grad():
        dUa=dt*(-duvUdxy+dt/2*ukduvU+dduddxxyy)
        dUc=-dt*(-duvUdxy-dt/2*ukduvU+dduddxxyy)
        dduvUdxy,dukduvU,ddduddxxyy=gradients(duvUdxy,x),gradients(ukduvU,x),gradients(dduddxxyy,x)
        dUadx=dt*(-dduvUdxy+dt/2*dukduvU+ddduddxxyy)
        dUcdx=-dt*(-dduvUdxy-dt/2*dukduvU+ddduddxxyy)

    #u*d(duVdx+dvVdy)/dx v*d(duVdx+dvVdy)/dy
    uddvVddyx=ub*(2*dvdx*dvdy+2*vb*ddvddxy)
    udduVddxx=ub*(ddvddxx*ub+dvdx*dudx+dvdx*dudx+vb*dduddxx)
    vddvVddyy=vb*(2*dvdy*dvdy+2*vb*ddvddyy)
    vdduVddxy=vb*(ddvddxy*ub+dvdx*dudy+dvdy*dudx+vb*dduddxy)

    duvVdxy=duVdx+dvVdy
    ukduvV=(uddvVddyx+udduVddxx+vddvVddyy+vdduVddxy)
    ddvddxxyy=ddvddxx+ddvddyy
    with torch.no_grad():
        dVa=dt*(-duvVdxy+dt/2*ukduvV+ddvddxxyy)
        dVc=-dt*(-duvVdxy-dt/2*ukduvV+ddvddxxyy)
        dduvVdxy,dukduvV,dddvddxxyy=gradients(duvVdxy,y),gradients(ukduvV,y),gradients(ddvddxxyy,y)
        dVady=dt*(-dduvVdxy+dt/2*dukduvV+dddvddxxyy)
        dVcdy=-dt*(-dduvVdxy-dt/2*dukduvV+dddvddxxyy)
    
    with torch.no_grad():
        dUdx=dudx
        dVdy=dvdy

    with torch.no_grad():
        ddpdd=ddpddxx+ddpddyy
        ukddpddx=ub*ddpddxx+vb*ddpddxy
        ukddpddy=ub*ddpddxy+vb*ddpddyy
    return ub.detach(),vb.detach(),pb.detach(),dUdx.detach(),dVdy.detach(),\
           dUa.detach(),dVa.detach(),dUadx.detach(),dVady.detach(),\
           dUc.detach(),dVc.detach(),dUcdx.detach(),dVcdy.detach(),\
           ddpdd.detach(),dpdx.detach(),dpdy.detach(),ukddpddx.detach(),ukddpddy.detach()


def P_loss(Pmodel,x,y,t,dt,p1,p0,dUVadxy,dUVcdxy,ddpdd):

    pc=Pmodel(x,y,t-dt)
    pb=Pmodel(x,y,t)
    pa=Pmodel(x,y,t+dt)

    ddpdda=(ddpdd + gradients(pa,x,2)+gradients(pa,y,2))/2
    ddpddc=(ddpdd + gradients(pc,x,2)+gradients(pc,y,2))/2
    dpa=-dt*(dUVadxy-dt/2*ddpdda)
    dpc=+dt*(dUVcdxy+dt/2*ddpddc)

    l1=loss1(pb,p0)/(dt**2)*0.1
    #l2=loss2(torch.zeros(p0.shape).to(p0.device),dpa)/(dt**2)
    l2=loss2(dUVadxy/dt*2,ddpdda)*(dt**2/4)
    #l3=loss2(dUVcdxy/dt*2,-ddpddc)*(dt**2/4)
    #l3=loss3(p0,dpc+p0)/(dt**2)
    return l1,l2

def dP_loss(Pmodel,x,y,t,dt):
    pa=Pmodel(x,y,t+dt)
    pc=Pmodel(x,y,t-dt)
    dPdxa,dPdya=gradients(pa,x,1),gradients(pa,y,1)
    dPdxc,dPdyc=gradients(pc,x,1),gradients(pc,y,1)
    return dPdxa.detach(),dPdya.detach(),dPdxc.detach(),dPdyc.detach()


def U_loss(Umodel,x,y,t,dt,u1,u0,p1,dUa,dUc,dPdxa,dPdxc,ukddPddx):
    
    uc=Umodel(x,y,t-dt)
    ub=Umodel(x,y,t)
    ua=Umodel(x,y,t+dt)

    dUa=dUa-dt*(dPdxa-dt/2*ukddPddx)
    dUc=dUc+dt*(dPdxc+dt/2*ukddPddx)
    l1=loss1(ub,u0)/(dt**2)*(1e+2)*0.3
    l2=loss2(ua,dUa+u0)/(dt**2)*(1e+2)
    #l3=loss3(uc,dUc+u0)/(dt**2)*(1e+2)
    return l1,l2

def V_loss(Vmodel,x,y,t,dt,v1,v0,p1,dVa,dVc,dPdya,dPdyc,ukddPddy):

    vc=Vmodel(x,y,t-dt)
    vb=Vmodel(x,y,t)
    va=Vmodel(x,y,t+dt)

    dVa=dVa-dt*(dPdya-dt/2*ukddPddy)
    dVc=dVc+dt*(dPdyc+dt/2*ukddPddy)
    l1=loss1(vb,v0)/(dt**2)*(1e+2)*0.3
    l2=loss2(va,dVa+v0)/(dt**2)*(1e+2)
    #l3=loss3(vc,dVc+v0)/(dt**2)*(1e+2)
    return l1,l2
