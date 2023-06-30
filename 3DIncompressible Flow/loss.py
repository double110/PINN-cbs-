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

def dUVW_loss(Umodel,Vmodel,Wmodel,Pmodel,x,y,z,t,dt):

    ub=Umodel(x,y,z,t)
    vb=Vmodel(x,y,z,t)
    wb=Wmodel(x,y,z,t)
    pb=Pmodel(x,y,z,t)

    ##各阶偏导
    dpdx,dpdy,dpdz=gradients(pb,x),gradients(pb,y),gradients(pb,z)
    dudx,dudy,dudz=gradients(ub,x),gradients(ub,y),gradients(ub,z)
    dvdx,dvdy,dvdz=gradients(vb,x),gradients(vb,y),gradients(vb,z)
    dwdx,dwdy,dwdz=gradients(wb,x),gradients(wb,y),gradients(wb,z)

    ddpddxx,ddpddxy,ddpddyy,ddpddyz,ddpddzz,ddpddzx = gradients(dpdx,x),gradients(dpdx,y),gradients(dpdy,y),gradients(dpdy,z),gradients(dpdz,z),gradients(dpdz,x)
    dduddxx,dduddxy,dduddyy,dduddyz,dduddzz,dduddzx = gradients(dudx,x),gradients(dudx,y),gradients(dudy,y),gradients(dudy,z),gradients(dudz,z),gradients(dudz,x)
    ddvddxx,ddvddxy,ddvddyy,ddvddyz,ddvddzz,ddvddzx = gradients(dvdx,x),gradients(dvdx,y),gradients(dvdy,y),gradients(dvdy,z),gradients(dvdz,z),gradients(dvdz,x)
    ddwddxx,ddwddxy,ddwddyy,ddwddyz,ddwddzz,ddwddzx = gradients(dwdx,x),gradients(dwdx,y),gradients(dwdy,y),gradients(dwdy,z),gradients(dwdz,z),gradients(dwdz,x)
    
    
    duUdx,dvUdy,dwUdz = 2*ub*dudx,dvdy*ub+vb*dudy,dwdz*ub+wb*dudz
    duVdx,dvVdy,dwVdz = dvdx*ub+vb*dudx,2*vb*dvdy,dwdz*vb+wb*dvdz
    duWdx,dvWdy,dwWdz = dwdx*ub+wb*dudx,dvdy*wb+vb*dwdy,2*wb*dwdz

    ##aaaaaaa=ub*gradients(duUdx,x) ##测试用
    #u*d(duUdx+dvUdy+dwUdz)/dx v*d(duUdx+dvUdy+dwUdz)/dy w*d(duUdx+dvUdy+dwUdz)/dz
    udduUddxx=ub*(2*dudx*dudx+2*ub*dduddxx)
    uddvUddyx=ub*(ddvddxy*ub+dvdy*dudx+dvdx*dudy+vb*dduddxy)
    uddwUddzx=ub*(ddwddzx*ub+dwdz*dudx+dwdx*dudz+wb*dduddzx)

    vdduUddxy=vb*(2*dudy*dudx+2*ub*dduddxy)
    vddvUddyy=vb*(ddvddyy*ub+dvdy*dudy+dvdy*dudy+vb*dduddyy)
    vddwUddzy=vb*(ddwddyz*ub+dwdz*dudy+dwdy*dudz+wb*dduddyz)

    wdduUddxz=wb*(2*dudz*dudx+2*ub*dduddzx)
    wddvUddyz=wb*(ddvddyz*ub+dvdy*dudz+dvdz*dudy+vb*dduddyz)
    wddwUddzz=wb*(ddwddzz*ub+dwdz*dudz+dwdz*dudz+wb*dduddzz)
    
    duvwUdxyz=duUdx+dvUdy+dwUdz
    ukduvwU=(udduUddxx+uddvUddyx+uddwUddzx+vdduUddxy+vddvUddyy+vddwUddzy+wdduUddxz+wddvUddyz+wddwUddzz)
    ddUddxxyyzz=dduddxx+dduddyy+dduddzz
      
    with torch.no_grad():
        dUa=dt*(-duvwUdxyz+dt/2*ukduvwU+ddUddxxyyzz)
        dUc=-dt*(-duvwUdxyz-dt/2*ukduvwU+ddUddxxyyzz)
        dduvwUdxyz,dukduvwU,ddduddxxyyzz=gradients(duvwUdxyz,x),gradients(ukduvwU,x),gradients(ddUddxxyyzz,x)
        dUadx=dt*(-dduvwUdxyz+dt/2*dukduvwU+ddUddxxyyzz)
        dUcdx=-dt*(-dduvwUdxyz-dt/2*dukduvwU+ddUddxxyyzz)

    #u*d(duVdx+dvVdy+dwVdz)/dx v*d(duVdx+dvVdy+dwVdz)/dy w*d(duVdx+dvVdy+dwVdz)/dz
    udduVddxx=ub*(ddvddxx*ub+dvdx*dudx+dvdx*dudx+vb*dduddxx)
    uddvVddyx=ub*(2*dvdx*dvdy+2*vb*ddvddxy)
    uddwVddzx=ub*(ddwddzx*vb+dwdz*dvdx+dwdx*dvdz+wb*ddvddzx)

    vdduVddxy=vb*(ddvddxy*ub+dvdx*dudy+dvdy*dudx+vb*dduddxy)
    vddvVddyy=vb*(2*dvdy*dvdy+2*vb*ddvddyy)
    vddwVddzy=vb*(ddwddyz*vb+dwdz*dvdy+dwdy*dvdz+wb*ddvddyz)

    wdduVddxz=wb*(ddvddzx*ub+dvdx*dudz+dvdz*dudx+vb*dduddzx)
    wddvVddyz=wb*(2*dvdz*dvdy+2*vb*ddvddyz)
    wddwVddzz=wb*(ddwddzz*vb+dwdz*dvdz+dwdz*dvdz+wb*ddvddzz)
    
    duvwVdxyz=duVdx+dvVdy+dwVdz
    ukduvwV=(udduVddxx+uddvVddyx+uddwVddzx+vdduVddxy+vddvVddyy+vddwVddzy+wdduVddxz+wddvVddyz+wddwVddzz)
    ddVddxxyyzz=ddvddxx+ddvddyy+ddvddzz
      
    with torch.no_grad():
        dVa=dt*(-duvwVdxyz+dt/2*ukduvwV+ddVddxxyyzz)
        dVc=-dt*(-duvwVdxyz-dt/2*ukduvwV+ddVddxxyyzz)
        dduvwVdxyz,dukduvwV,dddVddxxyyzz=gradients(duvwVdxyz,y),gradients(ukduvwV,y),gradients(ddVddxxyyzz,y)
        dVady=dt*(-dduvwVdxyz+dt/2*dukduvwV+ddVddxxyyzz)
        dVcdy=-dt*(-dduvwVdxyz-dt/2*dukduvwV+ddVddxxyyzz)

    #u*d(duWdx+dvWdy+dwWdz)/dx v*d(duWdx+dvWdy+dwWdz)/dy w*d(duWdx+dvWdy+dwWdz)/dz
    udduWddxx=ub*(ddwddxx*ub+dwdx*dudx+dwdx*dudx+wb*dduddxx)
    uddvWddyx=ub*(ddvddxy*wb+dvdy*dwdx+dvdx*dwdy+vb*ddwddxy)
    uddwWddzx=ub*(2*dwdx*dwdz+2*wb*ddwddzx)

    vdduWddxy=vb*(ddwddxy*ub+dwdx*dudy+dwdy*dudx+wb*dduddxy)
    vddvWddyy=vb*(ddvddyy*wb+dvdy*dwdy+dvdy*dwdy+vb*ddwddyy)
    vddwWddzy=vb*(2*dwdy*dwdz+2*wb*ddwddyz)

    wdduWddxz=wb*(ddwddzx*ub+dwdx*dudz+dwdz*dudx+wb*dduddzx)
    wddvWddyz=wb*(ddvddyz*wb+dvdy*dwdz+dvdz*dwdy+vb*ddwddyz)
    wddwWddzz=wb*(2*dwdz*dwdz+2*wb*ddwddzz)
    
    duvwWdxyz=duWdx+dvWdy+dwWdz
    ukduvwW=(udduWddxx+uddvWddyx+uddwWddzx+vdduWddxy+vddvWddyy+vddwWddzy+wdduWddxz+wddvWddyz+wddwWddzz)
    ddWddxxyyzz=ddwddxx+ddwddyy+ddwddzz
      
    with torch.no_grad():
        dWa=dt*(-duvwWdxyz+dt/2*ukduvwW+ddWddxxyyzz)
        dWc=-dt*(-duvwWdxyz-dt/2*ukduvwW+ddWddxxyyzz)
        dduvwWdxyz,dukduvwW,dddWddxxyyzz=gradients(duvwWdxyz,z),gradients(ukduvwW,z),gradients(ddWddxxyyzz,z)
        dWadz=dt*(-dduvwWdxyz+dt/2*dukduvwW+ddWddxxyyzz)
        dWcdz=-dt*(-dduvwWdxyz-dt/2*dukduvwW+ddWddxxyyzz)
    
    with torch.no_grad():
        dUdx=dudx
        dVdy=dvdy
        dWdz=dwdz

    with torch.no_grad():
        ddpdd=ddpddxx+ddpddyy+ddpddzz
        ukddpddx=ub*ddpddxx+vb*ddpddxy+wb*ddpddzx
        ukddpddy=ub*ddpddxy+vb*ddpddyy+wb*ddpddyz
        ukddpddz=ub*ddpddzx+vb*ddpddyz+wb*ddpddzz

    return ub.detach(),vb.detach(),wb.detach(),pb.detach(),\
           dUdx.detach(),dVdy.detach(),dWdz.detach(),\
           dUa.detach(),dVa.detach(),dWa.detach(),dUadx.detach(),dVady.detach(),dWadz.detach(),\
           dUc.detach(),dVc.detach(),dWc.detach(),dUcdx.detach(),dVcdy.detach(),dWcdz.detach(),\
           ddpdd.detach(),dpdx.detach(),dpdy.detach(),dpdz.detach(),ukddpddx.detach(),ukddpddy.detach(),ukddpddz.detach()


def P_loss(Pmodel,x,y,z,t,dt,p1,p0,dUVWadxyz,dUVWcdxyz,ddpdd):

    pc=Pmodel(x,y,z,t-dt)
    pb=Pmodel(x,y,z,t)
    pa=Pmodel(x,y,z,t+dt)

    ddpdda=(ddpdd + gradients(pa,x,2)+gradients(pa,y,2)+gradients(pa,z,2))/2
    ddpddc=(ddpdd + gradients(pc,x,2)+gradients(pc,y,2)+gradients(pc,z,2))/2
    dpa=-dt*(dUVWadxyz-dt/2*ddpdda)
    dpc=+dt*(dUVWcdxyz+dt/2*ddpddc)

    l1=loss1(pb,p0)/(dt**2)
    #l2=loss2(torch.zeros(p0.shape).to(p0.device),dpa)/(dt**2)
    l2=loss2(dUVWadxyz/dt*2,ddpdda)*(dt**2/4)
    #l3=loss2(dUVWcdxyz/dt*2,-ddpddc)*(dt**2/4)
    #l3=loss3(p0,dpc+p0)/(dt**2)
    return l1+l2

def dP_loss(Pmodel,x,y,z,t,dt):
    pa=Pmodel(x,y,z,t+dt)
    pc=Pmodel(x,y,z,t-dt)
    dPdxa,dPdya,dPdza=gradients(pa,x,1),gradients(pa,y,1),gradients(pa,z,1)
    dPdxc,dPdyc,dPdzc=gradients(pc,x,1),gradients(pc,y,1),gradients(pc,z,1)
    return dPdxa.detach(),dPdya.detach(),dPdza.detach(),dPdxc.detach(),dPdyc.detach(),dPdzc.detach()


def U_loss(Umodel,x,y,z,t,dt,u1,u0,p1,dUa,dUc,dPdxa,dPdxc,ukddPddx):
    
    uc=Umodel(x,y,z,t-dt)
    ub=Umodel(x,y,z,t)
    ua=Umodel(x,y,z,t+dt)

    dUa=dUa-dt*(dPdxa-dt/2*ukddPddx)
    dUc=dUc+dt*(dPdxc+dt/2*ukddPddx)
    l1=loss1(ub,u0)/(dt**2)
    l2=loss2(ua,dUa+u0)/(dt**2)
    #l3=loss3(uc,dUc+u0)/(dt**2)
    return l1+l2

def V_loss(Vmodel,x,y,z,t,dt,v1,v0,p1,dVa,dVc,dPdya,dPdyc,ukddPddy):

    vc=Vmodel(x,y,z,t-dt)
    vb=Vmodel(x,y,z,t)
    va=Vmodel(x,y,z,t+dt)

    dVa=dVa-dt*(dPdya-dt/2*ukddPddy)
    dVc=dVc+dt*(dPdyc+dt/2*ukddPddy)
    l1=loss1(vb,v0)/(dt**2)
    l2=loss2(va,dVa+v0)/(dt**2)
    #l3=loss3(vc,dVc+v0)/(dt**2)
    return l1+l2

def W_loss(Wmodel,x,y,z,t,dt,w1,w0,p1,dWa,dWc,dPdza,dPdzc,ukddPddz):

    wc=Wmodel(x,y,z,t-dt)
    wb=Wmodel(x,y,z,t)
    wa=Wmodel(x,y,z,t+dt)

    dWa=dWa-dt*(dPdza-dt/2*ukddPddz)
    dWc=dWc+dt*(dPdzc+dt/2*ukddPddz)
    l1=loss1(wb,w0)/(dt**2)
    l2=loss2(wa,dWa+w0)/(dt**2)
    #l3=loss3(wc,dWc+v0)/(dt**2)
    return l1+l2