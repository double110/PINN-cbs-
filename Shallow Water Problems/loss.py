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
    hb=Pmodel(x,y,t)

    ##各阶偏导
    dhdx,dhdy,dudx,dudy,dvdx,dvdy=gradients(hb,x),gradients(hb,y),gradients(ub,x),gradients(ub,y),gradients(vb,x),gradients(vb,y)
    ddhddxx,ddhddxy,ddhddyy = gradients(dhdx,x),gradients(dhdx,y),gradients(dhdy,y)
    dduddxx,dduddxy,dduddyy = gradients(dudx,x),gradients(dudx,y),gradients(dudy,y)
    ddvddxx,ddvddxy,ddvddyy = gradients(dvdx,x),gradients(dvdx,y),gradients(dvdy,y)
    

    duUdx,dvUdy = 2*ub*dudx*hb+ub*ub*dhdx,dvdy*ub*hb+vb*dudy*hb+vb*ub*dhdy
    duVdx,dvVdy = dvdx*ub*hb+vb*dudx*hb+vb*ub*dhdx,2*vb*dvdy*hb+vb*vb*dhdy

    #u*d(duUdx+dvUdy)/dx v*d(duUdx+dvUdy)/dy
    udduUddxx=ub*(2*dudx*dudx*hb+2*ub*dduddxx*hb+2*ub*dudx*dhdx+2*ub*dudx*dhdx+ub*ub*ddhddxx)
    uddvUddyx=ub*(ddvddxy*ub*hb+dvdy*dudx*hb+dvdy*ub*dhdx+dvdx*dudy*hb+vb*dduddxy*hb+vb*dudy*dhdx+dvdx*ub*dhdy+vb*dudx*dhdy+vb*ub*ddhddxy)
    vdduUddxy=vb*(2*dudy*dudx*hb+2*ub*dduddxy*hb+2*ub*dudx*dhdy+2*ub*dudy*dhdx+ub*ub*ddhddxy)
    vddvUddyy=vb*(ddvddyy*ub*hb+dvdy*dudy*hb+dvdy*ub*dhdy+dvdy*dudy*hb+vb*dduddyy*hb+vb*dudy*dhdy+dvdy*ub*dhdy+vb*dudy*dhdy+vb*ub*ddhddyy)
    
    duvUdxy=duUdx+dvUdy
    ukduvU=(udduUddxx+uddvUddyx+vdduUddxy+vddvUddyy)
      
    with torch.no_grad():
        dUa=dt*(-duvUdxy+dt/2*ukduvU)
        dUc=-dt*(-duvUdxy-dt/2*ukduvU)
        dduvUdxy,dukduvU=gradients(duvUdxy,x),gradients(ukduvU,x)
        dUadx=dt*(-dduvUdxy+dt/2*dukduvU)
        dUcdx=-dt*(-dduvUdxy-dt/2*dukduvU)

    #u*d(duVdx+dvVdy)/dx v*d(duVdx+dvVdy)/dy
    uddvVddyx=ub*(2*dvdx*dvdy*hb+2*vb*ddvddxy*hb+2*vb*dvdy*dhdx+2*vb*dvdx*dhdy+vb*vb*ddhddxy)
    udduVddxx=ub*(ddvddxx*ub*hb+dvdx*dudx*hb+dvdx*ub*dhdx+dvdx*dudx*hb+vb*dduddxx*hb+vb*dudx*dhdx+dvdx*ub*dhdx+vb*dudx*dhdx+vb*ub*ddhddxx)
    vddvVddyy=vb*(2*dvdy*dvdy*hb+2*vb*ddvddyy*hb+2*vb*dvdy*dhdy+2*vb*dvdy*dhdy+vb*vb*ddhddyy)
    vdduVddxy=vb*(ddvddxy*ub*hb+dvdx*dudy*hb+dvdx*ub*dhdy+dvdy*dudx*hb+vb*dduddxy*hb+vb*dudx*dhdy+dvdy*ub*dhdx+vb*dudy*dhdx+vb*ub*ddhddxy)

    duvVdxy=duVdx+dvVdy
    ukduvV=(uddvVddyx+udduVddxx+vddvVddyy+vdduVddxy)
    
    with torch.no_grad():
        dVa=dt*(-duvVdxy+dt/2*ukduvV)
        dVc=-dt*(-duvVdxy-dt/2*ukduvV)
        dduvVdxy,dukduvV=gradients(duvVdxy,y),gradients(ukduvV,y)
        dVady=dt*(-dduvVdxy+dt/2*dukduvV)
        dVcdy=-dt*(-dduvVdxy-dt/2*dukduvV)
    
    with torch.no_grad():
        dUdx=dudx*hb+ub*dhdx
        dVdy=dvdy*hb+vb*dhdy

    H=meandepth(x,y,t)
    dPdx,dPdy=gradients(hb**2-H**2,x,1)/2,gradients(hb**2-H**2,y,1)/2
    with torch.no_grad():
        ddPdxx,ddPdyy,ddPdxy=gradients(dPdx,x,1),gradients(dPdy,y,1),gradients(dPdx,y,1)
        ddPdd=ddPdxx+ddPdyy
        ukddPddx=ub*ddPdxx+vb*ddPdxy
        ukddPddy=ub*ddPdxy+vb*ddPdyy
    return ub.detach(),vb.detach(),hb.detach(),dUdx.detach(),dVdy.detach(),\
           dUa.detach(),dVa.detach(),dUadx.detach(),dVady.detach(),\
           dUc.detach(),dVc.detach(),dUcdx.detach(),dVcdy.detach(),\
           ddPdd.detach(),dPdx.detach(),dPdy.detach(),ukddPddx.detach(),ukddPddy.detach()


def P_loss(Pmodel,x,y,t,dt,h1,h0,dUVadxy,dUVcdxy,ddPdd):
    H=meandepth(x,y,t)

    hc=Pmodel(x,y,t-dt)
    hb=Pmodel(x,y,t)
    ha=Pmodel(x,y,t+dt)

    ddPdda=(ddPdd + gradients(ha**2-H**2,x,2)/2+gradients(ha**2-H**2,y,2)/2)/2
    ddPddc=(ddPdd + gradients(hc**2-H**2,x,2)/2+gradients(hc**2-H**2,y,2)/2)/2
    dha=-dt*(dUVadxy-dt/2*ddPdda)
    dhc=+dt*(dUVcdxy+dt/2*ddPddc)

    l1=loss1(hb,h0)/(dt**2)
    l2=loss2(ha,dha.detach()+h0)/(dt**2)
    l3=loss3(hc,dhc.detach()+h0)/(dt**2)
    return l1+l2+l3

def dP_loss(Pmodel,x,y,t,dt):
    Ha=meandepth(x,y,t+dt)
    Hc=meandepth(x,y,t-dt)

    ha=Pmodel(x,y,t+dt)
    hc=Pmodel(x,y,t-dt)
    dPdxa,dPdya=gradients(ha**2-Ha**2,x,1)/2,gradients(ha**2-Ha**2,y,1)/2
    dPdxc,dPdyc=gradients(hc**2-Hc**2,x,1)/2,gradients(hc**2-Hc**2,y,1)/2
    return dPdxa.detach(),dPdya.detach(),dPdxc.detach(),dPdyc.detach()


def U_loss(Umodel,x,y,t,dt,u1,u0,h1,dUa,dUc,dPdxa,dPdxc,ukddPddx):
    
    uc=Umodel(x,y,t-dt)
    ub=Umodel(x,y,t)
    ua=Umodel(x,y,t+dt)

    dUa=dUa-dt*(dPdxa-dt/2*ukddPddx)
    dUc=dUc+dt*(dPdxc+dt/2*ukddPddx)
    l1=loss1(ub,u0)/(dt**2)*(1e+2)
    l2=loss2(ua,dUa/h1+u0)/(dt**2)*(1e+2)
    l3=loss3(uc,dUc/h1+u0)/(dt**2)*(1e+2)
    return l1+l2+l3

def V_loss(Vmodel,x,y,t,dt,v1,v0,h1,dVa,dVc,dPdya,dPdyc,ukddPddy):

    vc=Vmodel(x,y,t-dt)
    vb=Vmodel(x,y,t)
    va=Vmodel(x,y,t+dt)

    dVa=dVa-dt*(dPdya-dt/2*ukddPddy)
    dVc=dVc+dt*(dPdyc+dt/2*ukddPddy)
    l1=loss1(vb,v0)/(dt**2)*(1e+2)
    l2=loss2(va,dVa/h1+v0)/(dt**2)*(1e+2)
    l3=loss3(vc,dVc/h1+v0)/(dt**2)*(1e+2)
    return l1+l2+l3