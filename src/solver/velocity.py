# ------------------------------------------------------------------------
#
# PyFlow: A GPU-accelerated CFD platform written in Python
#
# @file velocity.py
#
# The MIT License (MIT)
# Copyright (c) 2019 University of Illinois Board of Trustees
#
# Permission is hereby granted, free of charge, to any person 
# obtaining a copy of this software and associated documentation 
# files (the "Software"), to deal in the Software without 
# restriction, including without limitation the rights to use, 
# copy, modify, merge, publish, distribute, sublicense, and/or 
# sell copies of the Software, and to permit persons to whom the 
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be 
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
# OTHER DEALINGS IN THE SOFTWARE.
# 
# ------------------------------------------------------------------------

import numpy as np
import torch

# ----------------------------------------------------
# Scalar advection-diffusion equation RHS
# ----------------------------------------------------
class rhs_scalar:
    def __init__(self,IC_zeros):
        # Allocate rhs array
        if (torch.cuda.is_available()):
            self.rhs_u = torch.FloatTensor(IC_zeros).cuda()
            self.rhs_v = torch.FloatTensor(IC_zeros).cuda()
            self.rhs_w = torch.FloatTensor(IC_zeros).cuda()
        else:
            self.rhs_u = torch.FloatTensor(IC_zeros)
            self.rhs_v = torch.FloatTensor(IC_zeros)
            self.rhs_w = torch.FloatTensor(IC_zeros)

            
    # ----------------------------------------------------
    # Evaluate the RHS
    def evaluate(self,state_u,state_v,state_w,uConv,vConv,wConv,mu,rho,metric):
        # Zero the rhs
        self.rhs_u.zero_()
        self.rhs_v.zero_()
        self.rhs_w.zero_()
        
        # Compute velocity gradients for the viscous flux
        metric.grad_vel_visc(state_u)
        metric.grad_vel_visc(state_v)
        metric.grad_vel_visc(state_w)
        
        # Scalar diffusive fluxes
        # x
        FX = mu/rho * state_u.grad_x
        FY = mu/rho * state_u.grad_y
        FZ = mu/rho * state_u.grad_z
        metric.div_visc(FX,FY,FZ,self.rhs_u)
        
        # y
        FX = mu/rho * state_v.grad_x
        FY = mu/rho * state_v.grad_y
        FZ = mu/rho * state_v.grad_z
        metric.div_visc(FX,FY,FZ,self.rhs_v)
        
        # z
        FX = mu/rho * state_w.grad_x
        FY = mu/rho * state_w.grad_y
        FZ = mu/rho * state_w.grad_z
        metric.div_visc(FX,FY,FZ,self.rhs_w)
        
        # Scalar advective fluxes
        # xx
        metric.interp_u_xm(state_u)
        metric.interp_u_xm(uConv)
        metric.vel_conv_xx(state_u,uConv,self.rhs_u)
        # xy
        metric.interp_uw_y(state_u)
        metric.interp_vw_x(vConv)
        metric.vel_conv_y(state_u,vConv,self.rhs_u)
        # xz
        metric.interp_uv_z(state_u)
        metric.interp_vw_x(wConv)
        metric.vel_conv_z(state_u,wConv,self.rhs_u)
        
        # yx
        metric.interp_vw_x(state_v)
        metric.interp_uw_y(uConv)
        metric.vel_conv_x(state_v,uConv,self.rhs_v)
        # yy
        metric.interp_v_ym(state_v)
        metric.interp_v_ym(vConv)
        metric.vel_conv_yy(state_v,vConv,self.rhs_v)
        # yz
        metric.interp_uv_z(state_v)
        metric.interp_uw_y(wConv)
        metric.vel_conv_z(state_v,wConv,self.rhs_v)
        
        # zx
        metric.interp_vw_x(state_w)
        metric.interp_uv_z(uConv)
        metric.vel_conv_x(state_w,uConv,self.rhs_w)
        # zy
        metric.interp_uw_y(state_w)
        metric.interp_uv_z(vConv)
        metric.vel_conv_y(state_w,vConv,self.rhs_w)
        # zz
        metric.interp_w_zm(state_w)
        metric.interp_w_zm(wConv)
        metric.vel_conv_zz(state_w,wConv,self.rhs_w)



# ----------------------------------------------------
# Navier-Stokes equation RHS for pressure-projection
# ----------------------------------------------------
class rhs_NavierStokes:
    def __init__(self,IC_zeros):
        # Allocate rhs array
        if (torch.cuda.is_available()):
            self.rhs_u = torch.FloatTensor(IC_zeros).cuda()
            self.rhs_v = torch.FloatTensor(IC_zeros).cuda()
            self.rhs_w = torch.FloatTensor(IC_zeros).cuda()
        else:
            self.rhs_u = torch.FloatTensor(IC_zeros)
            self.rhs_v = torch.FloatTensor(IC_zeros)
            self.rhs_w = torch.FloatTensor(IC_zeros)

            
    # ----------------------------------------------------
    # Evaluate the RHS
    def evaluate(self,state_u,state_v,state_w,uConv,vConv,wConv,mu,rho,metric):
        # Zero the rhs
        self.rhs_u.zero_()
        self.rhs_v.zero_()
        self.rhs_w.zero_()
        
        # Compute velocity gradients for the viscous flux
        metric.grad_vel_visc(state_u)
        metric.grad_vel_visc(state_v)
        metric.grad_vel_visc(state_w)
        
        # Viscous fluxes
        # x
        FX = 2.0*mu/rho * state_u.grad_x
        FY = mu/rho * (state_u.grad_y + state_v.grad_x)
        FZ = mu/rho * (state_u.grad_z + state_w.grad_x)
        metric.div_visc(FX,FY,FZ,self.rhs_u)
        
        # y
        FX = mu/rho * (state_v.grad_x + state_u.grad_y)
        FY = 2.0*mu/rho * state_v.grad_y
        FZ = mu/rho * (state_v.grad_z + state_w.grad_y)
        metric.div_visc(FX,FY,FZ,self.rhs_v)
        
        # z
        FX = mu/rho * (state_w.grad_x + state_u.grad_z)
        FY = mu/rho * (state_w.grad_y + state_v.grad_z)
        FZ = 2.0*mu/rho * state_w.grad_z
        metric.div_visc(FX,FY,FZ,self.rhs_w)
        
        # Advective fluxes
        # xx
        metric.interp_u_xm(state_u)
        metric.vel_conv_xx(state_u,state_u,self.rhs_u)
        # xy
        metric.interp_uw_y(state_u)
        metric.interp_vw_x(state_v)
        metric.vel_conv_y(state_u,state_v,self.rhs_u)
        # xz
        metric.interp_uv_z(state_u)
        metric.interp_vw_x(state_w)
        metric.vel_conv_z(state_u,state_w,self.rhs_u)
        
        # yx
        metric.interp_vw_x(state_v)
        metric.interp_uw_y(state_u)
        metric.vel_conv_x(state_v,state_u,self.rhs_v)
        # yy
        metric.interp_v_ym(state_v)
        metric.interp_v_ym(state_v)
        metric.vel_conv_yy(state_v,state_v,self.rhs_v)
        # yz
        metric.interp_uv_z(state_v)
        metric.interp_uw_y(state_w)
        metric.vel_conv_z(state_v,state_w,self.rhs_v)
        
        # zx
        metric.interp_vw_x(state_w)
        metric.interp_uv_z(state_u)
        metric.vel_conv_x(state_w,state_u,self.rhs_w)
        # zy
        metric.interp_uw_y(state_w)
        metric.interp_uv_z(state_v)
        metric.vel_conv_y(state_w,state_v,self.rhs_w)
        # zz
        metric.interp_w_zm(state_w)
        metric.interp_w_zm(state_w)
        metric.vel_conv_zz(state_w,state_w,self.rhs_w)

