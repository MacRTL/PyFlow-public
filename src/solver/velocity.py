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
import sys

# Load PyFlow modules
sys.path.append("../data")
import state

# ----------------------------------------------------
# Scalar advection-diffusion equation RHS
# ----------------------------------------------------
class rhs_scalar:
    def __init__(self,geo,uConv,vConv,wConv):
        # Default precision
        prec = geo.prec

        # Data sizes
        nx_ = geo.nx_
        ny_ = geo.ny_
        nz_ = geo.nz_
        
        IC_ones_np = np.ones( (nx_,ny_,nz_) )
        self.state_uConv = state.data_P(geo,uConv*IC_ones_np)
        self.state_vConv = state.data_P(geo,vConv*IC_ones_np)
        self.state_wConv = state.data_P(geo,wConv*IC_ones_np)
        del IC_ones_np
        
        # Allocate rhs array
        if (torch.cuda.is_available()):
            self.rhs_u = torch.zeros(nx_,ny_,nz_,dtype=prec).cuda()
            self.rhs_v = torch.zeros(nx_,ny_,nz_,dtype=prec).cuda()
            self.rhs_w = torch.zeros(nx_,ny_,nz_,dtype=prec).cuda()
            self.FX    = torch.zeros(nx_+1,ny_,nz_,dtype=prec).cuda()
            self.FY    = torch.zeros(nx_,ny_+1,nz_,dtype=prec).cuda()
            self.FZ    = torch.zeros(nx_,ny_,nz_+1,dtype=prec).cuda()
        else:
            self.rhs_u = torch.zeros(nx_,ny_,nz_,dtype=prec)
            self.rhs_v = torch.zeros(nx_,ny_,nz_,dtype=prec)
            self.rhs_w = torch.zeros(nx_,ny_,nz_,dtype=prec)
            self.FX    = torch.zeros(nx_+1,ny_,nz_,dtype=prec)
            self.FY    = torch.zeros(nx_,ny_+1,nz_,dtype=prec)
            self.FZ    = torch.zeros(nx_,ny_,nz_+1,dtype=prec)

            
    # ----------------------------------------------------
    # Evaluate the RHS
    def evaluate(self,state_u,state_v,state_w,mu,rho,metric):
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
        self.FX = mu/rho * state_u.grad_x
        self.FY = mu/rho * state_u.grad_y
        self.FZ = mu/rho * state_u.grad_z
        metric.div_visc(self.FX,self.FY,self.FZ,self.rhs_u)
        
        # y
        self.FX = mu/rho * state_v.grad_x
        self.FY = mu/rho * state_v.grad_y
        self.FZ = mu/rho * state_v.grad_z
        metric.div_visc(self.FX,self.FY,self.FZ,self.rhs_v)
        
        # z
        self.FX = mu/rho * state_w.grad_x
        self.FY = mu/rho * state_w.grad_y
        self.FZ = mu/rho * state_w.grad_z
        metric.div_visc(self.FX,self.FY,self.FZ,self.rhs_w)
        
        # Scalar advective fluxes
        # xx
        metric.interp_u_xm(state_u)
        metric.interp_u_xm(self.state_uConv)
        metric.vel_conv_xx(state_u,self.state_uConv,self.rhs_u)
        # xy
        metric.interp_uw_y(state_u)
        metric.interp_vw_x(self.state_vConv)
        metric.vel_conv_y(state_u,self.state_vConv,self.rhs_u)
        # xz
        metric.interp_uv_z(state_u)
        metric.interp_vw_x(self.state_wConv)
        metric.vel_conv_z(state_u,self.state_wConv,self.rhs_u)
        
        # yx
        metric.interp_vw_x(state_v)
        metric.interp_uw_y(self.state_uConv)
        metric.vel_conv_x(state_v,self.state_uConv,self.rhs_v)
        # yy
        metric.interp_v_ym(state_v)
        metric.interp_v_ym(self.state_vConv)
        metric.vel_conv_yy(state_v,self.state_vConv,self.rhs_v)
        # yz
        metric.interp_uv_z(state_v)
        metric.interp_uw_y(self.state_wConv)
        metric.vel_conv_z(state_v,self.state_wConv,self.rhs_v)
        
        # zx
        metric.interp_vw_x(state_w)
        metric.interp_uv_z(self.state_uConv)
        metric.vel_conv_x(state_w,self.state_uConv,self.rhs_w)
        # zy
        metric.interp_uw_y(state_w)
        metric.interp_uv_z(self.state_vConv)
        metric.vel_conv_y(state_w,self.state_vConv,self.rhs_w)
        # zz
        metric.interp_w_zm(state_w)
        metric.interp_w_zm(self.state_wConv)
        metric.vel_conv_zz(state_w,self.state_wConv,self.rhs_w)



# ----------------------------------------------------
# Navier-Stokes equation RHS for pressure-projection
# ----------------------------------------------------
class rhs_NavierStokes:
    def __init__(self,geo):
        # Default precision
        prec = geo.prec

        # Data sizes
        nx_ = geo.nx_
        ny_ = geo.ny_
        nz_ = geo.nz_
        
        # Allocate rhs array
        if (torch.cuda.is_available()):
            self.rhs_u = torch.zeros(nx_,ny_,nz_,dtype=prec).cuda()
            self.rhs_v = torch.zeros(nx_,ny_,nz_,dtype=prec).cuda()
            self.rhs_w = torch.zeros(nx_,ny_,nz_,dtype=prec).cuda()
            self.FX    = torch.zeros(nx_+1,ny_+1,nz_+1,dtype=prec).cuda()
            self.FY    = torch.zeros(nx_+1,ny_+1,nz_+1,dtype=prec).cuda()
            self.FZ    = torch.zeros(nx_+1,ny_+1,nz_+1,dtype=prec).cuda()
        else:
            self.rhs_u = torch.zeros(nx_,ny_,nz_,dtype=prec)
            self.rhs_v = torch.zeros(nx_,ny_,nz_,dtype=prec)
            self.rhs_w = torch.zeros(nx_,ny_,nz_,dtype=prec)
            self.FX    = torch.zeros(nx_+1,ny_+1,nz_+1,dtype=prec)
            self.FY    = torch.zeros(nx_+1,ny_+1,nz_+1,dtype=prec)
            self.FZ    = torch.zeros(nx_+1,ny_+1,nz_+1,dtype=prec)

            
    # ----------------------------------------------------
    # Evaluate the RHS
    def evaluate(self,state_u,state_v,state_w,mu,rho,metric):
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
        self.FX = 2.0*mu/rho * state_u.grad_x
        self.FY = mu/rho * (state_u.grad_y + state_v.grad_x)
        self.FZ = mu/rho * (state_u.grad_z + state_w.grad_x)
        metric.div_visc(self.FX,self.FY,self.FZ,self.rhs_u)
        
        # y
        self.FX = mu/rho * (state_v.grad_x + state_u.grad_y)
        self.FY = 2.0*mu/rho * state_v.grad_y
        self.FZ = mu/rho * (state_v.grad_z + state_w.grad_y)
        metric.div_visc(self.FX,self.FY,self.FZ,self.rhs_v)
        
        # z
        self.FX = mu/rho * (state_w.grad_x + state_u.grad_z)
        self.FY = mu/rho * (state_w.grad_y + state_v.grad_z)
        self.FZ = 2.0*mu/rho * state_w.grad_z
        metric.div_visc(self.FX,self.FY,self.FZ,self.rhs_w)
        
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
        metric.vel_conv_zz(state_w,state_w,self.rhs_w)

