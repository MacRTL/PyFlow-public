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

# ----------------------------------------------------
# Velocity predictor rhs
# ----------------------------------------------------
def rhs_predictor(rhs_u,rhs_v,rhs_w,state_u,state_v,state_w,uConv,vConv,wConv,mu,rho,metric):

    # Zero the rhs
    rhs_u.zero_()
    rhs_v.zero_()
    rhs_w.zero_()
        
    # Compute velocity gradients for the viscous flux
    metric.grad_vel_visc(state_u)
    metric.grad_vel_visc(state_v)
    metric.grad_vel_visc(state_w)
        
    # Scalar diffusive fluxes
    # x
    FX = mu/rho * state_u.grad_x
    FY = mu/rho * state_u.grad_y
    FZ = mu/rho * state_u.grad_z
    metric.div_visc(FX,FY,FZ,rhs_u)
    
    # y
    FX = mu/rho * state_v.grad_x
    FY = mu/rho * state_v.grad_y
    FZ = mu/rho * state_v.grad_z
    metric.div_visc(FX,FY,FZ,rhs_v)

    # z
    FX = mu/rho * state_w.grad_x
    FY = mu/rho * state_w.grad_y
    FZ = mu/rho * state_w.grad_z
    metric.div_visc(FX,FY,FZ,rhs_w)
    
    #Diffusion_term_u = state_u.grad_xx + state_u.grad_yy + state_u.grad_zz
    #Diffusion_term_v = state_v.grad_xx + state_v.grad_yy + state_v.grad_zz
    #Diffusion_term_w = state_w.grad_xx + state_w.grad_yy + state_w.grad_zz

    
    # Scalar advective fluxes
    # xx
    metric.interp_u_xm(state_u)
    metric.interp_u_xm(uConv)
    metric.vel_conv_xx(state_u,uConv,rhs_u)
    # xy
    metric.interp_uw_y(state_u)
    metric.interp_vw_x(vConv)
    metric.vel_conv_y(state_u,vConv,rhs_u)
    # xz
    metric.interp_uv_z(state_u)
    metric.interp_vw_x(wConv)
    metric.vel_conv_z(state_u,wConv,rhs_u)
    
    # yx
    metric.interp_vw_x(state_v)
    metric.interp_uw_y(uConv)
    metric.vel_conv_x(state_v,uConv,rhs_v)
    # yy
    metric.interp_v_ym(state_v)
    metric.interp_v_ym(vConv)
    metric.vel_conv_yy(state_v,vConv,rhs_v)
    # yz
    metric.interp_uv_z(state_v)
    metric.interp_uw_y(wConv)
    metric.vel_conv_z(state_v,wConv,rhs_v)
    
    # zx
    metric.interp_vw_x(state_w)
    metric.interp_uv_z(uConv)
    metric.vel_conv_x(state_w,uConv,rhs_w)
    # zy
    metric.interp_uw_y(state_w)
    metric.interp_uv_z(vConv)
    metric.vel_conv_y(state_w,vConv,rhs_w)
    # zz
    metric.interp_w_zm(state_w)
    metric.interp_w_zm(wConv)
    metric.vel_conv_zz(state_w,wConv,rhs_w)
        
    
    # Scalar advection
    #Nonlinear_term_u = state_u.grad_x*uMax + state_u.grad_y*vMax  + state_u.grad_z*wMax
    #Nonlinear_term_v = state_v.grad_x*uMax + state_v.grad_y*vMax  + state_v.grad_z*wMax
    #Nonlinear_term_w = state_w.grad_x*uMax + state_w.grad_y*vMax  + state_w.grad_z*wMax
        
    # Navier-Stokes
    #Nonlinear_term_u = u_x*u + u_y*v  + u_z*w
    #Nonlinear_term_v = v_x*u + v_y*v  + v_z*w
    #Nonlinear_term_w = w_x*u + w_y*v  + w_z*w

    # Accumulate to rhs
    #rhs_u = (mu/rho)*Diffusion_term_u - Nonlinear_term_u
    #rhs_v = (mu/rho)*Diffusion_term_v - Nonlinear_term_v
    #rhs_w = (mu/rho)*Diffusion_term_w - Nonlinear_term_w


    
def SaveForLater():

    # Navier-Stokes viscous fluxes
    # x
    FX = 2.0*mu/rho * state_u.grad_x
    FY = mu/rho * (state_u.grad_y + state_v.grad_x)
    FZ = mu/rho * (state_u.grad_z + state_w.grad_x)
    metric.div_visc_x(FX,rhs_u)
    metric.div_visc_y(FY,rhs_u)
    metric.div_visc_z(FZ,rhs_u)
    
    # y
    FX = mu/rho * (state_v.grad_x + state_u.grad_y)
    FY = 2.0*mu/rho * state_v.grad_y
    FZ = mu/rho * (state_v.grad_z + state_w.grad_y)
    metric.div_visc_x(FX,rhs_v)
    metric.div_visc_y(FY,rhs_v)
    metric.div_visc_z(FZ,rhs_v)

    # z
    FX = mu/rho * (state_w.grad_x + state_u.grad_z)
    FY = mu/rho * (state_w.grad_y + state_v.grad_z)
    FZ = 2.0*mu/rho * state_w.grad_z
    metric.div_visc_x(FX,rhs_w)
    metric.div_visc_y(FY,rhs_w)
    metric.div_visc_z(FZ,rhs_w)
    
    
    # Navier-Stokes convective fluxes
    # xx
    metric.interp_u_xm(state_u)
    metric.vel_conv_xx(state_u,state_u,rhs_u)
    # xy
    metric.interp_v_ym(state_u)
    metric.interp_u_xm(state_v)
    metric.vel_conv_xy(state_u,state_v,rhs_u)
    # xz
    metric.interp_w_zm(state_u)
    metric.interp_u_xm(state_w)
    metric.vel_conv_xz(state_u,state_w,rhs_u)
    
    # yx
    metric.interp_u_xm(state_v)
    metric.interp_v_ym(state_u)
    metric.vel_conv_xx(state_v,state_u,rhs_v)
    # yy
    metric.interp_v_ym(state_v)
    metric.vel_conv_xy(state_v,state_v,rhs_v)
    # yz
    metric.interp_w_zm(state_v)
    metric.interp_v_ym(state_w)
    metric.vel_conv_xz(state_v,state_w,rhs_v)
    
    # zx
    metric.interp_u_xm(state_w)
    metric.interp_w_zm(state_u)
    metric.vel_conv_xx(state_w,state_u,rhs_w)
    # zy
    metric.interp_v_ym(state_w)
    metric.interp_w_zm(state_v)
    metric.vel_conv_xy(state_w,state_v,rhs_w)
    # zz
    metric.interp_w_zm(state_w)
    metric.vel_conv_xz(state_w,state_w,rhs_w)
