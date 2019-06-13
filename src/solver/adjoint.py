# ------------------------------------------------------------------------
#
# PyFlow: A GPU-accelerated CFD platform written in Python
#
# @file adjoint.py
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

# ----------------------------------------------------
# Navier-Stokes adjoint equation RHS
# ----------------------------------------------------
class rhs_adjPredictor:
    def __init__(self,decomp):
        # Default precision
        prec = decomp.prec

        # Data sizes
        nx_  = decomp.nx_
        ny_  = decomp.ny_
        nz_  = decomp.nz_
        nxo_ = decomp.nxo_
        nyo_ = decomp.nyo_
        nzo_ = decomp.nzo_
        
        # Allocate rhs arrays
        self.rhs_u     = torch.zeros(nx_,ny_,nz_,dtype=prec).to(decomp.device)
        self.rhs_v     = torch.zeros(nx_,ny_,nz_,dtype=prec).to(decomp.device)
        self.rhs_w     = torch.zeros(nx_,ny_,nz_,dtype=prec).to(decomp.device)
        self.FX        = torch.zeros(nxo_,nyo_,nzo_,dtype=prec).to(decomp.device)
        self.FY        = torch.zeros(nxo_,nyo_,nzo_,dtype=prec).to(decomp.device)
        self.FZ        = torch.zeros(nxo_,nyo_,nzo_,dtype=prec).to(decomp.device)
        self.div_vel   = torch.zeros(nxo_,nyo_,nzo_,dtype=prec).to(decomp.device)
        self.interp_SC = torch.zeros(nxo_,nyo_,nzo_,dtype=prec).to(decomp.device)

            
    # ----------------------------------------------------
    # Evaluate the adjoint equation RHS
    # ----------------------------------------------------
    def evaluate(self,state_u_adj,state_v_adj,state_w_adj,
                 state_u,state_v,state_w,VISC,rho,sfsmodel,metric):
        imin_ = metric.imin_; imax_ = metric.imax_+1
        jmin_ = metric.jmin_; jmax_ = metric.jmax_+1
        kmin_ = metric.kmin_; kmax_ = metric.kmax_+1
        
        # Zero the rhs
        self.rhs_u.zero_()
        self.rhs_v.zero_()
        self.rhs_w.zero_()
        
        #print(self.FX.device)
        #print(state_u.grad_x.device)

        # Compute velocity gradients for the viscous flux
        metric.grad_vel_visc(state_u_adj)
        metric.grad_vel_visc(state_v_adj)
        metric.grad_vel_visc(state_w_adj)

        # Compute velocity divergence for the viscous flux
        metric.div_vel_over(state_u_adj,state_v_adj,state_w_adj,self.div_vel)
        self.div_vel.div_( 3.0 )

        # Viscous fluxes -- DOUBLE CHECK NON-LAPLACIAN FORM
        #print(self.FX.device)
        #print(state_u_adj.grad_x.device)
        # xx
        self.FX.copy_( state_u_adj.grad_x )
        self.FX.sub_ ( self.div_vel )
        self.FX.mul_ ( 2.0*VISC/rho )
        # xy
        metric.interp_sc_xy(VISC,self.interp_SC)
        self.FY.copy_( state_u_adj.grad_y )
        self.FY.add_ ( state_v_adj.grad_x )
        self.FY.mul_ ( self.interp_SC/rho )
        # xz
        metric.interp_sc_xz(VISC,self.interp_SC)
        self.FZ.copy_( state_u_adj.grad_z )
        self.FZ.add_ ( state_w_adj.grad_x )
        self.FZ.mul_ ( self.interp_SC/rho )
        # Divergence of the viscous flux
        metric.div_visc(self.FX,self.FY,self.FZ,self.rhs_u)
        
        # yx
        metric.interp_sc_xy(VISC,self.interp_SC)
        self.FX.copy_( state_v_adj.grad_x )
        self.FX.add_ ( state_u_adj.grad_y )
        self.FX.mul_ ( self.interp_SC/rho )
        # yy
        self.FY.copy_( state_v_adj.grad_y )
        self.FY.sub_ ( self.div_vel )
        self.FY.mul_ ( 2.0*VISC/rho )
        # yz
        metric.interp_sc_yz(VISC,self.interp_SC)
        self.FZ.copy_( state_v_adj.grad_z )
        self.FZ.add_ ( state_w_adj.grad_y )
        self.FZ.mul_ ( self.interp_SC/rho )
        # Divergence of the viscous flux
        metric.div_visc(self.FX,self.FY,self.FZ,self.rhs_v)
        
        # zx
        metric.interp_sc_xz(VISC,self.interp_SC)
        self.FX.copy_( state_w_adj.grad_x )
        self.FX.add_ ( state_u_adj.grad_z )
        self.FX.mul_ ( self.interp_SC/rho )
        # zy
        metric.interp_sc_yz(VISC,self.interp_SC)
        self.FY.copy_( state_w_adj.grad_y )
        self.FY.add_ ( state_v_adj.grad_z )
        self.FY.mul_ ( self.interp_SC/rho )
        # zz
        self.FZ.copy_( state_w_adj.grad_z )
        self.FZ.sub_ ( self.div_vel )
        self.FZ.mul_ ( 2.0*VISC/rho )
        # Divergence of the viscous flux
        metric.div_visc(self.FX,self.FY,self.FZ,self.rhs_w)

        
        # Cross-advective fluxes -- divergence form
        # -adj(u)*u
        metric.interp_u_xm(state_u_adj)
        metric.interp_u_xm(state_u)
        metric.vel_conv_xx(state_u_adj,state_u,self.rhs_u,sign=-1.0)
        # -adj(u)*v
        metric.interp_uw_y(state_u_adj)
        metric.interp_vw_x(state_v)
        metric.vel_conv_y(state_u_adj,state_v,self.rhs_u,sign=-1.0)
        # -adj(u)*w
        metric.interp_uv_z(state_u_adj)
        metric.interp_vw_x(state_w)
        metric.vel_conv_z(state_u_adj,state_w,self.rhs_u,sign=-1.0)
        
        # -adj(v)*u
        metric.interp_vw_x(state_v_adj)
        metric.interp_uw_y(state_u)
        metric.vel_conv_x(state_v_adj,state_u,self.rhs_v,sign=-1.0)
        # -adj(v)*v
        metric.interp_v_ym(state_v_adj)
        metric.interp_v_ym(state_v)
        metric.vel_conv_yy(state_v_adj,state_v,self.rhs_v,sign=-1.0)
        # -adj(v)*w
        metric.interp_uv_z(state_v_adj)
        metric.interp_uw_y(state_w)
        metric.vel_conv_z(state_v_adj,state_w,self.rhs_v,sign=-1.0)
        
        # -adj(w)*u
        metric.interp_vw_x(state_w_adj)
        metric.interp_uv_z(state_u)
        metric.vel_conv_x(state_w_adj,state_u,self.rhs_w,sign=-1.0)
        # -adj(w)*v
        metric.interp_uw_y(state_w_adj)
        metric.interp_uv_z(state_v)
        metric.vel_conv_y(state_w_adj,state_v,self.rhs_w,sign=-1.0)
        # -adj(w)*w
        metric.interp_w_zm(state_w_adj)
        metric.interp_w_zm(state_w)
        metric.vel_conv_zz(state_w_adj,state_w,self.rhs_w,sign=-1.0)


        # Cross-advective fluxes
        #   [JFM] Do these terms have physical meaning?
        # Compute gradients of non-adjoint velocity field
        metric.grad_vel_adj(state_u,'u')
        metric.grad_vel_adj(state_v,'v')
        metric.grad_vel_adj(state_w,'w')
        
        # adj(u) equation - compute at x-face
        # -adj(u)*grad1(u)
        self.FX.copy_( state_u.grad_x )
        self.FX.mul_ ( state_u_adj.var )
        
        # -adj(v)*grad1(v) - compute at xy-edge
        # Interpolate v-adj to x-face
        metric.interp_v_ym  (state_v_adj)
        metric.interp_uvwi_x(state_v_adj)
        self.div_vel.copy_( state_v.grad_x )
        self.FY.copy_     ( state_v.grad_x )
        self.FY[:,:-1,:].add_( self.div_vel[:,1:,:] )
        self.FY.mul_( metric.interp_ym )
        self.FY.mul_( state_v_adj.var_i )
        
        # -adj(w)*grad1(w) - compute at xz-edge, interpolate to x-face
        metric.interp_w_zm  (state_w_adj)
        metric.interp_uvwi_x(state_w_adj)
        self.div_vel.copy_( state_w.grad_x )
        self.FZ.copy_     ( state_w.grad_x )
        self.FZ[:,:,:-1].add_( self.div_vel[:,:,1:] )
        self.FZ.mul_( metric.interp_zm )
        self.FZ.mul_( state_w_adj.var_i )

        # Accumulate to RHS
        self.rhs_u.sub_( self.FX[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )
        self.rhs_u.sub_( self.FY[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )
        self.rhs_u.sub_( self.FZ[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )

        # adj(v) equation
        # -adj(u)*grad2(u)
        metric.interp_u_xm  (state_u_adj)
        metric.interp_uvwi_y(state_u_adj)
        self.div_vel.copy_( state_u.grad_y )
        self.FX.copy_     ( state_u.grad_y )
        self.FX[:-1,:,:].add_( self.div_vel[1:,:,:] )
        self.FX.mul_( metric.interp_xm )
        self.FX.mul_( state_u_adj.var_i )
        
        # -adj(v)*grad2(v)
        self.FY.copy_( state_v.grad_y )
        self.FY.mul_ ( state_v_adj.var )

        # -adj(w)*grad2(w)
        metric.interp_w_zm  (state_w_adj)
        metric.interp_uvwi_y(state_w_adj)
        self.div_vel.copy_( state_w.grad_y )
        self.FZ.copy_     ( state_w.grad_y )
        self.FZ[:,:,:-1].add_( self.div_vel[:,:,1:] )
        self.FZ.mul_( metric.interp_zm )
        self.FZ.mul_( state_w_adj.var_i )

        # Accumulate to RHS
        self.rhs_v.sub_( self.FX[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )
        self.rhs_v.sub_( self.FY[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )
        self.rhs_v.sub_( self.FZ[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )

        # adj(w) equation
        # -adj(u)*grad3(u)
        metric.interp_u_xm  (state_u_adj)
        metric.interp_uvwi_z(state_u_adj)
        self.div_vel.copy_( state_u.grad_z )
        self.FX.copy_     ( state_u.grad_z )
        self.FX[:-1,:,:].add_( self.div_vel[1:,:,:] )
        self.FX.mul_( metric.interp_xm )
        self.FX.mul_( state_u_adj.var_i )

        # -adj(v)*grad3(v)
        metric.interp_v_ym  (state_v_adj)
        metric.interp_uvwi_z(state_v_adj)
        self.div_vel.copy_( state_v.grad_z )
        self.FY.copy_     ( state_v.grad_z )
        self.FY[:,:-1,:].add_( self.div_vel[:,1:,:] )
        self.FY.mul_( metric.interp_ym )
        self.FY.mul_( state_v_adj.var_i )

        # -adj(w)*grad3(w)
        self.FZ.copy_( state_w.grad_z )
        self.FZ.mul_ ( state_w_adj.var )

        # Accumulate to RHS
        self.rhs_w.sub_( self.FX[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )
        self.rhs_w.sub_( self.FY[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )
        self.rhs_w.sub_( self.FZ[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )



        # Closure model terms -- delta^t

        # Evaluate the SFS model using PyTorch automatic differentiation
        # Enable computational graph generation
        # Gradients in model inputs need to be computed using non-in-place operations
