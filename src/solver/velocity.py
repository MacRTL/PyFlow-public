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
    def __init__(self,decomp,uConv,vConv,wConv):
        # Default precision
        prec = decomp.prec

        # Data sizes
        nx_ = decomp.nx_
        ny_ = decomp.ny_
        nz_ = decomp.nz_
        
        IC_ones_np = np.ones( (nx_,ny_,nz_) )
        self.state_uConv = state.state_P(decomp,uConv*IC_ones_np)
        self.state_vConv = state.state_P(decomp,vConv*IC_ones_np)
        self.state_wConv = state.state_P(decomp,wConv*IC_ones_np)
        del IC_ones_np
        
        # Allocate rhs arrays
        self.rhs_u = torch.zeros(nx_,ny_,nz_,dtype=prec).to(decomp.device)
        self.rhs_v = torch.zeros(nx_,ny_,nz_,dtype=prec).to(decomp.device)
        self.rhs_w = torch.zeros(nx_,ny_,nz_,dtype=prec).to(decomp.device)
        self.FX    = torch.zeros(nx_+1,ny_+1,nz_+1,dtype=prec).to(decomp.device)
        self.FY    = torch.zeros(nx_+1,ny_+1,nz_+1,dtype=prec).to(decomp.device)
        self.FZ    = torch.zeros(nx_+1,ny_+1,nz_+1,dtype=prec).to(decomp.device)

            
    # ----------------------------------------------------
    # Evaluate the RHS
    def evaluate(self,state_u,state_v,state_w,VISC,rho,sfsmodel,metric):
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
        self.FX = VISC/rho * state_u.grad_x
        self.FY = VISC/rho * state_u.grad_y
        self.FZ = VISC/rho * state_u.grad_z
        metric.div_visc(self.FX,self.FY,self.FZ,self.rhs_u)
        
        # y
        self.FX = VISC/rho * state_v.grad_x
        self.FY = VISC/rho * state_v.grad_y
        self.FZ = VISC/rho * state_v.grad_z
        metric.div_visc(self.FX,self.FY,self.FZ,self.rhs_v)
        
        # z
        self.FX = VISC/rho * state_w.grad_x
        self.FY = VISC/rho * state_w.grad_y
        self.FZ = VISC/rho * state_w.grad_z
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
    # Evaluate the RHS
    def evaluate(self,state_u,state_v,state_w,VISC,rho,sfsmodel,metric):
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
        metric.grad_vel_visc(state_u)
        metric.grad_vel_visc(state_v)
        metric.grad_vel_visc(state_w)

        # Compute velocity divergence for the viscous flux
        metric.div_vel_over(state_u,state_v,state_w,self.div_vel)
        self.div_vel.div_( 3.0 )

        # Viscous fluxes
        # VISC includes the eddy viscosity if required
        #print(self.FX.device)
        #print(state_u.grad_x.device)
        # xx
        self.FX.copy_( state_u.grad_x )
        self.FX.sub_ ( self.div_vel )
        self.FX.mul_ ( 2.0*VISC/rho )
        # xy
        metric.interp_sc_xy(VISC,self.interp_SC)
        self.FY.copy_( state_u.grad_y )
        self.FY.add_ ( state_v.grad_x )
        self.FY.mul_ ( self.interp_SC/rho )
        # xz
        metric.interp_sc_xz(VISC,self.interp_SC)
        self.FZ.copy_( state_u.grad_z )
        self.FZ.add_ ( state_w.grad_x )
        self.FZ.mul_ ( self.interp_SC/rho )
        # Add the modeled SFS flux
        if (sfsmodel.modelType=='tensor'):
            self.FX.add_( sfsmodel.FXX )
            metric.interp_sc_xy(sfsmodel.FXY,self.interp_SC)
            self.FY.add_( self.interp_SC )
            metric.interp_sc_xz(sfsmodel.FXZ,self.interp_SC)
            self.FZ.add_( self.interp_SC )
        # Divergence of the viscous+SFS flux
        metric.div_visc(self.FX,self.FY,self.FZ,self.rhs_u)
        
        # yx
        metric.interp_sc_xy(VISC,self.interp_SC)
        self.FX.copy_( state_v.grad_x )
        self.FX.add_ ( state_u.grad_y )
        self.FX.mul_ ( self.interp_SC/rho )
        # yy
        self.FY.copy_( state_v.grad_y )
        self.FY.sub_ ( self.div_vel )
        self.FY.mul_ ( 2.0*VISC/rho )
        # yz
        metric.interp_sc_yz(VISC,self.interp_SC)
        self.FZ.copy_( state_v.grad_z )
        self.FZ.add_ ( state_w.grad_y )
        self.FZ.mul_ ( self.interp_SC/rho )
        # Add the modeled SFS flux
        if (sfsmodel.modelType=='tensor'):
            metric.interp_sc_xy(sfsmodel.FXY,self.interp_SC)
            self.FX.add_( self.interp_SC )
            self.FY.add_( sfsmodel.FYY )
            metric.interp_sc_yz(sfsmodel.FYZ,self.interp_SC)
            self.FZ.add_( self.interp_SC )
        # Divergence of the viscous+SFS flux
        metric.div_visc(self.FX,self.FY,self.FZ,self.rhs_v)
        
        # zx
        metric.interp_sc_xz(VISC,self.interp_SC)
        self.FX.copy_( state_w.grad_x )
        self.FX.add_ ( state_u.grad_z )
        self.FX.mul_ ( self.interp_SC/rho )
        # zy
        metric.interp_sc_yz(VISC,self.interp_SC)
        self.FY.copy_( state_w.grad_y )
        self.FY.add_ ( state_v.grad_z )
        self.FY.mul_ ( self.interp_SC/rho )
        # zz
        self.FZ.copy_( state_w.grad_z )
        self.FZ.sub_ ( self.div_vel )
        self.FZ.mul_ ( 2.0*VISC/rho )
        # Add the modeled SFS flux
        if (sfsmodel.modelType=='tensor'):
            metric.interp_sc_xz(sfsmodel.FXZ,self.interp_SC)
            self.FX.add_( self.interp_SC )
            metric.interp_sc_yz(sfsmodel.FYZ,self.interp_SC)
            self.FY.add_( self.interp_SC )
            self.FZ.add_( sfsmodel.FZZ )
        # Divergence of the viscous+SFS flux
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

        
        # Source-type SFS models, including ML models
        if (sfsmodel.modelType=='source'):
            # Use interpolated variables
            #   --> ML model can be improved by using staggered derivatives internally
            metric.interp_u_xm( state_u )
            metric.interp_v_ym( state_v )
            metric.interp_w_zm( state_w )
            
            # Evaluate the model
            sfsmodel.update(state_u.var_i,state_v.var_i,state_w.var_i,metric)

            # Interpolate source terms to cell faces and accumulate to the RHS
            #  --> [JFM] should accumulation be + or -?
            # x
            metric.interp_sc_x( sfsmodel.GX[:,:,:,0], self.interp_SC )
            self.rhs_u.sub_( self.interp_SC[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )
            # y
            metric.interp_sc_y( sfsmodel.GY[:,:,:,0], self.interp_SC )
            self.rhs_v.sub_( self.interp_SC[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )
            # z
            metric.interp_sc_z( sfsmodel.GZ[:,:,:,0], self.interp_SC )
            self.rhs_w.sub_( self.interp_SC[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )
            
            # Accumulate to RHS   
            #self.rhs_u.sub_( sfsmodel.GX[imin_:imax_,jmin_:jmax_,kmin_:kmax_,0] )
            #self.rhs_v.sub_( sfsmodel.GY[imin_:imax_,jmin_:jmax_,kmin_:kmax_,0] )
            #self.rhs_w.sub_( sfsmodel.GZ[imin_:imax_,jmin_:jmax_,kmin_:kmax_,0] )
