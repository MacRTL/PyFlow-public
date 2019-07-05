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
from torch.autograd import Variable


# ----------------------------------------------------
# Adjoint RHS class
# ----------------------------------------------------
class AdjointRHS:
    def __init__(self,inputConfig,decomp,metric,rho,VISC,sfsmodel,
                 state_u,state_v,state_w):

        # Allocate RHS objects
        self.rhs1 = rhs_adjPredictor(decomp,metric,rho,VISC,sfsmodel,state_u,state_v,state_w)
        if (inputConfig.advancerName[:-1]=="RK"):
            self.rhs2 = rhs_adjPredictor(decomp,metric,rho,VISC,sfsmodel,state_u,state_v,state_w)
            self.rhs3 = rhs_adjPredictor(decomp,metric,rho,VISC,sfsmodel,state_u,state_v,state_w)
            self.rhs4 = rhs_adjPredictor(decomp,metric,rho,VISC,sfsmodel,state_u,state_v,state_w)
        

# ----------------------------------------------------
# Navier-Stokes adjoint equation RHS
# ----------------------------------------------------
class rhs_adjPredictor:
    def __init__(self,decomp,metric,rho,VISC,sfsmodel,
                 state_u,state_v,state_w):
        # Default precision and offloading settings
        prec = decomp.prec
        self.device = decomp.device

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

        # Save a few pointers
        self.metric = metric
        self.rho    = rho
        self.VISC   = VISC
        self.sfsmodel = sfsmodel
        self.state_u  = state_u
        self.state_v  = state_v
        self.state_w  = state_w

            
    # ----------------------------------------------------
    # Evaluate the adjoint equation RHS
    # ----------------------------------------------------
    def evaluate(self,state_u_adj,state_v_adj,state_w_adj):

        imin_ = self.metric.imin_; imax_ = self.metric.imax_+1
        jmin_ = self.metric.jmin_; jmax_ = self.metric.jmax_+1
        kmin_ = self.metric.kmin_; kmax_ = self.metric.kmax_+1
        
        # Zero the rhs
        self.rhs_u.zero_()
        self.rhs_v.zero_()
        self.rhs_w.zero_()
        
        #print(self.FX.device)
        #print(self.state_u.grad_x.device)

        # Compute velocity gradients for the viscous flux
        self.metric.grad_vel_visc(state_u_adj)
        self.metric.grad_vel_visc(state_v_adj)
        self.metric.grad_vel_visc(state_w_adj)

        # Compute velocity divergence for the viscous flux
        self.metric.div_vel_over(state_u_adj,state_v_adj,state_w_adj,self.div_vel)
        self.div_vel.div_( 3.0 )

        # Viscous fluxes
        # xx
        self.FX.copy_( state_u_adj.grad_x )
        self.FX.sub_ ( self.div_vel )
        self.FX.mul_ ( 2.0*self.VISC/self.rho )
        # xy
        self.metric.interp_sc_xy(self.VISC,self.interp_SC)
        self.FY.copy_( state_u_adj.grad_y )
        self.FY.add_ ( state_v_adj.grad_x )
        self.FY.mul_ ( self.interp_SC/self.rho )
        # xz
        self.metric.interp_sc_xz(self.VISC,self.interp_SC)
        self.FZ.copy_( state_u_adj.grad_z )
        self.FZ.add_ ( state_w_adj.grad_x )
        self.FZ.mul_ ( self.interp_SC/self.rho )
        # Divergence of the viscous flux
        self.metric.div_visc(self.FX,self.FY,self.FZ,self.rhs_u)
        
        # yx
        self.metric.interp_sc_xy(self.VISC,self.interp_SC)
        self.FX.copy_( state_v_adj.grad_x )
        self.FX.add_ ( state_u_adj.grad_y )
        self.FX.mul_ ( self.interp_SC/self.rho )
        # yy
        self.FY.copy_( state_v_adj.grad_y )
        self.FY.sub_ ( self.div_vel )
        self.FY.mul_ ( 2.0*self.VISC/self.rho )
        # yz
        self.metric.interp_sc_yz(self.VISC,self.interp_SC)
        self.FZ.copy_( state_v_adj.grad_z )
        self.FZ.add_ ( state_w_adj.grad_y )
        self.FZ.mul_ ( self.interp_SC/self.rho )
        # Divergence of the viscous flux
        self.metric.div_visc(self.FX,self.FY,self.FZ,self.rhs_v)
        
        # zx
        self.metric.interp_sc_xz(self.VISC,self.interp_SC)
        self.FX.copy_( state_w_adj.grad_x )
        self.FX.add_ ( state_u_adj.grad_z )
        self.FX.mul_ ( self.interp_SC/self.rho )
        # zy
        self.metric.interp_sc_yz(self.VISC,self.interp_SC)
        self.FY.copy_( state_w_adj.grad_y )
        self.FY.add_ ( state_v_adj.grad_z )
        self.FY.mul_ ( self.interp_SC/self.rho )
        # zz
        self.FZ.copy_( state_w_adj.grad_z )
        self.FZ.sub_ ( self.div_vel )
        self.FZ.mul_ ( 2.0*self.VISC/self.rho )
        # Divergence of the viscous flux
        self.metric.div_visc(self.FX,self.FY,self.FZ,self.rhs_w)

        
        # Advective fluxes -- NON-CONSERVATIVE -- from CONSERVATIVE
        # form of NS equations
        #
        # Compute gradients of the adjoint velocity field
        self.metric.grad_vel_adj_cons(state_u_adj,'u')
        self.metric.grad_vel_adj_cons(state_v_adj,'v')
        self.metric.grad_vel_adj_cons(state_w_adj,'w')
        
        # -2*grad1T(adj(u))*u
        self.FX.copy_( state_u_adj.grad_x )
        self.FX.mul_ ( self.state_u.var )
        self.FX.mul_ ( 2.0 )

        # -grad2T(adj(u))*v
        self.metric.interp_v_ym  ( self.state_v )
        self.metric.interp_uvwi_x( self.state_v )
        self.FY.copy_( state_u_adj.grad_y )
        self.FY.mul_ ( self.state_v.var_i )
        
        # -grad3T(adj(u))*w
        self.metric.interp_w_zm  ( self.state_w )
        self.metric.interp_uvwi_x( self.state_w )
        self.FZ.copy_( state_u_adj.grad_z )
        self.FZ.mul_ ( self.state_w.var_i )

        # Accumulate to adj(u) RHS, note negative signs from above
        self.rhs_u.add_( self.FX[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )
        self.rhs_u.add_( self.FY[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )
        self.rhs_u.add_( self.FZ[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )


        # -grad1T(adj(v))*u
        self.metric.interp_u_xm  ( self.state_u )
        self.metric.interp_uvwi_y( self.state_u )
        self.FX.copy_( state_v_adj.grad_x )
        self.FX.mul_ ( self.state_u.var_i )

        # -2*grad2T(adj(v))*v
        self.FY.copy_( state_v_adj.grad_y )
        self.FY.mul_ ( self.state_v.var )
        self.FY.mul_ ( 2.0 )

        # -grad3T(adj(v))*w
        self.metric.interp_w_zm  ( self.state_w )
        self.metric.interp_uvwi_y( self.state_w )
        self.FZ.copy_( state_v_adj.grad_z )
        self.FZ.mul_ ( self.state_w.var_i )

        # Accumulate to adj(v) RHS, note negative signs from above
        self.rhs_v.add_( self.FX[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )
        self.rhs_v.add_( self.FY[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )
        self.rhs_v.add_( self.FZ[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )


        # -grad1T(adj(w))*u
        self.metric.interp_u_xm  ( self.state_u )
        self.metric.interp_uvwi_z( self.state_u )
        self.FX.copy_( state_w_adj.grad_x )
        self.FX.mul_ ( self.state_u.var_i )

        # -grad2T(adj(w))*v
        self.metric.interp_v_ym  ( self.state_v )
        self.metric.interp_uvwi_z( self.state_v )
        self.FY.copy_( state_w_adj.grad_y )
        self.FY.mul_ ( self.state_v.var_i )

        # -2*grad3T(adj(w))*w
        self.FZ.copy_( state_w_adj.grad_z )
        self.FZ.mul_ ( self.state_w.var )
        self.FZ.mul_ ( 2.0 )

        # Accumulate to adj(w) RHS, note negative signs from above
        self.rhs_w.add_( self.FX[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )
        self.rhs_w.add_( self.FY[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )
        self.rhs_w.add_( self.FZ[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )
        
        
        #print("Done adj visc")
        #return

    def dont_use(self):
        print("oops")

        
        # Cross-advective fluxes -- divergence form
        #   ---> from NON-CONSERVATIVE form of NS equations; not appropriate??
        # -adj(u)*u
        self.metric.interp_u_xm(state_u_adj)
        self.metric.interp_u_xm(self.state_u)
        self.metric.vel_conv_xx(state_u_adj,self.state_u,self.rhs_u,sign=-1.0)
        # -adj(u)*v
        self.metric.interp_uw_y(state_u_adj)
        self.metric.interp_vw_x(self.state_v)
        self.metric.vel_conv_y(state_u_adj,self.state_v,self.rhs_u,sign=-1.0)
        # -adj(u)*w
        self.metric.interp_uv_z(state_u_adj)
        self.metric.interp_vw_x(self.state_w)
        self.metric.vel_conv_z(state_u_adj,self.state_w,self.rhs_u,sign=-1.0)
        
        # -adj(v)*u
        self.metric.interp_vw_x(state_v_adj)
        self.metric.interp_uw_y(self.state_u)
        self.metric.vel_conv_x(state_v_adj,self.state_u,self.rhs_v,sign=-1.0)
        # -adj(v)*v
        self.metric.interp_v_ym(state_v_adj)
        self.metric.interp_v_ym(self.state_v)
        self.metric.vel_conv_yy(state_v_adj,self.state_v,self.rhs_v,sign=-1.0)
        # -adj(v)*w
        self.metric.interp_uv_z(state_v_adj)
        self.metric.interp_uw_y(self.state_w)
        self.metric.vel_conv_z(state_v_adj,self.state_w,self.rhs_v,sign=-1.0)
        
        # -adj(w)*u
        self.metric.interp_vw_x(state_w_adj)
        self.metric.interp_uv_z(self.state_u)
        self.metric.vel_conv_x(state_w_adj,self.state_u,self.rhs_w,sign=-1.0)
        # -adj(w)*v
        self.metric.interp_uw_y(state_w_adj)
        self.metric.interp_uv_z(self.state_v)
        self.metric.vel_conv_y(state_w_adj,self.state_v,self.rhs_w,sign=-1.0)
        # -adj(w)*w
        self.metric.interp_w_zm(state_w_adj)
        self.metric.interp_w_zm(self.state_w)
        self.metric.vel_conv_zz(state_w_adj,self.state_w,self.rhs_w,sign=-1.0)


        # Cross-advective fluxes
        #   [JFM] Do these terms have physical meaning?
        # Compute gradients of non-adjoint velocity field
        self.metric.grad_vel_adj(self.state_u,'u')
        self.metric.grad_vel_adj(self.state_v,'v')
        self.metric.grad_vel_adj(self.state_w,'w')
        
        # adj(u) equation - compute at x-face
        # -adj(u)*grad1(u)
        self.FX.copy_( self.state_u.grad_x )
        self.FX.mul_ ( state_u_adj.var )
        
        # -adj(v)*grad1(v) - compute at xy-edge
        # Interpolate v-adj to x-face
        self.metric.interp_v_ym  (state_v_adj)
        self.metric.interp_uvwi_x(state_v_adj)
        #state_v_adj.update_border_i()
        self.div_vel.copy_( self.state_v.grad_x )
        self.FY.copy_     ( self.state_v.grad_x )
        self.FY[:,:-1,:].add_( self.div_vel[:,1:,:] )
        self.FY.mul_( self.metric.interp_ym )
        self.FY.mul_( state_v_adj.var_i )
        
        # -adj(w)*grad1(w) - compute at xz-edge, interpolate to x-face
        self.metric.interp_w_zm  (state_w_adj)
        self.metric.interp_uvwi_x(state_w_adj)
        #state_w_adj.update_border_i()
        self.div_vel.copy_( self.state_w.grad_x )
        self.FZ.copy_     ( self.state_w.grad_x )
        self.FZ[:,:,:-1].add_( self.div_vel[:,:,1:] )
        self.FZ.mul_( self.metric.interp_zm )
        self.FZ.mul_( state_w_adj.var_i )

        # Accumulate to adj(u) RHS
        self.rhs_u.sub_( self.FX[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )
        self.rhs_u.sub_( self.FY[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )
        self.rhs_u.sub_( self.FZ[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )

        
        # adj(v) equation
        # -adj(u)*grad2(u)
        self.metric.interp_u_xm  (state_u_adj)
        self.metric.interp_uvwi_y(state_u_adj)
        #state_u_adj.update_border_i()
        self.div_vel.copy_( self.state_u.grad_y )
        self.FX.copy_     ( self.state_u.grad_y )
        self.FX[:-1,:,:].add_( self.div_vel[1:,:,:] )
        self.FX.mul_( self.metric.interp_xm )
        self.FX.mul_( state_u_adj.var_i )
        
        # -adj(v)*grad2(v)
        self.FY.copy_( self.state_v.grad_y )
        self.FY.mul_ ( state_v_adj.var )

        # -adj(w)*grad2(w)
        self.metric.interp_w_zm  (state_w_adj)
        self.metric.interp_uvwi_y(state_w_adj)
        #state_w_adj.update_border_i()
        self.div_vel.copy_( self.state_w.grad_y )
        self.FZ.copy_     ( self.state_w.grad_y )
        self.FZ[:,:,:-1].add_( self.div_vel[:,:,1:] )
        self.FZ.mul_( self.metric.interp_zm )
        self.FZ.mul_( state_w_adj.var_i )

        # Accumulate to adj(v) RHS
        self.rhs_v.sub_( self.FX[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )
        self.rhs_v.sub_( self.FY[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )
        self.rhs_v.sub_( self.FZ[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )

        
        # adj(w) equation
        # -adj(u)*grad3(u)
        self.metric.interp_u_xm  (state_u_adj)
        self.metric.interp_uvwi_z(state_u_adj)
        #state_u_adj.update_border_i()
        self.div_vel.copy_( self.state_u.grad_z )
        self.FX.copy_     ( self.state_u.grad_z )
        self.FX[:-1,:,:].add_( self.div_vel[1:,:,:] )
        self.FX.mul_( self.metric.interp_xm )
        self.FX.mul_( state_u_adj.var_i )

        # -adj(v)*grad3(v)
        self.metric.interp_v_ym  (state_v_adj)
        self.metric.interp_uvwi_z(state_v_adj)
        #state_v_adj.update_border_i()
        self.div_vel.copy_( self.state_v.grad_z )
        self.FY.copy_     ( self.state_v.grad_z )
        self.FY[:,:-1,:].add_( self.div_vel[:,1:,:] )
        self.FY.mul_( self.metric.interp_ym )
        self.FY.mul_( state_v_adj.var_i )

        # -adj(w)*grad3(w)
        self.FZ.copy_( self.state_w.grad_z )
        self.FZ.mul_ ( state_w_adj.var )

        # Accumulate to adj(w) RHS
        self.rhs_w.sub_( self.FX[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )
        self.rhs_w.sub_( self.FY[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )
        self.rhs_w.sub_( self.FZ[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )
        
        
        

        # Closure model terms -- delta^t

        # Use interpolated variables
        #   --> ML model can be improved by using staggered derivatives internally
        self.metric.interp_u_xm( self.state_u )
        self.metric.interp_v_ym( self.state_v )
        self.metric.interp_w_zm( self.state_w )

        # Enable computational graph generation
        #   --> Can memory management be improved by pre-allocating?
        u_V = Variable( self.state_u.var_i, requires_grad=True ).to(self.device)
        v_V = Variable( self.state_v.var_i, requires_grad=True ).to(self.device)
        w_V = Variable( self.state_w.var_i, requires_grad=True ).to(self.device)
        
        # Evaluate the SFS model using PyTorch automatic differentiation
        # Gradients in model inputs need to be computed using non-in-place operations
        self.sfsmodel.update(u_V,v_V,w_V,requires_grad=True)
        
        # Treat the adjoint as a constant when calculating the chain rule
        self.metric.interp_u_xm( state_u_adj )
        self.metric.interp_v_ym( state_v_adj )
        self.metric.interp_w_zm( state_w_adj )
        # Pre-allocate u_A_det, etc?
        u_A_det = Variable(state_u_adj.var_i).detach()
        v_A_det = Variable(state_v_adj.var_i).detach()
        w_A_det = Variable(state_w_adj.var_i).detach()

        #print(u_A_det.size())
        
        # Compute g
        g = ( torch.sum(self.sfsmodel.GX[:,:,:,0]*u_A_det) +
              torch.sum(self.sfsmodel.GY[:,:,:,0]*v_A_det) +
              torch.sum(self.sfsmodel.GZ[:,:,:,0]*w_A_det) )

        #print(g.size())
        
        # Compute the gradient of g wrt. u
        g.backward()

        # Compute \delta = \partial{ u_{i,A}*g^i }/\partial{ u_i }
        grad_ML1 = u_V.grad.data.type(torch.FloatTensor).detach()
        grad_ML2 = v_V.grad.data.type(torch.FloatTensor).detach()
        grad_ML3 = w_V.grad.data.type(torch.FloatTensor).detach()
        
        #print(grad_ML1.size())

        # Interpolate source terms to cell faces and accumulate to the RHS
        # x
        self.metric.interp_sc_x( grad_ML1, self.interp_SC )
        self.rhs_u.sub_( self.interp_SC[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )
        # y
        self.metric.interp_sc_y( grad_ML2, self.interp_SC )
        self.rhs_v.sub_( self.interp_SC[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )
        # z
        self.metric.interp_sc_z( grad_ML3, self.interp_SC )
        self.rhs_w.sub_( self.interp_SC[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )

        # Accumulate the AD gradient to the adjoint equation RHS
        #self.rhs_u.sub_( self.sfsmodel.GX[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )
        #self.rhs_v.sub_( self.sfsmodel.GY[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )
        #self.rhs_w.sub_( self.sfsmodel.GZ[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )

        # Clean up
        del g
        del u_V
        del v_V
        del w_V
        del u_A_det
        del v_A_det
        del w_A_det
        del grad_ML1
        del grad_ML2
        del grad_ML3
