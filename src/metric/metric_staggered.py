# ------------------------------------------------------------------------
#
# PyFlow: A GPU-accelerated CFD platform written in Python
#
# @file metric_staggered.py
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

# ------------------------------------------------------
# Staggered central-difference schemes for UNIFORM grids
# ------------------------------------------------------
class metric_uniform:
    def __init__(self,geo):

        # Local data sizes and indices
        nxo_ = geo.nxo_
        nyo_ = geo.nyo_
        nzo_ = geo.nzo_
        self.imin_  = geo.imin_;  self.imax_ = geo.imax_
        self.jmin_  = geo.jmin_;  self.jmax_ = geo.jmax_
        self.kmin_  = geo.kmin_;  self.kmax_ = geo.kmax_
        self.imino_ = geo.imino_; self.imaxo_= geo.imaxo_
        self.jmino_ = geo.jmino_; self.jmaxo_= geo.jmaxo_
        self.kmino_ = geo.kmino_; self.kmaxo_= geo.kmaxo_
        
        # Initialize the grid metrics
        self.grid_geo = geo.type
        if (self.grid_geo=='uniform'):
            self.div_x  = 1.0*geo.dxi
            self.div_y  = 1.0*geo.dyi
            self.div_z  = 1.0*geo.dzi
            self.grad_x  = 1.0*geo.dxi
            self.grad_y  = 1.0*geo.dyi
            self.grad_z  = 1.0*geo.dzi
            self.grad_xm = 1.0*geo.dxi
            self.grad_ym = 1.0*geo.dyi
            self.grad_zm = 1.0*geo.dzi
            self.interp_x  = 0.5
            self.interp_y  = 0.5
            self.interp_z  = 0.5
            self.interp_xm = 0.5
            self.interp_ym = 0.5
            self.interp_zm = 0.5
        else:
            print("grid type not implemented")


    # ----------------------------------------------------
    # Interpolation of a cell-centered variable to x-faces
    def interp_sc_x(self,state):
        imin_ = self.imin_; imax_ = self.imax_+1
        jmin_ = self.jmin_; jmax_ = self.jmax_+1
        kmin_ = self.kmin_; kmax_ = self.kmax_+1
        
        state.var_i[:-1,:,:] = self.interp_x*(state.var[1:,:,:] + state.var[:-1,:,:])

        # Periodic boundary
        state.var_i[-1,:,:] = self.interp_x*(state.var[0,:,:] + state.var[-1,:,:])

    # ----------------------------------------------------
    # Interpolation of a cell-centered variable to y-faces
    def interp_sc_y(self,state):
        imin_ = self.imin_; imax_ = self.imax_+1
        jmin_ = self.jmin_; jmax_ = self.jmax_+1
        kmin_ = self.kmin_; kmax_ = self.kmax_+1
        
        state.var_i[:,:-1,:] = self.interp_y*(state.var[:,1:,:] + state.var[:,:-1,:])

        # Periodic boundary
        state.var_i[:,-1,:] = self.interp_y*(state.var[:,0,:] + state.var[:,-1,:])

    # ----------------------------------------------------
    # Interpolation of a cell-centered variable to z-faces
    def interp_sc_z(self,state):
        imin_ = self.imin_; imax_ = self.imax_+1
        jmin_ = self.jmin_; jmax_ = self.jmax_+1
        kmin_ = self.kmin_; kmax_ = self.kmax_+1
        
        state.var_i[:,:,:-1] = self.interp_z*(state.var[:,:,1:] + state.var[:,:,:-1])

        # Periodic boundary
        state.var_i[:,:,-1] = self.interp_z*(state.var[:,:,0] + state.var[:,:,-1])


    # -----------------------------------------------
    # Interpolation of the x-velocity to cell centers
    def interp_u_xm(self,state):
        imin_ = self.imin_; imax_ = self.imax_+1
        jmin_ = self.jmin_; jmax_ = self.jmax_+1
        kmin_ = self.kmin_; kmax_ = self.kmax_+1
        
        state.var_i = self.interp_xm*( state.var[imin_  :imax_+1,jmin_:jmax_+1,kmin_:kmax_+1] +
                                       state.var[imin_-1:imax_  ,jmin_:jmax_+1,kmin_:kmax_+1] )

    # -----------------------------------------------
    # Interpolation of the y-velocity to cell centers
    def interp_v_ym(self,state):
        imin_ = self.imin_; imax_ = self.imax_+1
        jmin_ = self.jmin_; jmax_ = self.jmax_+1
        kmin_ = self.kmin_; kmax_ = self.kmax_+1
        
        state.var_i = self.interp_ym*( state.var[imin_:imax_+1,jmin_  :jmax_+1,kmin_:kmax_+1] +
                                       state.var[imin_:imax_+1,jmin_-1:jmax_  ,kmin_:kmax_+1] )

    # -----------------------------------------------
    # Interpolation of the z-velocity to cell centers
    def interp_w_zm(self,state):
        imin_ = self.imin_; imax_ = self.imax_+1
        jmin_ = self.jmin_; jmax_ = self.jmax_+1
        kmin_ = self.kmin_; kmax_ = self.kmax_+1
        
        state.var_i = self.interp_zm*( state.var[imin_:imax_+1,jmin_:jmax_+1,kmin_  :kmax_+1] +
                                       state.var[imin_:imax_+1,jmin_:jmax_+1,kmin_-1:kmax_  ] )

        
    # ---------------------------------------------------
    # Interpolation of the v- and w-velocities to x-edges
    def interp_vw_x(self,state):
        imin_ = self.imin_; imax_ = self.imax_+1
        jmin_ = self.jmin_; jmax_ = self.jmax_+1
        kmin_ = self.kmin_; kmax_ = self.kmax_+1
        
        state.var_i = self.interp_x*( state.var[imin_  :imax_+1,jmin_:jmax_+1,kmin_:kmax_+1] +
                                      state.var[imin_-1:imax_  ,jmin_:jmax_+1,kmin_:kmax_+1] )
        
    # ---------------------------------------------------
    # Interpolation of the u- and w-velocities to y-edges
    def interp_uw_y(self,state):
        imin_ = self.imin_; imax_ = self.imax_+1
        jmin_ = self.jmin_; jmax_ = self.jmax_+1
        kmin_ = self.kmin_; kmax_ = self.kmax_+1
        
        state.var_i = self.interp_y*( state.var[imin_:imax_+1,jmin_  :jmax_+1,kmin_:kmax_+1] +
                                      state.var[imin_:imax_+1,jmin_-1:jmax_  ,kmin_:kmax_+1] )

    # ---------------------------------------------------
    # Interpolation of the u- and v-velocities to z-edges
    def interp_uv_z(self,state):
        imin_ = self.imin_; imax_ = self.imax_+1
        jmin_ = self.jmin_; jmax_ = self.jmax_+1
        kmin_ = self.kmin_; kmax_ = self.kmax_+1
        
        state.var_i = self.interp_z*( state.var[imin_:imax_+1,jmin_:jmax_+1,kmin_  :kmax_+1] +
                                      state.var[imin_:imax_+1,jmin_:jmax_+1,kmin_-1:kmax_  ] )


    # ----------------------------------------------------------------
    # Gradient of the pressure to the cell faces (pressure correction)
    def grad_P(self,state):
        imin_ = self.imin_; imax_ = self.imax_+1
        jmin_ = self.jmin_; jmax_ = self.jmax_+1
        kmin_ = self.kmin_; kmax_ = self.kmax_+1
        
        state.grad_x = self.grad_x*( state.var[imin_  :imax_+1,jmin_:jmax_+1,kmin_:kmax_+1] -
                                     state.var[imin_-1:imax_  ,jmin_:jmax_+1,kmin_:kmax_+1] )
        state.grad_y = self.grad_y*( state.var[imin_:imax_+1,jmin_  :jmax_+1,kmin_:kmax_+1] -
                                     state.var[imin_:imax_+1,jmin_-1:jmax_  ,kmin_:kmax_+1] )
        state.grad_z = self.grad_z*( state.var[imin_:imax_+1,jmin_:jmax_+1,kmin_  :kmax_+1] -
                                     state.var[imin_:imax_+1,jmin_:jmax_+1,kmin_-1:kmax_  ] )


    # ------------------------------------------------------------
    # Gradient of velocity to locations needed by the viscous flux
    def grad_vel_visc(self,state):
        imin_ = self.imin_; imax_ = self.imax_+1
        jmin_ = self.jmin_; jmax_ = self.jmax_+1
        kmin_ = self.kmin_; kmax_ = self.kmax_+1
        
        state.grad_x = self.grad_x*( state.var[imin_  :imax_+1,jmin_:jmax_+1,kmin_:kmax_+1] -
                                     state.var[imin_-1:imax_  ,jmin_:jmax_+1,kmin_:kmax_+1] )
        state.grad_y = self.grad_y*( state.var[imin_:imax_+1,jmin_  :jmax_+1,kmin_:kmax_+1] -
                                     state.var[imin_:imax_+1,jmin_-1:jmax_  ,kmin_:kmax_+1] )
        state.grad_z = self.grad_z*( state.var[imin_:imax_+1,jmin_:jmax_+1,kmin_  :kmax_+1] -
                                     state.var[imin_:imax_+1,jmin_:jmax_+1,kmin_-1:kmax_  ] )


    # -------------------------------------------------
    # Divergence of the viscous fluxes
    def div_visc(self,FX,FY,FZ,rhs):
        rhs += ( FX[1:,:-1,:-1] - FX[:-1,:-1,:-1] )*self.div_x
        rhs += ( FY[:-1,1:,:-1] - FY[:-1,:-1,:-1] )*self.div_y
        rhs += ( FZ[:-1,:-1,1:] - FZ[:-1,:-1,:-1] )*self.div_z

        
    # -------------------------------------------------
    # Divergence of the velocity field at cell centers
    def div_vel(self,state_u,state_v,state_w,divg):
        imin_ = self.imin_; imax_ = self.imax_+1
        jmin_ = self.jmin_; jmax_ = self.jmax_+1
        kmin_ = self.kmin_; kmax_ = self.kmax_+1
        
        # Zero the divergence
        divg.zero_()
        
        divg += ( state_u.var[imin_+1:imax_+1,jmin_:jmax_,kmin_:kmax_] -
                  state_u.var[imin_  :imax_  ,jmin_:jmax_,kmin_:kmax_] )*self.grad_x
        divg += ( state_u.var[imin_:imax_,jmin_+1:jmax_+1,kmin_:kmax_] -
                  state_u.var[imin_:imax_,jmin_  :jmax_  ,kmin_:kmax_] )*self.grad_y
        divg += ( state_u.var[imin_:imax_,jmin_:jmax_,kmin_+1:kmax_+1] -
                  state_u.var[imin_:imax_,jmin_:jmax_,kmin_  :kmax_  ] )*self.grad_z


    # -------------------------------------------------
    # Convective flux xx
    def vel_conv_xx(self,state_u,state_v,grad_x):
        grad_x -= ( state_u.var_i[1: ,:-1,:-1] * state_v.var_i[1: ,:-1,:-1] -
                    state_u.var_i[:-1,:-1,:-1] * state_v.var_i[:-1,:-1,:-1] )*self.grad_x
        
    # -------------------------------------------------
    # Convective flux yy
    def vel_conv_yy(self,state_u,state_v,grad_y):
        grad_y -= ( state_u.var_i[:-1,1: ,:-1] * state_v.var_i[:-1,1: ,:-1] -
                    state_u.var_i[:-1,:-1,:-1] * state_v.var_i[:-1,:-1,:-1] )*self.grad_y
        
    # -------------------------------------------------
    # Convective flux zz
    def vel_conv_zz(self,state_u,state_w,grad_z):
        grad_z -= ( state_u.var_i[:-1,:-1,1: ] * state_w.var_i[:-1,:-1,1: ] -
                    state_u.var_i[:-1,:-1,:-1] * state_w.var_i[:-1,:-1,:-1] )*self.grad_z
        
    # -------------------------------------------------
    # Off-diagonal convective flux x
    def vel_conv_x(self,state_u,state_v,grad_x):
        grad_x -= ( state_u.var_i[1: ,:-1,:-1] * state_v.var_i[1: ,:-1,:-1] -
                    state_u.var_i[:-1,:-1,:-1] * state_v.var_i[:-1,:-1,:-1] )*self.grad_x
        
    # -------------------------------------------------
    # Off-diagonal convective flux y
    def vel_conv_y(self,state_u,state_v,grad_y):
        grad_y -= ( state_u.var_i[:-1,1: ,:-1] * state_v.var_i[:-1,1: ,:-1] -
                    state_u.var_i[:-1,:-1,:-1] * state_v.var_i[:-1,:-1,:-1] )*self.grad_y
        
    # -------------------------------------------------
    # Off-diagonal convective flux z
    def vel_conv_z(self,state_u,state_w,grad_z):
        grad_z -= ( state_u.var_i[:-1,:-1,1: ] * state_w.var_i[:-1,:-1,1: ] -
                    state_u.var_i[:-1,:-1,:-1] * state_w.var_i[:-1,:-1,:-1] )*self.grad_z





        

def deprecated():
    

    # ------------------------------------------------------------
    # Gradient of velocity to locations needed by the viscous flux
    def grad_vel_visc(self,state):
        state.grad_x[1:,:,:] = self.grad_x*(state.var[1:,:,:] - state.var[:-1,:,:])
        state.grad_y[:,1:,:] = self.grad_y*(state.var[:,1:,:] - state.var[:,:-1,:])
        state.grad_z[:,:,1:] = self.grad_z*(state.var[:,:,1:] - state.var[:,:,:-1])
    
        # Periodic boundary conditions
        # x
        state.grad_x[0,:,:] = self.grad_x*(state.var[0,:,:] - state.var[-1,:,:])
        # y
        state.grad_y[:,0,:] = self.grad_y*(state.var[:,0,:] - state.var[:,-1,:])
        # z
        state.grad_z[:,:,0] = self.grad_z*(state.var[:,:,0] - state.var[:,:,-1])


    # -------------------------------------------------
    # Divergence of the viscous fluxes
    def div_visc(self,FX,FY,FZ,rhs_u):
        rhs_u[:-1,:,:] += (FX[1:,:,:] - FX[:-1,:,:])*self.div_x
        rhs_u[:,:-1,:] += (FY[:,1:,:] - FY[:,:-1,:])*self.div_y
        rhs_u[:,:,:-1] += (FZ[:,:,1:] - FZ[:,:,:-1])*self.div_z
    
        # Periodic boundary conditions
        rhs_u[-1,:,:] += (FX[0,:,:] - FX[-1,:,:])*self.div_x
        rhs_u[:,-1,:] += (FY[:,0,:] - FY[:,-1,:])*self.div_y
        rhs_u[:,:,-1] += (FZ[:,:,0] - FZ[:,:,-1])*self.div_z
