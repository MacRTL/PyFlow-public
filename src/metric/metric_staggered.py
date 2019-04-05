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
from scipy.sparse import diags

# ------------------------------------------------------
# Staggered central-difference schemes for uniform grids
# ------------------------------------------------------
class metric_uniform:
    def __init__(self,geo):

        # Local data sizes and indices
        nx_ = geo.nx_
        ny_ = geo.ny_
        nz_ = geo.nz_
        self.imin_  = geo.imin_;  self.imax_ = geo.imax_
        self.jmin_  = geo.jmin_;  self.jmax_ = geo.jmax_
        self.kmin_  = geo.kmin_;  self.kmax_ = geo.kmax_
        nxo_ = geo.nxo_
        nyo_ = geo.nyo_
        nzo_ = geo.nzo_
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

            # Laplace operator
            N = nx_*ny_*nz_
            dxi = geo.dxi; dyi = geo.dyi; dzi = geo.dzi
            # i,j,k; i-1,j,k; i+1,j,k; i,j-1,k; i,j+1,k; i,j,k-1; i,j,k+1
            diagonals = [ -2*dxi*dxi - 2*dyi*dyi - 2*dzi*dzi,
                          dxi*dxi, dxi*dxi,
                          dyi*dyi, dyi*dyi,
                          dzi*dzi, dzi*dzi ]
            offsets = [0, -nx_*ny_, nx_*ny_, -nx_, nx_, -1, +1]
            self.Laplace = diags(diagonals, offsets, shape=(N,N), dtype=np.float32)

            # MPI: update Laplace operator border
            
        else:
            print("grid type not implemented")


    # -----------------------------------------------
    # Interpolation of a scalar to xy-edges
    def interp_sc_xy(self,SC,interp_SC):
        interp_SC[1:,1:,:].copy_(self.interp_x*self.interp_y*( SC[1:,1: ,:] + SC[:-1,1: ,:] +
                                                               SC[1:,:-1,:] + SC[:-1,:-1,:] ))
        
    # -----------------------------------------------
    # Interpolation of a scalar to xz-edges
    def interp_sc_xz(self,SC,interp_SC):
        interp_SC[1:,:,1:].copy_(self.interp_x*self.interp_z*( SC[1:,:,1: ] + SC[:-1,:,1: ] +
                                                               SC[1:,:,:-1] + SC[:-1,:,:-1] ))
        
    # -----------------------------------------------
    # Interpolation of a scalar to yz-edges
    def interp_sc_yz(self,SC,interp_SC):
        interp_SC[:,1:,1:].copy_(self.interp_y*self.interp_z*( SC[:,1:,1: ] + SC[:,:-1,1: ] +
                                                               SC[:,1:,:-1] + SC[:,:-1,:-1] ))


    # -----------------------------------------------
    # Interpolation of the x-velocity to cell centers
    def interp_u_xm(self,state):
        state.var_i[:-1,:,:] = 0.5*( state.var[1:,:,:] + state.var[:-1,:,:] )
        #state.update_border_i()

    # -----------------------------------------------
    # Interpolation of the y-velocity to cell centers
    def interp_v_ym(self,state):
        state.var_i[:,:-1,:] = 0.5*( state.var[:,1:,:] + state.var[:,:-1,:] )
        #state.update_border_i()

    # -----------------------------------------------
    # Interpolation of the z-velocity to cell centers
    def interp_w_zm(self,state):
        state.var_i[:,:,:-1] = 0.5*( state.var[:,:,1:] + state.var[:,:,:-1] )
        #state.update_border_i()

        
    # ---------------------------------------------------
    # Interpolation of the v- and w-velocities to x-edges
    def interp_vw_x(self,state):
        state.var_i[1:,:,:] = self.interp_x*( state.var[1:,:,:] + state.var[:-1,:,:] )
        #state.update_border_i()
        
    # ---------------------------------------------------
    # Interpolation of the u- and w-velocities to y-edges
    def interp_uw_y(self,state):
        state.var_i[:,1:,:] = self.interp_y*( state.var[:,1:,:] + state.var[:,:-1,:] )
        #state.update_border_i()

    # ---------------------------------------------------
    # Interpolation of the u- and v-velocities to z-edges
    def interp_uv_z(self,state):
        state.var_i[:,:,1:] = self.interp_z*( state.var[:,:,1:] + state.var[:,:,:-1] )
        #state.update_border_i()

        
    # ----------------------------------------------------
    # Interpolation of cell-centered velocities to x-faces
    def interp_uvwi_x(self,state):
        state.var_i[1:,:,:] = self.interp_x*( state.var_i[1:,:,:] + state.var_i[:-1,:,:] )
        #state.update_border_i()
        
    # ----------------------------------------------------
    # Interpolation of cell-centered velocities to y-faces
    def interp_uvwi_y(self,state):
        state.var_i[:,1:,:] = self.interp_y*( state.var_i[:,1:,:] + state.var_i[:,:-1,:] )
        #state.update_border_i()
        
    # ----------------------------------------------------
    # Interpolation of cell-centered velocities to z-faces
    def interp_uvwi_z(self,state):
        state.var_i[:,:,1:] = self.interp_z*( state.var_i[:,:,1:] + state.var_i[:,:,:-1] )
        #state.update_border_i()


    # ------------------------------------------------------------
    # Gradient of velocity at cell centers
    def grad_vel_center(self,state,comp):
        if (comp=='u'):
            # x
            state.grad_x[:-1,:,:].copy_( state.var[1: ,:,:] )
            state.grad_x[:-1,:,:].sub_ ( state.var[:-1,:,:] )
            state.grad_x[:-1,:,:].mul_ ( self.grad_x )
            # y
            self.interp_u_xm(state)
            self.interp_uvwi_y(state)
            state.grad_y[:,:-1,:].copy_( state.var_i[:,1: ,:] )
            state.grad_y[:,:-1,:].sub_ ( state.var_i[:,:-1,:] )
            state.grad_y[:,:-1,:].mul_ ( self.grad_y )
            # z
            self.interp_u_xm(state)
            self.interp_uvwi_z(state)
            state.grad_z[:,:,:-1].copy_( state.var_i[:,:,1: ] )
            state.grad_z[:,:,:-1].sub_ ( state.var_i[:,:,:-1] )
            state.grad_z[:,:,:-1].mul_ ( self.grad_z )
            
        elif (comp=='v'):
            # x
            self.interp_v_ym(state)
            self.interp_uvwi_x(state)
            state.grad_x[:-1,:,:].copy_( state.var_i[1: ,:,:] )
            state.grad_x[:-1,:,:].sub_ ( state.var_i[:-1,:,:] )
            state.grad_x[:-1,:,:].mul_ ( self.grad_x )
            # y
            state.grad_y[:,:-1,:].copy_( state.var[:,1: ,:] )
            state.grad_y[:,:-1,:].sub_ ( state.var[:,:-1,:] )
            state.grad_y[:,:-1,:].mul_ ( self.grad_y )
            # z
            self.interp_v_ym(state)
            self.interp_uvwi_z(state)
            state.grad_z[:,:,:-1].copy_( state.var_i[:,:,1: ] )
            state.grad_z[:,:,:-1].sub_ ( state.var_i[:,:,:-1] )
            state.grad_z[:,:,:-1].mul_ ( self.grad_z )

        elif (comp=='w'):
            # x
            self.interp_w_zm(state)
            self.interp_uvwi_x(state)
            state.grad_x[:-1,:,:].copy_( state.var_i[1: ,:,:] )
            state.grad_x[:-1,:,:].sub_ ( state.var_i[:-1,:,:] )
            state.grad_x[:-1,:,:].mul_ ( self.grad_x )
            # y
            self.interp_w_zm(state)
            self.interp_uvwi_y(state)
            state.grad_y[:,:-1,:].copy_( state.var_i[:,1: ,:] )
            state.grad_y[:,:-1,:].sub_ ( state.var_i[:,:-1,:] )
            state.grad_y[:,:-1,:].mul_ ( self.grad_y )
            # z
            state.grad_z[:,:,:-1].copy_( state.var[:,:,1: ] )
            state.grad_z[:,:,:-1].sub_ ( state.var[:,:,:-1] )
            state.grad_z[:,:,:-1].mul_ ( self.grad_z )


    # -------------------------------------------------------------
    # Gradient of velocity at locations needed for the viscous flux
    #  --> This method only works for uniform grids
    #  --> On-diagonal gradients computed at cell centers
    #  --> Off-diagonal gradients computed at +1/2 edge
    def grad_vel_visc(self,state):
        state.grad_x[:-1,:,:].copy_( state.var[1: ,:,:] )
        state.grad_x[:-1,:,:].sub_ ( state.var[:-1,:,:] )
        state.grad_x[:-1,:,:].mul_ ( self.grad_x )

        state.grad_y[:,:-1,:].copy_( state.var[:,1: ,:] )
        state.grad_y[:,:-1,:].sub_ ( state.var[:,:-1,:] )
        state.grad_y[:,:-1,:].mul_ ( self.grad_y )

        state.grad_z[:,:,:-1].copy_( state.var[:,:,1: ] )
        state.grad_z[:,:,:-1].sub_ ( state.var[:,:,:-1] )
        state.grad_z[:,:,:-1].mul_ ( self.grad_z )


    # ----------------------------------------------------------------
    # Gradient of the pressure to the cell faces (pressure correction)
    #  --> This method only works for uniform grids
    def grad_P(self,state):
        state.grad_x[1:,:,:].copy_( state.var[1: ,:,:] )
        state.grad_x[1:,:,:].sub_ ( state.var[:-1,:,:] )
        state.grad_x[1:,:,:].mul_ ( self.grad_x )

        state.grad_y[:,1:,:].copy_( state.var[:,1: ,:] )
        state.grad_y[:,1:,:].sub_ ( state.var[:,:-1,:] )
        state.grad_y[:,1:,:].mul_ ( self.grad_y )

        state.grad_z[:,:,1:].copy_( state.var[:,:,1: ] )
        state.grad_z[:,:,1:].sub_ ( state.var[:,:,:-1] )
        state.grad_z[:,:,1:].mul_ ( self.grad_z )


    # -------------------------------------------------
    # Divergence of the viscous flux
    def div_visc(self,FX,FY,FZ,rhs):
        imin_ = self.imin_; imax_ = self.imax_+1
        jmin_ = self.jmin_; jmax_ = self.jmax_+1
        kmin_ = self.kmin_; kmax_ = self.kmax_+1
        
        rhs.add_( self.div_x, FX[imin_  :imax_  ,jmin_:jmax_,kmin_:kmax_] )
        rhs.sub_( self.div_x, FX[imin_-1:imax_-1,jmin_:jmax_,kmin_:kmax_] )
        
        rhs.add_( self.div_y, FY[imin_:imax_,jmin_  :jmax_  ,kmin_:kmax_] )
        rhs.sub_( self.div_y, FY[imin_:imax_,jmin_-1:jmax_-1,kmin_:kmax_] )
        
        rhs.add_( self.div_z, FZ[imin_:imax_,jmin_:jmax_,kmin_  :kmax_  ] )
        rhs.sub_( self.div_z, FZ[imin_:imax_,jmin_:jmax_,kmin_-1:kmax_-1] )

        
    # -------------------------------------------------
    # Divergence of the velocity field at cell centers
    #   including overlap cells - needed for viscous fluxes
    def div_vel_over(self,state_u,state_v,state_w,divg_o):
        # Zero the divergence
        divg_o.zero_()
        
        divg_o[:-1,:,:].add_( state_u.var[1: ,:,:] )
        divg_o[:-1,:,:].sub_( state_u.var[:-1,:,:] )
        divg_o[:-1,:,:].mul_( self.grad_x )
                  
        divg_o[:,:-1,:] += ( state_v.var[:,1: ,:] - state_v.var[:,:-1,:] )*self.grad_y
        divg_o[:,:,:-1] += ( state_w.var[:,:,1: ] - state_w.var[:,:,:-1] )*self.grad_z

        
    # -------------------------------------------------
    # Divergence of the velocity field at cell centers
    def div_vel(self,state_u,state_v,state_w,divg):
        imin_ = self.imin_; imax_ = self.imax_
        jmin_ = self.jmin_; jmax_ = self.jmax_
        kmin_ = self.kmin_; kmax_ = self.kmax_
        
        # Zero the divergence
        divg.zero_()
        
        divg.add_( state_u.var[imin_+1:imax_+2,jmin_:jmax_+1,kmin_:kmax_+1] )
        divg.sub_( state_u.var[imin_  :imax_+1,jmin_:jmax_+1,kmin_:kmax_+1] )
        divg.mul_( self.grad_x )
                  
        divg += ( state_v.var[imin_:imax_+1,jmin_+1:jmax_+2,kmin_:kmax_+1] -
                  state_v.var[imin_:imax_+1,jmin_  :jmax_+1,kmin_:kmax_+1] )*self.grad_y
        divg += ( state_w.var[imin_:imax_+1,jmin_:jmax_+1,kmin_+1:kmax_+2] -
                  state_w.var[imin_:imax_+1,jmin_:jmax_+1,kmin_  :kmax_+1] )*self.grad_z


    # -------------------------------------------------
    # Convective flux xx
    def vel_conv_xx(self,state_u,state_v,grad_x):
        imin_ = self.imin_; imax_ = self.imax_
        jmin_ = self.jmin_; jmax_ = self.jmax_+1
        kmin_ = self.kmin_; kmax_ = self.kmax_+1

        grad_x -= ( state_u.var_i[imin_:imax_+1,jmin_:jmax_,kmin_:kmax_] *
                    state_v.var_i[imin_:imax_+1,jmin_:jmax_,kmin_:kmax_] -
                    state_u.var_i[imin_-1:imax_,jmin_:jmax_,kmin_:kmax_] *
                    state_v.var_i[imin_-1:imax_,jmin_:jmax_,kmin_:kmax_] )*self.grad_x
        
    # -------------------------------------------------
    # Convective flux yy
    def vel_conv_yy(self,state_u,state_v,grad_y):
        imin_ = self.imin_; imax_ = self.imax_+1
        jmin_ = self.jmin_; jmax_ = self.jmax_
        kmin_ = self.kmin_; kmax_ = self.kmax_+1

        grad_y -= ( state_u.var_i[imin_:imax_,jmin_:jmax_+1,kmin_:kmax_] *
                    state_v.var_i[imin_:imax_,jmin_:jmax_+1,kmin_:kmax_] -
                    state_u.var_i[imin_:imax_,jmin_-1:jmax_,kmin_:kmax_] *
                    state_v.var_i[imin_:imax_,jmin_-1:jmax_,kmin_:kmax_] )*self.grad_y
        
    # -------------------------------------------------
    # Convective flux zz
    def vel_conv_zz(self,state_u,state_w,grad_z):
        imin_ = self.imin_; imax_ = self.imax_+1
        jmin_ = self.jmin_; jmax_ = self.jmax_+1
        kmin_ = self.kmin_; kmax_ = self.kmax_

        grad_z -= ( state_u.var_i[imin_:imax_,jmin_:jmax_,kmin_:kmax_+1] *
                    state_w.var_i[imin_:imax_,jmin_:jmax_,kmin_:kmax_+1] -
                    state_u.var_i[imin_:imax_,jmin_:jmax_,kmin_-1:kmax_] *
                    state_w.var_i[imin_:imax_,jmin_:jmax_,kmin_-1:kmax_] )*self.grad_z
        
    # -------------------------------------------------
    # Off-diagonal convective flux x
    def vel_conv_x(self,state_u,state_v,grad_x):
        imin_ = self.imin_; imax_ = self.imax_+1
        jmin_ = self.jmin_; jmax_ = self.jmax_+1
        kmin_ = self.kmin_; kmax_ = self.kmax_+1

        grad_x -= ( state_u.var_i[imin_+1:imax_+1,jmin_:jmax_,kmin_:kmax_] *
                    state_v.var_i[imin_+1:imax_+1,jmin_:jmax_,kmin_:kmax_] -
                    state_u.var_i[imin_  :imax_  ,jmin_:jmax_,kmin_:kmax_] *
                    state_v.var_i[imin_  :imax_  ,jmin_:jmax_,kmin_:kmax_] )*self.grad_x
        
    # -------------------------------------------------
    # Off-diagonal convective flux y
    def vel_conv_y(self,state_u,state_v,grad_y):
        imin_ = self.imin_; imax_ = self.imax_+1
        jmin_ = self.jmin_; jmax_ = self.jmax_+1
        kmin_ = self.kmin_; kmax_ = self.kmax_+1

        grad_y -= ( state_u.var_i[imin_:imax_,jmin_+1:jmax_+1,kmin_:kmax_] *
                    state_v.var_i[imin_:imax_,jmin_+1:jmax_+1,kmin_:kmax_] -
                    state_u.var_i[imin_:imax_,jmin_  :jmax_  ,kmin_:kmax_] *
                    state_v.var_i[imin_:imax_,jmin_  :jmax_  ,kmin_:kmax_] )*self.grad_y
        
    # -------------------------------------------------
    # Off-diagonal convective flux z
    def vel_conv_z(self,state_u,state_w,grad_z):
        imin_ = self.imin_; imax_ = self.imax_+1
        jmin_ = self.jmin_; jmax_ = self.jmax_+1
        kmin_ = self.kmin_; kmax_ = self.kmax_+1

        grad_z -= ( state_u.var_i[imin_:imax_,jmin_:jmax_,kmin_+1:kmax_+1] *
                    state_w.var_i[imin_:imax_,jmin_:jmax_,kmin_+1:kmax_+1] -
                    state_u.var_i[imin_:imax_,jmin_:jmax_,kmin_  :kmax_  ] *
                    state_w.var_i[imin_:imax_,jmin_:jmax_,kmin_  :kmax_  ] )*self.grad_z


