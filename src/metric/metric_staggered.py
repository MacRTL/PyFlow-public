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

# ----------------------------------------------------
# Generic staggered central-difference schemes
# ----------------------------------------------------
class metric_generic:
    def __init__(self,geometry):
        # Initialize the grid metrics
        self.grid_geometry = geometry.type
        if (self.grid_geometry=='uniform'):
            self.div_x  = 1.0*geometry.dxi
            self.div_y  = 1.0*geometry.dyi
            self.div_z  = 1.0*geometry.dzi
            self.grad_x  = 1.0*geometry.dxi
            self.grad_y  = 1.0*geometry.dyi
            self.grad_z  = 1.0*geometry.dzi
            self.grad_xm = 1.0*geometry.dxi
            self.grad_ym = 1.0*geometry.dyi
            self.grad_zm = 1.0*geometry.dzi
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
        state.var_i[:-1,:,:] = self.interp_x*(state.var[1:,:,:] + state.var[:-1,:,:])

        # Periodic boundary
        state.var_i[-1,:,:] = self.interp_x*(state.var[0,:,:] + state.var[-1,:,:])

    # ----------------------------------------------------
    # Interpolation of a cell-centered variable to y-faces
    def interp_sc_y(self,state):
        state.var_i[:,:-1,:] = self.interp_y*(state.var[:,1:,:] + state.var[:,:-1,:])

        # Periodic boundary
        state.var_i[:,-1,:] = self.interp_y*(state.var[:,0,:] + state.var[:,-1,:])

    # ----------------------------------------------------
    # Interpolation of a cell-centered variable to z-faces
    def interp_sc_z(self,state):
        state.var_i[:,:,:-1] = self.interp_z*(state.var[:,:,1:] + state.var[:,:,:-1])

        # Periodic boundary
        state.var_i[:,:,-1] = self.interp_z*(state.var[:,:,0] + state.var[:,:,-1])


    # -----------------------------------------------
    # Interpolation of the x-velocity to cell centers
    def interp_u_xm(self,state):
        state.var_i[:-1,:,:] = self.interp_xm*(state.var[1:,:,:] + state.var[:-1,:,:])

        # Periodic boundary
        state.var_i[-1,:,:] = self.interp_xm*(state.var[0,:,:] + state.var[-1,:,:])

    # -----------------------------------------------
    # Interpolation of the y-velocity to cell centers
    def interp_v_ym(self,state):
        state.var_i[:,:-1,:] = self.interp_ym*(state.var[:,1:,:] + state.var[:,:-1,:])

        # Periodic boundary
        state.var_i[:,-1,:] = self.interp_ym*(state.var[:,0,:] + state.var[:,-1,:])

    # -----------------------------------------------
    # Interpolation of the z-velocity to cell centers
    def interp_w_zm(self,state):
        state.var_i[:,:,:-1] = self.interp_zm*(state.var[:,:,1:] + state.var[:,:,:-1])

        # Periodic boundary
        state.var_i[:,:,-1] = self.interp_zm*(state.var[:,:,0] + state.var[:,:,-1])

        
    # ---------------------------------------------------
    # Interpolation of the v- and w-velocities to x-edges
    def interp_vw_x(self,state):
        state.var_i[1:,:,:] = self.interp_x*(state.var[1:,:,:] + state.var[:-1,:,:])

        # Periodic boundary
        state.var_i[0,:,:] = self.interp_x*(state.var[0,:,:] + state.var[-1,:,:])
        
    # ---------------------------------------------------
    # Interpolation of the u- and w-velocities to y-edges
    def interp_uw_y(self,state):
        state.var_i[:,1:,:] = self.interp_y*(state.var[:,1:,:] + state.var[:,:-1,:])

        # Periodic boundary
        state.var_i[:,0,:] = self.interp_y*(state.var[:,0,:] + state.var[:,-1,:])

    # ---------------------------------------------------
    # Interpolation of the u- and v-velocities to z-edges
    def interp_uv_z(self,state):
        state.var_i[:,:,1:] = self.interp_z*(state.var[:,:,1:] + state.var[:,:,:-1])

        # Periodic boundary
        state.var_i[:,:,0] = self.interp_z*(state.var[:,:,0] + state.var[:,:,-1])


    # ------------------------------------------------------------
    # Gradient of velocity to locations needed by the viscous flux
    def grad_vel_visc(self,state):
        state.grad_x[:-1,:,:] = self.grad_x*(state.var[1:,:,:] - state.var[:-1,:,:])
        state.grad_y[:,:-1,:] = self.grad_y*(state.var[:,1:,:] - state.var[:,:-1,:])
        state.grad_z[:,:,:-1] = self.grad_z*(state.var[:,:,1:] - state.var[:,:,:-1])
    
        # Periodic boundary conditions
        # x
        state.grad_x[-1,:,:] = self.grad_x*(state.var[0,:,:] - state.var[-2,:,:])
        # y
        state.grad_y[:,-1,:] = self.grad_y*(state.var[:,0,:] - state.var[:,-2,:])
        # z
        state.grad_z[:,:,-1] = self.grad_z*(state.var[:,:,0] - state.var[:,:,-2])



    # -------------------------------------------------
    # Divergence of the viscous flux to the cell x-face
    def div_visc_x(self,SC,div_x):
        div_x[:-1,:,:] += self.div_x*(SC[1:,:,:] - SC[:-1,:,:])
    
        # Periodic boundary conditions
        div_x[-1,:,:] += self.div_x*(SC[0,:,:] - SC[-2,:,:])

        
    # -------------------------------------------------
    # Divergence of the viscous flux to the cell y-face
    def div_visc_y(self,SC,div_y):
        div_y[:,:-1,:] += self.div_y*(SC[:,1:,:] - SC[:,:-1,:])

        # Periodic boundary conditions
        div_y[:,-1,:] += self.div_y*(SC[:,0,:] - SC[:,-2,:])

        
    # -------------------------------------------------
    # Divergence of the viscous flux to the cell z-face
    def div_visc_z(self,SC,div_z):
        div_z[:,:,:-1] += self.div_z*(SC[:,:,1:] - SC[:,:,:-1])

        # Periodic boundary conditions
        div_z[:,:,-1] += self.div_z*(SC[:,:,0] - SC[:,:,-2])

        
    # -------------------------------------------------
    # Convective flux xx
    def vel_conv_xx(self,state_u,state_v,grad_x):
        
        grad_x[1:,:,:] -= self.grad_x*( state_u.var_i[1:,:,:] * state_v.var_i[1:,:,:]
                                        - state_u.var_i[:-1,:,:] * state_v.var_i[:-1,:,:] )

        # Periodic boundary conditions
        grad_x[0,:,:] -= self.grad_x*( state_u.var_i[0,:,:] * state_v.var_i[0,:,:]
                                       - state_u.var_i[-1,:,:] * state_v.var_i[-1,:,:] )
        
    # -------------------------------------------------
    # Convective flux yy
    def vel_conv_yy(self,state_u,state_v,grad_y):
        
        grad_y[:,1:,:] -= self.grad_y*( state_u.var_i[:,1:,:] * state_v.var_i[:,1:,:]
                                        - state_u.var_i[:,:-1,:] * state_v.var_i[:,:-1,:] )

        # Periodic boundary conditions
        grad_y[:,0,:] -= self.grad_y*( state_u.var_i[:,0,:] * state_v.var_i[:,0,:]
                                       - state_u.var_i[:,-1,:] * state_v.var_i[:,-1,:] )
        
    # -------------------------------------------------
    # Convective flux zz
    def vel_conv_zz(self,state_u,state_v,grad_z):
        
        grad_z[:,:,1:] -= self.grad_z*( state_u.var_i[:,:,1:] * state_v.var_i[:,:,1:]
                                        - state_u.var_i[:,:,:-1] * state_v.var_i[:,:,:-1] )

        # Periodic boundary conditions
        grad_z[:,:,0] -= self.grad_z*( state_u.var_i[:,:,0] * state_v.var_i[:,:,0]
                                       - state_u.var_i[:,:,-1] * state_v.var_i[:,:,-1] )
        
    # -------------------------------------------------
    # Convective flux yy
    def vel_conv_yy(self,state_u,state_v,grad_y):
        
        grad_y[:,1:,:] -= self.grad_y*( state_u.var_i[:,1:,:] * state_v.var_i[:,1:,:]
                                        - state_u.var_i[:,:-1,:] * state_v.var_i[:,:-1,:] )

        # Periodic boundary conditions
        grad_y[:,0,:] -= self.grad_y*( state_u.var_i[:,0,:] * state_v.var_i[:,0,:]
                                       - state_u.var_i[:,-1,:] * state_v.var_i[:,-1,:] )
        
    # -------------------------------------------------
    # Off-diagonal convective flux x
    def vel_conv_x(self,state_u,state_v,grad_x):
        
        grad_x[:-1,:,:] -= self.grad_x*( state_u.var_i[1:,:,:] * state_v.var_i[1:,:,:]
                                          - state_u.var_i[:-1,:,:] * state_v.var_i[:-1,:,:] )

        # Periodic boundary conditions
        grad_x[-1,:,:] -= self.grad_x*( state_u.var_i[0,:,:] * state_v.var_i[0,:,:]
                                        - state_u.var_i[-1,:,:] * state_v.var_i[-1,:,:] )
        
    # -------------------------------------------------
    # Off-diagonal convective flux y
    def vel_conv_y(self,state_u,state_v,grad_y):
        
        grad_y[:,:-1,:] -= self.grad_y*( state_u.var_i[:,1:,:] * state_v.var_i[:,1:,:]
                                          - state_u.var_i[:,:-1,:] * state_v.var_i[:,:-1,:] )

        # Periodic boundary conditions
        grad_y[:,-1,:] -= self.grad_y*( state_u.var_i[:,0,:] * state_v.var_i[:,0,:]
                                        - state_u.var_i[:,-1,:] * state_v.var_i[:,-1,:] )
        
    # -------------------------------------------------
    # Off-diagonal convective flux z
    def vel_conv_z(self,state_u,state_v,grad_z):
        
        grad_z[:,:,:-1] -= self.grad_z*( state_u.var_i[:,:,1:] * state_v.var_i[:,:,1:]
                                          - state_u.var_i[:,:,:-1] * state_v.var_i[:,:,:-1] )

        # Periodic boundary conditions
        grad_z[:,:,-1] -= self.grad_z*( state_u.var_i[:,:,0] * state_v.var_i[:,:,0]
                                        - state_u.var_i[:,:,-1] * state_v.var_i[:,:,-1] )

        
    # -------------------------------------------------
    # Divergence of the velocity field
    #def div_vel(self,state_u,state_v,state_w):

        # PASS THIS SOME MEMORY!

        #div_vel = 
