# ------------------------------------------------------------------------
#
# PyFlow: A GPU-accelerated CFD platform written in Python
#
# @file metric_collocated.py
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
# Collocated central-difference schemes for velocity
# ----------------------------------------------------
def FiniteDifference_u(genericOrder,dx,state_P):
    # Inverse dx
    dxi = 1.0/dx
    
    if (genericOrder==2):
        # First derivatives - central points - 2nd-order CD
        state_P.grad_x[1:-1,:,:] = (state_P.var[2:,:,:] - state_P.var[0:-2,:,:])*0.5*dxi
        state_P.grad_y[:,1:-1,:] = (state_P.var[:,2:,:] - state_P.var[:,0:-2,:])*0.5*dxi
        state_P.grad_z[:,:,1:-1] = (state_P.var[:,:,2:] - state_P.var[:,:,0:-2])*0.5*dxi

        # Second derivatives
        state_P.grad_xx[1:-1,:,:] = (state_P.var[2:,:,:] -2*state_P.var[1:-1,:,:]+ state_P.var[0:-2,:,:])/(dx*dx)   
        state_P.grad_yy[:,1:-1,:] = (state_P.var[:,2:,:] -2*state_P.var[:,1:-1,:]+ state_P.var[:,0:-2,:])/(dx*dx)  
        state_P.grad_zz[:,:,1:-1] = (state_P.var[:,:,2:] -2*state_P.var[:,:,1:-1]+ state_P.var[:,:,0:-2])/(dx*dx) 
        
        state_P.grad_xy[1:-1,1:-1,:] = (state_P.var[2:,2:,:] - state_P.var[0:-2,2:,:]
                                        - state_P.var[2:,0:-2,:] + state_P.var[0:-2,0:-2,:] )/(4*dx*dx)     
        state_P.grad_xz[1:-1,:,1:-1] = (state_P.var[2:,:,2:] - state_P.var[0:-2,:,2:]
                                        - state_P.var[2:,:,0:-2] + state_P.var[0:-2,:,0:-2] )/(4*dx*dx)     
        state_P.grad_yz[:,1:-1,1:-1] = (state_P.var[:,2:,2:] - state_P.var[:,0:-2,2:]
                                        - state_P.var[:,2:,0:-2] + state_P.var[:,0:-2,0:-2] )/(4*dx*dx)    

    elif (genericOrder==6):
        #
        # [JFM] do not use -- needs boundary stencils
        #
        #higher-order finite difference....
        state_P.grad_x[3:-3,1:-1,1:-1] = ( - (1.0/60.0)*state_P.var[0:-6,1:-1,1:-1]
                                  + (3.0/20.0)*state_P.var[1:-5,1:-1,1:-1]
                                  - (3.0/4.0)*state_P.var[2:-4,1:-1,1:-1]
                                  + (3.0/4.0)*state_P.var[4:-2,1:-1,1:-1]
                                  - (3.0/20.0)*state_P.var[5:-1,1:-1,1:-1]
                                  + (1.0/60.0)*state_P.var[6:,1:-1,1:-1])/(dx)
        
        state_P.grad_y[1:-1,3:-3,1:-1] = ( - (1.0/60.0)*state_P.var[1:-1,0:-6,1:-1]
                                  + (3.0/20.0)*state_P.var[1:-1,1:-5,1:-1]
                                  - (3.0/4.0)*state_P.var[1:-1,2:-4,1:-1]
                                  + (3.0/4.0)*state_P.var[1:-1,4:-2,1:-1]
                                  - (3.0/20.0)*state_P.var[1:-1,5:-1,1:-1]
                                  + (1.0/60.0)*state_P.var[1:-1,6:,1:-1])/(dx)
        
        state_P.grad_z[1:-1,1:-1,3:-3] = ( - (1.0/60.0)*state_P.var[1:-1,1:-1,0:-6]
                                  + (3.0/20.0)*state_P.var[1:-1,1:-1,1:-5]
                                  - (3.0/4.0)*state_P.var[1:-1,1:-1,2:-4]
                                  + (3.0/4.0)*state_P.var[1:-1,1:-1,4:-2]
                                  - (3.0/20.0)*state_P.var[1:-1,1:-1,5:-1]
                                  + (1.0/60.0)*state_P.var[1:-1,1:-1,6:])/(dx)      
    
    
    #u periodic boundary conditions
    state_P.grad_x[-1,:,:] = (state_P.var[0,:,:] - state_P.var[-2,:,:])*0.5*dxi
    state_P.grad_x[ 0,:,:] = (state_P.var[1,:,:] - state_P.var[-1,:,:])*0.5*dxi
    
    state_P.grad_y[:,-1,:] = (state_P.var[:,0,:] - state_P.var[:,-2,:])*0.5*dxi
    state_P.grad_y[:, 0,:] = (state_P.var[:,1,:] - state_P.var[:,-1,:])*0.5*dxi
    
    state_P.grad_z[:,:,-1] = (state_P.var[:,:,0] - state_P.var[:,:,-2])*0.5*dxi
    state_P.grad_z[:,:, 0] = (state_P.var[:,:,1] - state_P.var[:,:,-1])*0.5*dxi

    
    state_P.grad_xx[-1,:,:] = (state_P.var[0,:,:] -2*state_P.var[-1,:,:]+ state_P.var[-2,:,:])/(dx*dx)   
    state_P.grad_xx[0,:,:] = (state_P.var[1,:,:] -2*state_P.var[0,:,:]+ state_P.var[-1,:,:])/(dx*dx)   

    state_P.grad_yy[:,-1,:] = (state_P.var[:,0,:] -2*state_P.var[:,-1,:]+ state_P.var[:,-2,:])/(dx*dx)  
    state_P.grad_yy[:,0,:] = (state_P.var[:,1,:] -2*state_P.var[:,0,:]+ state_P.var[:,-1,:])/(dx*dx)  

    state_P.grad_zz[:,:,-1] = (state_P.var[:,:,0] -2*state_P.var[:,:,-1]+ state_P.var[:,:,-2])/(dx*dx)  
    state_P.grad_zz[:,:,0] = (state_P.var[:,:,1] -2*state_P.var[:,:,0]+ state_P.var[:,:,-1])/(dx*dx) 

    state_P.grad_xy[-1,1:-1,:] = (state_P.var[0,2:,:] - state_P.var[-2,2:,:] - state_P.var[0,0:-2,:] + state_P.var[-2,0:-2,:] )/(4*dx*dx)   
    state_P.grad_xy[0,1:-1,:] = (state_P.var[1,2:,:] - state_P.var[-1,2:,:] - state_P.var[1,0:-2,:] + state_P.var[-1,0:-2,:] )/(4*dx*dx)     
    state_P.grad_xy[1:-1,-1,:] = (state_P.var[2:,0,:] - state_P.var[0:-2,0,:] - state_P.var[2:,-2,:] + state_P.var[0:-2,-2,:] )/(4*dx*dx)
    state_P.grad_xy[1:-1,0,:] = (state_P.var[2:,1,:] - state_P.var[0:-2,1,:] - state_P.var[2:,-1,:] + state_P.var[0:-2,-1,:] )/(4*dx*dx)
    state_P.grad_xy[-1,-1,:] = (state_P.var[0,0,:] - state_P.var[-2,0,:] - state_P.var[0,-2,:] + state_P.var[-2,-2,:] )/(4*dx*dx)   
    state_P.grad_xy[-1,0,:] = (state_P.var[0,1,:] - state_P.var[-2,1,:] - state_P.var[0,-1,:] + state_P.var[-2,-1,:] )/(4*dx*dx)
    state_P.grad_xy[0,-1,:] = (state_P.var[1,0,:] - state_P.var[-1,0,:] - state_P.var[1,-2,:] + state_P.var[-1,-2,:] )/(4*dx*dx) 
    state_P.grad_xy[0,0,:] = (state_P.var[1,1,:] - state_P.var[-1,1,:] - state_P.var[1,-1,:] + state_P.var[-1,-1,:] )/(4*dx*dx)

    state_P.grad_xy[-1,1:-1,:] = (state_P.var[0,2:,:] - state_P.var[-2,2:,:] - state_P.var[0,0:-2,:] + state_P.var[-2,0:-2,:] )/(4*dx*dx)   
    state_P.grad_xy[0,1:-1,:] = (state_P.var[1,2:,:] - state_P.var[-1,2:,:] - state_P.var[1,0:-2,:] + state_P.var[-1,0:-2,:] )/(4*dx*dx)     
    state_P.grad_xy[1:-1,-1,:] = (state_P.var[2:,0,:] - state_P.var[0:-2,0,:] - state_P.var[2:,-2,:] + state_P.var[0:-2,-2,:] )/(4*dx*dx)
    state_P.grad_xy[1:-1,0,:] = (state_P.var[2:,1,:] - state_P.var[0:-2,1,:] - state_P.var[2:,-1,:] + state_P.var[0:-2,-1,:] )/(4*dx*dx)
    state_P.grad_xy[-1,-1,:] = (state_P.var[0,0,:] - state_P.var[-2,0,:] - state_P.var[0,-2,:] + state_P.var[-2,-2,:] )/(4*dx*dx)   
    state_P.grad_xy[-1,0,:] = (state_P.var[0,1,:] - state_P.var[-2,1,:] - state_P.var[0,-1,:] + state_P.var[-2,-1,:] )/(4*dx*dx)
    state_P.grad_xy[0,-1,:] = (state_P.var[1,0,:] - state_P.var[-1,0,:] - state_P.var[1,-2,:] + state_P.var[-1,-2,:] )/(4*dx*dx) 
    state_P.grad_xy[0,0,:] = (state_P.var[1,1,:] - state_P.var[-1,1,:] - state_P.var[1,-1,:] + state_P.var[-1,-1,:] )/(4*dx*dx)





# ----------------------------------------------------
# Collocated central-difference schemes for pressure
# ----------------------------------------------------
#def FiniteDifference_P(genericOrder, dx, p_P, p_OLD_P, Source_term)
#    if (genericOrder==2):
