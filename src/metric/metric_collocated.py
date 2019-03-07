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



#Upwind FD scheme
def FiniteDifference_u_UPWIND(u_P):
    #u derivatives
    u_x_P[1:-1,:,:] = (u_P[1:-1,:,:] - u_P[0:-2,:,:])/(1*dx)
    u_y_P[:,1:-1,:] = (u_P[:,1:-1,:] - u_P[:,0:-2,:])/(1*dx)
    u_z_P[:,:,1:-1] = (u_P[:,:,1:-1] - u_P[:,:,0:-2])/(1*dx)


    #u periodic boundary conditions
    u_x_P[-1,:,:] = (u_P[-1,:,:] - u_P[-2,:,:])/(1*dx)
    u_x_P[0,:,:] = (u_P[0,:,:] - u_P[-1,:,:])/(1*dx)
    
    u_y_P[:,-1,:] = (u_P[:,-1,:] - u_P[:,-2,:])/(1*dx)
    u_y_P[:,0,:] = (u_P[:,0,:] - u_P[:,-1,:])/(1*dx)
    
    u_z_P[:,:,-1] = (u_P[:,:,-1] - u_P[:,:,-2])/(1*dx)
    u_z_P[:,:,0] = (u_P[:,:,0] - u_P[:,:,-1])/(1*dx)

    return u_x_P, u_y_P, u_z_P





def FiniteDifference_u_DOWNWIND(u_P):
    #u derivatives
    u_x_P[1:-1,:,:] = (u_P[2:,:,:] - u_P[1:-1,:,:])/(1*dx)
    u_y_P[:,1:-1,:] = (u_P[:,2:,:] - u_P[:,1:-1,:])/(1*dx)
    u_z_P[:,:,1:-1] = (u_P[:,:,2:] - u_P[:,:,1:-1])/(1*dx)


    #u periodic boundary conditions
    u_x_P[-1,:,:] = (u_P[0,:,:] - u_P[-1,:,:])/(1*dx)
    u_x_P[0,:,:] = (u_P[1,:,:] - u_P[0,:,:])/(1*dx)
    
    u_y_P[:,-1,:] = (u_P[:,0,:] - u_P[:,-1,:])/(1*dx)
    u_y_P[:,0,:] = (u_P[:,1,:] - u_P[:,0,:])/(1*dx)
    
    u_z_P[:,:,-1] = (u_P[:,:,0] - u_P[:,:,-1])/(1*dx)
    u_z_P[:,:,0] = (u_P[:,:,1] - u_P[:,:,0])/(1*dx)


    return u_x_P, u_y_P, u_z_P


#Central Difference FD scheme
def FiniteDifference_u(dx, u_P, u_x_P, u_y_P, u_z_P, u_xx_P, u_yy_P, u_zz_P, u_xy_P, u_xz_P, u_yz_P):
    #u derivatives
    u_x_P[1:-1,:,:] = (u_P[2:,:,:] - u_P[0:-2,:,:])/(2*dx)
    u_y_P[:,1:-1,:] = (u_P[:,2:,:] - u_P[:,0:-2,:])/(2*dx)
    u_z_P[:,:,1:-1] = (u_P[:,:,2:] - u_P[:,:,0:-2])/(2*dx)
    u_xx_P[1:-1,:,:] = (u_P[2:,:,:] -2*u_P[1:-1,:,:]+ u_P[0:-2,:,:])/(dx*dx)   
    u_yy_P[:,1:-1,:] = (u_P[:,2:,:] -2*u_P[:,1:-1,:]+ u_P[:,0:-2,:])/(dx*dx)  
    u_zz_P[:,:,1:-1] = (u_P[:,:,2:] -2*u_P[:,:,1:-1]+ u_P[:,:,0:-2])/(dx*dx) 

    u_xy_P[1:-1,1:-1,:] = (u_P[2:,2:,:] - u_P[0:-2,2:,:] - u_P[2:,0:-2,:] + u_P[0:-2,0:-2,:] )/(4*dx*dx)     
    u_xz_P[1:-1,:,1:-1] = (u_P[2:,:,2:] - u_P[0:-2,:,2:] - u_P[2:,:,0:-2] + u_P[0:-2,:,0:-2] )/(4*dx*dx)     
    u_yz_P[:,1:-1,1:-1] = (u_P[:,2:,2:] - u_P[:,0:-2,2:] - u_P[:,2:,0:-2] + u_P[:,0:-2,0:-2] )/(4*dx*dx)    
    
    #higher-order finite difference....
    u_x_P[3:-3,1:-1,1:-1] = ( -(1.0/60.0)*u_P[0:-6,1:-1,1:-1] + (3.0/20.0)*u_P[1:-5,1:-1,1:-1] -(3.0/4.0)*u_P[2:-4,1:-1,1:-1] +  (3.0/4.0)*u_P[4:-2,1:-1,1:-1] - (3.0/20.0)*u_P[5:-1,1:-1,1:-1] + (1.0/60.0)*u_P[6:,1:-1,1:-1])/(dx)
    
    u_y_P[1:-1,3:-3,1:-1] = ( -(1.0/60.0)*u_P[1:-1,0:-6,1:-1] + (3.0/20.0)*u_P[1:-1,1:-5,1:-1] -(3.0/4.0)*u_P[1:-1,2:-4,1:-1] +  (3.0/4.0)*u_P[1:-1,4:-2,1:-1] - (3.0/20.0)*u_P[1:-1,5:-1,1:-1] + (1.0/60.0)*u_P[1:-1,6:,1:-1])/(dx)       
    
    u_z_P[1:-1,1:-1,3:-3] = ( -(1.0/60.0)*u_P[1:-1,1:-1,0:-6] + (3.0/20.0)*u_P[1:-1,1:-1,1:-5] -(3.0/4.0)*u_P[1:-1,1:-1,2:-4] +  (3.0/4.0)*u_P[1:-1,1:-1,4:-2] - (3.0/20.0)*u_P[1:-1,1:-1,5:-1] + (1.0/60.0)*u_P[1:-1,1:-1,6:])/(dx)      
    
    #data_in[12:-12,12:-12,12:-12,4:7] = ( -(1.0/60.0)*data_in[0:-24,12:-12,12:-12,0:3] + (3.0/20.0)*data_in[4:-20,12:-12,12:-12,0:3] -(3.0/4.0)*data_in[8:-16,12:-12,12:-12,0:3] +  (3.0/4.0)*data_in[16:-8,12:-12,12:-12,0:3] - (3.0/20.0)*data_in[20:-4,12:-12,12:-12,0:3] + (1.0/60.0)*data_in[24:,12:-12,12:-12,0:3])/(Delta_LES_grid)    
    


    #u periodic boundary conditions
    u_x_P[-1,:,:] = (u_P[0,:,:] - u_P[-2,:,:])/(2*dx)
    u_x_P[0,:,:] = (u_P[1,:,:] - u_P[-1,:,:])/(2*dx)
    
    u_y_P[:,-1,:] = (u_P[:,0,:] - u_P[:,-2,:])/(2*dx)
    u_y_P[:,0,:] = (u_P[:,1,:] - u_P[:,-1,:])/(2*dx)
    
    u_z_P[:,:,-1] = (u_P[:,:,0] - u_P[:,:,-2])/(2*dx)
    u_z_P[:,:,0] = (u_P[:,:,1] - u_P[:,:,-1])/(2*dx)

    u_xx_P[-1,:,:] = (u_P[0,:,:] -2*u_P[-1,:,:]+ u_P[-2,:,:])/(dx*dx)   
    u_xx_P[0,:,:] = (u_P[1,:,:] -2*u_P[0,:,:]+ u_P[-1,:,:])/(dx*dx)   

    u_yy_P[:,-1,:] = (u_P[:,0,:] -2*u_P[:,-1,:]+ u_P[:,-2,:])/(dx*dx)  
    u_yy_P[:,0,:] = (u_P[:,1,:] -2*u_P[:,0,:]+ u_P[:,-1,:])/(dx*dx)  

    u_zz_P[:,:,-1] = (u_P[:,:,0] -2*u_P[:,:,-1]+ u_P[:,:,-2])/(dx*dx)  
    u_zz_P[:,:,0] = (u_P[:,:,1] -2*u_P[:,:,0]+ u_P[:,:,-1])/(dx*dx) 

    u_xy_P[-1,1:-1,:] = (u_P[0,2:,:] - u_P[-2,2:,:] - u_P[0,0:-2,:] + u_P[-2,0:-2,:] )/(4*dx*dx)   
    u_xy_P[0,1:-1,:] = (u_P[1,2:,:] - u_P[-1,2:,:] - u_P[1,0:-2,:] + u_P[-1,0:-2,:] )/(4*dx*dx)     
    u_xy_P[1:-1,-1,:] = (u_P[2:,0,:] - u_P[0:-2,0,:] - u_P[2:,-2,:] + u_P[0:-2,-2,:] )/(4*dx*dx)
    u_xy_P[1:-1,0,:] = (u_P[2:,1,:] - u_P[0:-2,1,:] - u_P[2:,-1,:] + u_P[0:-2,-1,:] )/(4*dx*dx)
    u_xy_P[-1,-1,:] = (u_P[0,0,:] - u_P[-2,0,:] - u_P[0,-2,:] + u_P[-2,-2,:] )/(4*dx*dx)   
    u_xy_P[-1,0,:] = (u_P[0,1,:] - u_P[-2,1,:] - u_P[0,-1,:] + u_P[-2,-1,:] )/(4*dx*dx)
    u_xy_P[0,-1,:] = (u_P[1,0,:] - u_P[-1,0,:] - u_P[1,-2,:] + u_P[-1,-2,:] )/(4*dx*dx) 
    u_xy_P[0,0,:] = (u_P[1,1,:] - u_P[-1,1,:] - u_P[1,-1,:] + u_P[-1,-1,:] )/(4*dx*dx)

    u_xy_P[-1,1:-1,:] = (u_P[0,2:,:] - u_P[-2,2:,:] - u_P[0,0:-2,:] + u_P[-2,0:-2,:] )/(4*dx*dx)   
    u_xy_P[0,1:-1,:] = (u_P[1,2:,:] - u_P[-1,2:,:] - u_P[1,0:-2,:] + u_P[-1,0:-2,:] )/(4*dx*dx)     
    u_xy_P[1:-1,-1,:] = (u_P[2:,0,:] - u_P[0:-2,0,:] - u_P[2:,-2,:] + u_P[0:-2,-2,:] )/(4*dx*dx)
    u_xy_P[1:-1,0,:] = (u_P[2:,1,:] - u_P[0:-2,1,:] - u_P[2:,-1,:] + u_P[0:-2,-1,:] )/(4*dx*dx)
    u_xy_P[-1,-1,:] = (u_P[0,0,:] - u_P[-2,0,:] - u_P[0,-2,:] + u_P[-2,-2,:] )/(4*dx*dx)   
    u_xy_P[-1,0,:] = (u_P[0,1,:] - u_P[-2,1,:] - u_P[0,-1,:] + u_P[-2,-1,:] )/(4*dx*dx)
    u_xy_P[0,-1,:] = (u_P[1,0,:] - u_P[-1,0,:] - u_P[1,-2,:] + u_P[-1,-2,:] )/(4*dx*dx) 
    u_xy_P[0,0,:] = (u_P[1,1,:] - u_P[-1,1,:] - u_P[1,-1,:] + u_P[-1,-1,:] )/(4*dx*dx)

    return u_x_P, u_y_P, u_z_P, u_xx_P, u_yy_P, u_zz_P, u_xy_P, u_xz_P, u_yz_P
