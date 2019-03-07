# ------------------------------------------------------------------------
#
# PyFlow: A GPU-accelerated CFD platform written in Python
#
# @author Justin A. Sirignano
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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import datasets, transforms
from torch.autograd import Variable

import time
import copy
import sys
import os

# Load PyFlow modules
sys.path.append("../data")
import dataReader as dr
import constants as const
#
sys.path.append("../metric")
import metric_collocated as metric
#
sys.path.append("../sfsmodel")
import sfsmodel_nn
import sfsmodel_smagorinsky


# Restart and target files to read
restartFileStr = '../../examples/filtered_vol_dnsbox_1024_Lx0.045_NR_00000020_coarse'
targetFileStr  = restartFileStr 

#restartFileStr = '/projects/sciteam/baxh/Downsampled_Data_Folder/data_1024_Lx0.045_NR_Delta32_downsample8x/filtered_vol_dnsbox_1024_Lx0.045_NR_00000020_coarse'
#targetFileStr  = '/projects/sciteam/baxh/Downsampled_Data_Folder/data_1024_Lx0.045_NR_Delta32_downsample8x/filtered_vol_dnsbox_1024_Lx0.045_NR_00000030_coarse'


# ----------------------------------------------------
# Initialize data files
# ----------------------------------------------------
# Read restart data file
xGrid,yGrid,zGrid,names,dataTime,data = dr.readNGA(restartFileStr)
data_restart = data[:,:,:,0:4]

# Grid parameters
#   Currently configured for uniform grid
Nx = len(xGrid); Lx = xGrid[-1]-xGrid[0]; dx = Lx/float(Nx)
Ny = len(yGrid); Ly = yGrid[-1]-yGrid[0]; dy = Ly/float(Ny)
Nz = len(zGrid); Lz = zGrid[-1]-zGrid[0]; dz = Lz/float(Nz)

# Read target data file
xGrid_t,yGrid_t,zGrid_t,names_t,dataTime_t,data_t = dr.readNGA(targetFileStr)
data_target10 = data_t[:,:,:,0:4]

# Set up Torch state
haveCuda = torch.cuda.is_available()
if (haveCuda):
    x_max_P = torch.FloatTensor( const.x_max16 ).cuda()
    target_P = torch.FloatTensor( data_target10 ).cuda()
else:
    x_max_P = torch.FloatTensor( const.x_max16 )
    target_P = torch.FloatTensor( data_target10 )

    
# ----------------------------------------------------
# Initialize closure model(s)
# ----------------------------------------------------
# Initialize the neural network closure model
model = sfsmodel_nn.ClosureModel2()
model_name = 'LES_model_NR_March2019'
#model.load_state_dict(torch.load(model_name))
#if (haveCuda):
    #model.cuda()


#Set up some vectors which will be helpful for the PyTorch finite difference implementation..

#initial condition

# ----------------------------------------------------
# Set up initial condition
# ----------------------------------------------------
IC_u_np = data_restart[:,:,:,0:1]

#IC_v_np = np.zeros( ( 64, 64,1) )
IC_v_np = data_restart[:,:,:,1:2]

IC_w_np = data_restart[:,:,:,2:3]

IC_p_np  = data_restart[:,:,:,3:4]

IC_zeros_np = np.zeros( (Nx,Ny,Nz,1) )

Num_pressure_iterations = 100

mu = 1.8678e-5

rho = 1.2

#dt_LES = 0.000000001
dt_LES = 0.0000025

#T_LES = 2*0.0001

T_LES = 1*0.0001

    

#Constant0_P = Variable( torch.FloatTensor( Constant0_np ) )

#model.eval()

# ----------------------------------------------------
# Allocate memory for state data
# ----------------------------------------------------
if (haveCuda):
    u_P =  torch.FloatTensor( IC_u_np ).cuda()
    v_P = torch.FloatTensor( IC_v_np ).cuda()
    w_P = torch.FloatTensor( IC_w_np ).cuda()
    
    Closure_u_P =  torch.FloatTensor( IC_zeros_np ).cuda()
    Closure_v_P = torch.FloatTensor( IC_zeros_np ).cuda()
    Closure_w_P = torch.FloatTensor( IC_zeros_np ).cuda()    
    
    p_P =  torch.FloatTensor( IC_zeros_np ).cuda()
    p_OLD_P =  torch.FloatTensor( IC_zeros_np).cuda()
    p_x_P =  torch.FloatTensor( IC_p_np ).cuda()
    p_y_P =  torch.FloatTensor( IC_p_np ).cuda()
    p_z_P =  torch.FloatTensor( IC_p_np ).cuda()
    
    u_x_P =  torch.FloatTensor( IC_u_np ).cuda()
    u_y_P =  torch.FloatTensor( IC_u_np ).cuda()
    u_z_P =  torch.FloatTensor( IC_u_np ).cuda()
    u_xx_P =  torch.FloatTensor( IC_u_np ).cuda()
    u_yy_P =  torch.FloatTensor( IC_u_np ).cuda()
    u_zz_P =  torch.FloatTensor( IC_u_np ).cuda()
    u_xz_P =  torch.FloatTensor( IC_u_np ).cuda()
    u_xy_P =  torch.FloatTensor( IC_u_np ).cuda()
    u_yz_P =  torch.FloatTensor( IC_u_np ).cuda()
    
    v_x_P =  torch.FloatTensor( IC_v_np ).cuda()
    v_y_P =  torch.FloatTensor( IC_v_np ).cuda()
    v_z_P =  torch.FloatTensor( IC_v_np ).cuda()
    v_xx_P =  torch.FloatTensor( IC_v_np ).cuda()
    v_yy_P =  torch.FloatTensor( IC_v_np ).cuda()
    v_zz_P =  torch.FloatTensor( IC_v_np ).cuda()
    v_xz_P =  torch.FloatTensor( IC_v_np ).cuda()
    v_xy_P =  torch.FloatTensor( IC_v_np ).cuda()
    v_yz_P =  torch.FloatTensor( IC_v_np ).cuda()
    
    w_x_P =  torch.FloatTensor( IC_w_np ).cuda()
    w_y_P =  torch.FloatTensor( IC_w_np ).cuda()
    w_z_P =  torch.FloatTensor( IC_w_np ).cuda()
    w_xx_P =  torch.FloatTensor( IC_w_np ).cuda()
    w_yy_P =  torch.FloatTensor( IC_w_np ).cuda()
    w_zz_P =  torch.FloatTensor( IC_w_np ).cuda()
    w_xz_P =  torch.FloatTensor( IC_w_np ).cuda()
    w_xy_P =  torch.FloatTensor( IC_w_np ).cuda()
    w_yz_P =  torch.FloatTensor( IC_w_np ).cuda()
else:
    u_P =  torch.FloatTensor( IC_u_np )
    v_P = torch.FloatTensor( IC_v_np )
    w_P = torch.FloatTensor( IC_w_np )
    
    Closure_u_P =  torch.FloatTensor( IC_zeros_np )
    Closure_v_P = torch.FloatTensor( IC_zeros_np )
    Closure_w_P = torch.FloatTensor( IC_zeros_np )    
    
    p_P =  torch.FloatTensor( IC_zeros_np )
    p_OLD_P =  torch.FloatTensor( IC_zeros_np)
    p_x_P =  torch.FloatTensor( IC_p_np )
    p_y_P =  torch.FloatTensor( IC_p_np )
    p_z_P =  torch.FloatTensor( IC_p_np )
    
    u_x_P =  torch.FloatTensor( IC_u_np )
    u_y_P =  torch.FloatTensor( IC_u_np )
    u_z_P =  torch.FloatTensor( IC_u_np )
    u_xx_P =  torch.FloatTensor( IC_u_np )
    u_yy_P =  torch.FloatTensor( IC_u_np )
    u_zz_P =  torch.FloatTensor( IC_u_np )
    u_xz_P =  torch.FloatTensor( IC_u_np )
    u_xy_P =  torch.FloatTensor( IC_u_np )
    u_yz_P =  torch.FloatTensor( IC_u_np )
    
    v_x_P =  torch.FloatTensor( IC_v_np )
    v_y_P =  torch.FloatTensor( IC_v_np )
    v_z_P =  torch.FloatTensor( IC_v_np )
    v_xx_P =  torch.FloatTensor( IC_v_np )
    v_yy_P =  torch.FloatTensor( IC_v_np )
    v_zz_P =  torch.FloatTensor( IC_v_np )
    v_xz_P =  torch.FloatTensor( IC_v_np )
    v_xy_P =  torch.FloatTensor( IC_v_np )
    v_yz_P =  torch.FloatTensor( IC_v_np )
    
    w_x_P =  torch.FloatTensor( IC_w_np )
    w_y_P =  torch.FloatTensor( IC_w_np )
    w_z_P =  torch.FloatTensor( IC_w_np )
    w_xx_P =  torch.FloatTensor( IC_w_np )
    w_yy_P =  torch.FloatTensor( IC_w_np )
    w_zz_P =  torch.FloatTensor( IC_w_np )
    w_xz_P =  torch.FloatTensor( IC_w_np )
    w_xy_P =  torch.FloatTensor( IC_w_np )
    w_yz_P =  torch.FloatTensor( IC_w_np )




# ----------------------------------------------------
# Sub-iteration loop
# ----------------------------------------------------
for iterations in range(1):
    
    time1 = time.time()
    
    #for param_group in optimizer.param_groups:
    #        param_group['lr'] = 0.1*LR
    
    #optimizer.zero_grad()

    #u_P = Variable( torch.FloatTensor( IC_u_np ) )
    #v_P = Variable( torch.FloatTensor( IC_v_np ) )
    #
    Loss = 0.0
    
    #Chorin's projection method for solution of incompressible Navier-Stokes PDE with periodic boundary conditions in a box. 
        
    for i in range(int(T_LES/dt_LES)):
        
        u_x_P, u_y_P, u_z_P, u_xx_P, u_yy_P, u_zz_P, u_xy_P, u_xz_P, u_yz_P = metric.FiniteDifference_u(dx, u_P, u_x_P, u_y_P, u_z_P, u_xx_P, u_yy_P, u_zz_P, u_xy_P, u_xz_P, u_yz_P)
        
        v_x_P, v_y_P, v_z_P, v_xx_P, v_yy_P, v_zz_P, v_xy_P, v_xz_P, v_yz_P = metric.FiniteDifference_u(dx, v_P, v_x_P, v_y_P, v_z_P, v_xx_P, v_yy_P, v_zz_P, v_xy_P, v_xz_P, v_yz_P)
        
        w_x_P, w_y_P, w_z_P, w_xx_P, w_yy_P, w_zz_P, w_xy_P, w_xz_P, w_yz_P = metric.FiniteDifference_u(dx, w_P, w_x_P, w_y_P, w_z_P, w_xx_P, w_yy_P, w_zz_P, w_xy_P, w_xz_P, w_yz_P)
 
        Nonlinear_term_u_P = u_x_P*u_P + u_y_P*v_P  + u_z_P*w_P

        Nonlinear_term_v_P = v_x_P*u_P + v_y_P*v_P  + v_z_P*w_P
        
        Nonlinear_term_w_P = w_x_P*u_P + w_y_P*v_P  + w_z_P*w_P
        
        
        #Diffusion term
        
        Diffusion_term_u = u_xx_P + u_yy_P + u_zz_P
        
        Diffusion_term_v = v_xx_P + v_yy_P + v_zz_P
        
        Diffusion_term_w = w_xx_P + w_yy_P + w_zz_P
     
        
        #model_output = model( u_P)
        
        #model_output2 = model_output.cpu()
        
        Closure_u_P, Closure_v_P, Closure_w_P = sfsmodel_smagorinsky.eval(dx, u_x_P, u_y_P, u_z_P, v_x_P, v_y_P, v_z_P, w_x_P, w_y_P, w_z_P, Closure_u_P, Closure_v_P, Closure_w_P)

        u_P = u_P + ( (mu/rho)*Diffusion_term_u -Nonlinear_term_u_P  - Closure_u_P  )*dt_LES
        
        v_P = v_P + ( (mu/rho)*Diffusion_term_v -Nonlinear_term_v_P  - Closure_v_P   )*dt_LES
        
        w_P = w_P + ( (mu/rho)*Diffusion_term_w -Nonlinear_term_w_P  - Closure_w_P   )*dt_LES
        
        
        
        #Update pressure using an iteration...
        
        Source_term = rho*dx*dx*(u_x_P + v_y_P + w_z_P)
        
        for j in range( Num_pressure_iterations ):
            
            p_OLD_P[:,:,:,:] = p_P[0:,0:,0:,0:]
            
            Pressure_update_j1 = p_OLD_P[2:,1:-1,1:-1,:] + p_OLD_P[0:-2,1:-1,1:-1,:]
                 
            
            Pressure_update_j2 = p_OLD_P[1:-1,2:,1:-1,:] + p_OLD_P[1:-1,0:-2,1:-1,:]
            
            Pressure_update_j3 = p_OLD_P[1:-1,1:-1,2:,:] + p_OLD_P[1:-1,1:-1,0:-2,:]

            #interior of domain
            p_P[1:-1, 1:-1,1:-1,:] = (1.0/6.0)*( Pressure_update_j1 + Pressure_update_j2  + Pressure_update_j3 - Source_term[1:-1,1:-1,1:-1,:])
            
            #Edges of domain
            p_P[0, 1:-1,1:-1,:] = (1.0/6.0)*( (p_OLD_P[1,1:-1,1:-1,:] + p_OLD_P[-1,1:-1,1:-1,:])  + (p_OLD_P[0,2:,1:-1,:] + p_OLD_P[0,0:-2,1:-1,:] )  +  (p_OLD_P[0,1:-1,2:,:] + p_OLD_P[0,1:-1,0:-2,:])      - Source_term[0,1:-1,1:-1,:])
            p_P[-1, 1:-1,1:-1,:] = (1.0/6.0)*( (p_OLD_P[0,1:-1,1:-1,:] + p_OLD_P[-2,1:-1,1:-1,:])  + (p_OLD_P[-1,2:,1:-1,:] + p_OLD_P[-1,0:-2,1:-1,:] )  +  (p_OLD_P[-1,1:-1,2:,:] + p_OLD_P[-1,1:-1,0:-2,:])      - Source_term[-1,1:-1,1:-1,:])  
            
            p_P[1:-1, 0,1:-1,:] = (1.0/6.0)*( p_OLD_P[2:,0,1:-1,:] + p_OLD_P[0:-2,0,1:-1,:] + p_OLD_P[1:-1,1,1:-1,:] + p_OLD_P[1:-1,-1,1:-1,:]  + p_OLD_P[1:-1,0,2:,:] + p_OLD_P[1:-1,0,0:-2,:]      - Source_term[1:-1,0,1:-1,:])
            p_P[1:-1, -1,1:-1,:] = (1.0/6.0)*( p_OLD_P[2:,-1,1:-1,:] + p_OLD_P[0:-2,-1,1:-1,:] + p_OLD_P[1:-1,0,1:-1,:] + p_OLD_P[1:-1,-2,1:-1,:]  + p_OLD_P[1:-1,-1,2:,:] + p_OLD_P[1:-1,-1,0:-2,:]      - Source_term[1:-1,-1,1:-1,:])            
            
            p_P[1:-1, 1:-1,0,:] = (1.0/6.0)*( p_OLD_P[2:,1:-1,0,:] + p_OLD_P[0:-2,1:-1,0,:]+ p_OLD_P[1:-1,2:,0,:] + p_OLD_P[1:-1,0:-2,0,:] + p_OLD_P[1:-1,1:-1,1,:] + p_OLD_P[1:-1,1:-1,-1,:]  - Source_term[1:-1,1:-1,0,:]) 
            p_P[1:-1, 1:-1,-1,:] = (1.0/6.0)*( p_OLD_P[2:,1:-1,-1,:] + p_OLD_P[0:-2,1:-1,-1,:]+ p_OLD_P[1:-1,2:,-1,:] + p_OLD_P[1:-1,0:-2,-1,:] + p_OLD_P[1:-1,1:-1,0,:] + p_OLD_P[1:-1,1:-1,-2,:]  - Source_term[1:-1,1:-1,-1,:])            
            
            #Corners of domain
            p_P[0, 0,0,:] = (1.0/6.0)*( (p_OLD_P[1,0,0,:] + p_OLD_P[-1,0,0,:])  + (p_OLD_P[0,1,0,:] + p_OLD_P[0,-1,0,:] )  +  (p_OLD_P[0,0,1,:] + p_OLD_P[0,0,-1,:]) - Source_term[0,0,0,:])
            p_P[0, 0,-1,:] = (1.0/6.0)*( (p_OLD_P[1,0,-1,:] + p_OLD_P[-1,0,-1,:])  + (p_OLD_P[0,1,-1,:] + p_OLD_P[0,-1,-1,:] )  +  (p_OLD_P[0,0,0,:] + p_OLD_P[0,0,-2,:]) - Source_term[0,0,-1,:])    
            
            p_P[0, -1,0,:] = (1.0/6.0)*( (p_OLD_P[1,-1,0,:] + p_OLD_P[-1,-1,0,:])  + (p_OLD_P[0,0,0,:] + p_OLD_P[0,-2,0,:] )  +  (p_OLD_P[0,-1,1,:] + p_OLD_P[0,-1,-1,:]) - Source_term[0,-1,0,:])     
            p_P[0, -1,-1,:] = (1.0/6.0)*( (p_OLD_P[1,-1,-1,:] + p_OLD_P[-1,-1,-1,:])  + (p_OLD_P[0,0,-1,:] + p_OLD_P[0,-2,-1,:] )  +  (p_OLD_P[0,-1,0,:] + p_OLD_P[0,-1,-2,:]) - Source_term[0,-1,-1,:])    
            
            
            p_P[-1, 0,0,:] = (1.0/6.0)*( (p_OLD_P[0,0,0,:] + p_OLD_P[-2,0,0,:])  + (p_OLD_P[-1,1,0,:] + p_OLD_P[-1,-1,0,:] )  +  (p_OLD_P[-1,0,1,:] + p_OLD_P[-1,0,-1,:]) - Source_term[-1,0,0,:])     
            p_P[-1, 0,-1,:] = (1.0/6.0)*( (p_OLD_P[0,0,-1,:] + p_OLD_P[-2,0,-1,:])  + (p_OLD_P[-1,1,-1,:] + p_OLD_P[-1,-1,-1,:] )  +  (p_OLD_P[-1,0,0,:] + p_OLD_P[-1,0,-2,:]) - Source_term[-1,0,-1,:])              
            
            #Need to still work on this...!
            p_P[-1, -1,0,:] = (1.0/6.0)*( (p_OLD_P[0,-1,0,:] + p_OLD_P[-2,-1,0,:])  + (p_OLD_P[-1,0,0,:] + p_OLD_P[-1,-2,0,:] )  +  (p_OLD_P[-1,-1,1,:] + p_OLD_P[-1,-1,-1,:]) - Source_term[-1,-1,0,:])     
            p_P[-1, -1,-1,:] = (1.0/6.0)*( (p_OLD_P[0,-1,-1,:] + p_OLD_P[-2,-1,-1,:])  + (p_OLD_P[-1,0,-1,:] + p_OLD_P[-1,-2,-1,:] )  +  (p_OLD_P[-1,-1,0,:] + p_OLD_P[-1,-1,-2,:]) - Source_term[-1,-1,-1,:])   
            

        
        #pressure gradients
        p_x_P[1:-1,:,:,:] = (p_P[2:,:, :,:] - p_P[0:-2,:,:,:])/(2*dx)
        p_x_P[-1,:,:,:] = (p_P[0,:, :,:] - p_P[-2,:,:,:])/(2*dx)
        p_x_P[0,:,:,:] = (p_P[1,:, :,:] - p_P[-1,:,:,:])/(2*dx)
        
        p_y_P[:,1:-1,:,:] = (p_P[:,2:,:,:] - p_P[:,0:-2,:,:])/(2*dx)
        p_y_P[:,-1,:,:] = (p_P[:,0,:,:] - p_P[:,-2,:,:])/(2*dx)
        p_y_P[:,0,:,:] = (p_P[:,1,:,:] - p_P[:,-1,:,:])/(2*dx)
        
        p_z_P[:,:,1:-1,:] = (p_P[:,:,2:,:] - p_P[:,:,0:-2,:])/(2*dx)
        p_z_P[:,:,-1,:] = (p_P[:,:,0,:] - p_P[:,:,-2,:])/(2*dx)
        p_z_P[:,:,0,:] = (p_P[:,:,1,:] - p_P[:,:,-1,:])/(2*dx)
       
        
        #Update velocities using updated pressure...
        u_P = u_P - p_x_P*(1.0/rho)
        v_P = v_P - p_y_P*(1.0/rho)
        w_P = w_P - p_z_P*(1.0/rho)
        

        #Diff = u_P - Variable( torch.FloatTensor( np.matrix( u_DNS_downsamples[T_factor*(i+1),:]).T ) )
    Diff = u_P - target_P
    Loss_i = torch.mean( torch.abs( Diff ) )
    Loss = Loss + Loss_i
    
    #Loss_np = Loss.cpu().numpy()
    
    time2 = time.time()
    time_elapsed = time2 - time1
    
    test = torch.mean( u_P)
    
    error = np.mean(np.abs( u_P.cpu().numpy()[:,:,:,0:1] -  target_P.cpu().numpy()[:,:,:,0:1] ) )
    
    print(iterations, test, error,  time_elapsed)
