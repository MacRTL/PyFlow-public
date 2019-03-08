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
import state
import dataReader as dr
import constants as const
#
sys.path.append("../metric")
import metric_collocated as metric
#
sys.path.append("../solver")
import velocity
#
sys.path.append("../sfsmodel")
import sfsmodel_nn
import sfsmodel_smagorinsky


####### TODO
#  1. Staggered grid
#  2. Pressure correction
#  3. RK3
#  4. Krylov


# ----------------------------------------------------
# User-specified parameters
# ----------------------------------------------------

# Simulation geometry
#   Options: restart, periodicGaussian
configName = "periodicGaussian"
configNx   = 64
configNy   = 64
configNz   = 64
configLx   = 1.0
configLy   = 1.0
configLz   = 1.0

# Restart and target files to read
if (configName=='restart'):
    restartFileStr = '../../examples/filtered_vol_dnsbox_1024_Lx0.045_NR_00000020_coarse'

# Model constants
mu  = 0.0 #1.8678e-5
rho = 1.2

# Time step
simDt    = 2.5e-3
numIt    = 200
stopTime = numIt*simDt

# SFS model
SFSModel = False

# Solver settings
#   Options: Euler, RK4
solverName   = "RK4"
genericOrder = 2
Num_pressure_iterations = 10

# Comparison options
useTargetData = False

if (useTargetData):
    targetFileStr  = restartFileStr


#restartFileStr = '/projects/sciteam/baxh/Downsampled_Data_Folder/data_1024_Lx0.045_NR_Delta32_downsample8x/filtered_vol_dnsbox_1024_Lx0.045_NR_00000020_coarse'
#targetFileStr  = '/projects/sciteam/baxh/Downsampled_Data_Folder/data_1024_Lx0.045_NR_Delta32_downsample8x/filtered_vol_dnsbox_1024_Lx0.045_NR_00000030_coarse'


# ----------------------------------------------------
# Configure simulation
# ----------------------------------------------------
# Set up the initial state
if (configName=='restart'):
    # Read grid and state data from the restart file
    xGrid,yGrid,zGrid,names,dataTime,data = dr.readNGA(restartFileStr)
    data_IC = data[:,:,:,0:4]
    
    # Clean up
    del data
    
elif (configName=='periodicGaussian'):
    # Initialize the uniform grid
    xGrid = np.linspace(-0.5*configLx,0.5*configLx,configNx)
    yGrid = np.linspace(-0.5*configLy,0.5*configLy,configNy)
    zGrid = np.linspace(-0.5*configLz,0.5*configLz,configNz)
    
    # Initial condition
    uMax = 2.0
    vMax = 2.0
    wMax = 2.0
    stdDev = 0.1
    gaussianBump = ( np.exp(  -0.5*(xGrid[:,np.newaxis,np.newaxis]/stdDev)**2)
                     * np.exp(-0.5*(yGrid[np.newaxis,:,np.newaxis]/stdDev)**2)
                     * np.exp(-0.5*(zGrid[np.newaxis,np.newaxis,:]/stdDev)**2) )
    data_IC = np.zeros((configNx,configNy,configNz,4),dtype='float64')
    data_IC[:,:,:,0] = uMax * gaussianBump
    data_IC[:,:,:,1] = vMax * gaussianBump
    data_IC[:,:,:,2] = wMax * gaussianBump
    del gaussianBump

    
# Set the simulation grid parameters
#   Currently configured for uniform grid
#   [JFM] move this to its own module based on dataClasses
#
Nx = len(xGrid); Lx = xGrid[-1]-xGrid[0]; dx = Lx/float(Nx)
Ny = len(yGrid); Ly = yGrid[-1]-yGrid[0]; dy = Ly/float(Ny)
Nz = len(zGrid); Lz = zGrid[-1]-zGrid[0]; dz = Lz/float(Nz)


# Set up Torch state
haveCuda = torch.cuda.is_available()

# Read target data if requested
if (useTargetData):
    # Read target data file
    xGrid_t,yGrid_t,zGrid_t,names_t,dataTime_t,data_t = dr.readNGA(targetFileStr)
    # Just save U for now
    data_target10 = data_t[:,:,:,0]
    
    # Clean up
    del data_t
    
    if (haveCuda):
        x_max_P = torch.FloatTensor( const.x_max16 ).cuda()
        target_P = torch.FloatTensor( data_target10 ).cuda()
    else:
        x_max_P = torch.FloatTensor( const.x_max16 )
        target_P = torch.FloatTensor( data_target10 )
        
    # Clean up
    del data_target10



# ----------------------------------------------------
# Set up initial condition
# ----------------------------------------------------
IC_u_np = data_IC[:,:,:,0]
IC_v_np = data_IC[:,:,:,1]
IC_w_np = data_IC[:,:,:,2]
IC_p_np = data_IC[:,:,:,3]

IC_zeros_np = np.zeros( (Nx,Ny,Nz) )


    
# ----------------------------------------------------
# Initialize closure model(s)
# ----------------------------------------------------
# Initialize the neural network closure model
model = sfsmodel_nn.ClosureModel2()
model_name = 'LES_model_NR_March2019'
#model.load_state_dict(torch.load(model_name))
#if (haveCuda):
    #model.cuda()


# ----------------------------------------------------
# Allocate memory for state data
# ----------------------------------------------------

# Allocate state data using PyTorch variables
state_u_P = state.data_P(IC_u_np,IC_zeros_np)
state_v_P = state.data_P(IC_v_np,IC_zeros_np)
state_w_P = state.data_P(IC_w_np,IC_zeros_np)

# Need a temporary velocity state for RK solvers
if (solverName[:-1]=="RK"):
    state_uTmp_P = state.data_P(IC_u_np,IC_zeros_np)
    state_vTmp_P = state.data_P(IC_v_np,IC_zeros_np)
    state_wTmp_P = state.data_P(IC_w_np,IC_zeros_np)

# Allocate pressure arrays
if (haveCuda):
    Closure_u_P =  torch.FloatTensor( IC_zeros_np ).cuda()
    Closure_v_P = torch.FloatTensor( IC_zeros_np ).cuda()
    Closure_w_P = torch.FloatTensor( IC_zeros_np ).cuda()    
    
    p_P =  torch.FloatTensor( IC_zeros_np ).cuda()
    p_OLD_P =  torch.FloatTensor( IC_zeros_np).cuda()
else:
    Closure_u_P =  torch.FloatTensor( IC_zeros_np )
    Closure_v_P = torch.FloatTensor( IC_zeros_np )
    Closure_w_P = torch.FloatTensor( IC_zeros_np )    
    
    p_P =  torch.FloatTensor( IC_zeros_np )
    p_OLD_P =  torch.FloatTensor( IC_zeros_np)
    p_x_P =  torch.FloatTensor( IC_zeros_np )
    p_y_P =  torch.FloatTensor( IC_zeros_np )
    p_z_P =  torch.FloatTensor( IC_zeros_np )

# Clean up
del IC_u_np
del IC_v_np
del IC_w_np
del IC_zeros_np
del data_IC


# ----------------------------------------------------
# Simulation loop
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

    # Iteration counter and simulation time
    itCount = 0
    simTime = 0.0

    # Write the stdout header
    headStr = "  {:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}"
    print(headStr.format("Step","Time","max CFL","max U","max V","max W"))
    
    # Write initial condition stats
    maxU = torch.max(state_u_P.var)
    maxV = torch.max(state_v_P.var)
    maxW = torch.max(state_w_P.var)
    maxCFL = max((maxU/dx,maxV/dy,maxW/dz))*simDt
    lineStr = "  {:10d}\t{:10.6E}\t{:10.6E}\t{:10.6E}\t{:10.6E}\t{:10.6E}"
    print(lineStr.format(itCount,simTime,maxCFL,maxU,maxV,maxW))

    # Plot the initial state
    dr.plotData(state_u_P.var[:,:,int(Nz/2)].cpu().numpy(),"state_U_"+str(itCount))
    dr.plotData(state_v_P.var[:,:,int(Nz/2)].cpu().numpy(),"state_V_"+str(itCount))
    dr.plotData(state_w_P.var[:,:,int(Nz/2)].cpu().numpy(),"state_W_"+str(itCount))

    # Compute the initial energy
    initEnergy = torch.sum(state_u_P.var**2 + state_v_P.var**2 + state_w_P.var**2)
    
    #Chorin's projection method for solution of incompressible Navier-Stokes PDE with periodic boundary conditions in a box.
    
    # Main iteration loop
    while (simTime < stopTime):

        # [JFM] need new sub-iteration loop
        
        # ----------------------------------------------------
        # Velocity prediction step
        # ----------------------------------------------------
        
        #model_output = model( state_u_P.var)
        #model_output2 = model_output.cpu()

        # Do we use an SFS model?
        if (SFSModel):
            Closure_u_P, Closure_v_P, Closure_w_P = sfsmodel_smagorinsky.eval(dx, u_x_P, u_y_P, u_z_P, v_x_P, v_y_P,
                                                                              v_z_P, w_x_P, w_y_P, w_z_P,
                                                                              Closure_u_P, Closure_v_P, Closure_w_P)
        else:
            Closure_u_P = 0.0
            Closure_v_P = 0.0
            Closure_w_P = 0.0
        
        # Compute velocity prediction
        if (solverName=="Euler"):
            # rhs
            rhs_u, rhs_v, rhs_w = velocity.rhs_predictor(state_u_P,state_v_P,state_w_P,uMax,vMax,wMax,mu,rho,genericOrder,dx)

            # Update the state using explicit Euler
            state_u_P.var = state_u_P.var + ( rhs_u - Closure_u_P )*simDt
            state_v_P.var = state_v_P.var + ( rhs_v - Closure_v_P )*simDt
            state_w_P.var = state_w_P.var + ( rhs_w - Closure_w_P )*simDt

        elif (solverName=="RK4"):
            #
            # [JFM] needs turbulence models
            
            # Stage 1
            rhs_u1, rhs_v1, rhs_w1 = velocity.rhs_predictor(state_u_P,state_v_P,state_w_P,uMax,vMax,wMax,mu,rho,genericOrder,dx)

            # Stage 2
            state_uTmp_P.ZAXPY(0.5*simDt,rhs_u1,state_u_P.var)
            state_vTmp_P.ZAXPY(0.5*simDt,rhs_v1,state_v_P.var)
            state_wTmp_P.ZAXPY(0.5*simDt,rhs_w1,state_w_P.var)
            rhs_u2, rhs_v2, rhs_w2 = velocity.rhs_predictor(state_uTmp_P,state_vTmp_P,state_wTmp_P,uMax,vMax,wMax,mu,rho,genericOrder,dx)

            # Stage 3
            state_uTmp_P.ZAXPY(0.5*simDt,rhs_u2,state_u_P.var)
            state_vTmp_P.ZAXPY(0.5*simDt,rhs_v2,state_v_P.var)
            state_wTmp_P.ZAXPY(0.5*simDt,rhs_w2,state_w_P.var)
            rhs_u3, rhs_v3, rhs_w3 = velocity.rhs_predictor(state_uTmp_P,state_vTmp_P,state_wTmp_P,uMax,vMax,wMax,mu,rho,genericOrder,dx)

            # Stage 4
            state_uTmp_P.ZAXPY(simDt,rhs_u3,state_u_P.var)
            state_vTmp_P.ZAXPY(simDt,rhs_v3,state_v_P.var)
            state_wTmp_P.ZAXPY(simDt,rhs_w3,state_w_P.var)
            rhs_u4, rhs_v4, rhs_w4 = velocity.rhs_predictor(state_uTmp_P,state_vTmp_P,state_wTmp_P,uMax,vMax,wMax,mu,rho,genericOrder,dx)

            # Update the state
            state_u_P.var = state_u_P.var + simDt/6.0*( rhs_u1 + 2.0*rhs_u2 + 2.0*rhs_u3 + rhs_u4 )
            state_v_P.var = state_v_P.var + simDt/6.0*( rhs_v1 + 2.0*rhs_v2 + 2.0*rhs_v3 + rhs_v4 )
            state_w_P.var = state_w_P.var + simDt/6.0*( rhs_w1 + 2.0*rhs_w2 + 2.0*rhs_w3 + rhs_w4 )
            
        # Update periodic BCs
        #u_P[-1,:,:] = u_P[0,:,:]; v_P[-1,:,:] = v_P[0,:,:]; w_P[-1,:,:] = w_P[0,:,:]
        #u_P[:,-1,:] = u_P[:,0,:]; v_P[:,-1,:] = v_P[:,0,:]; w_P[:,-1,:] = w_P[:,0,:]
        #u_P[:,:,-1] = u_P[:,:,0]; v_P[:,:,-1] = v_P[:,:,0]; w_P[:,:,-1] = w_P[:,:,0]

        
        # ----------------------------------------------------
        # Pressure Poisson equation
        # ----------------------------------------------------

        # [JFM] move this to its own module
        
        Source_term = rho*dx*dx*(state_u_P.grad_x + state_v_P.grad_y + state_w_P.grad_z)
        
        for j in range( Num_pressure_iterations ):
            
            p_OLD_P[:,:,:] = p_P[0:,0:,0:]
            
            Pressure_update_j1 = p_OLD_P[2:,1:-1,1:-1] + p_OLD_P[0:-2,1:-1,1:-1]
                 
            
            Pressure_update_j2 = p_OLD_P[1:-1,2:,1:-1] + p_OLD_P[1:-1,0:-2,1:-1]
            
            Pressure_update_j3 = p_OLD_P[1:-1,1:-1,2:] + p_OLD_P[1:-1,1:-1,0:-2]

            #interior of domain
            p_P[1:-1, 1:-1,1:-1] = (1.0/6.0)*( Pressure_update_j1 + Pressure_update_j2
                                               + Pressure_update_j3 - Source_term[1:-1,1:-1,1:-1])
            
            #Edges of domain
            p_P[0, 1:-1,1:-1] = (1.0/6.0)*( p_OLD_P[1,1:-1,1:-1] + p_OLD_P[-1,1:-1,1:-1] 
                                            + p_OLD_P[0,2:,1:-1] + p_OLD_P[0,0:-2,1:-1]  
                                            + p_OLD_P[0,1:-1,2:] + p_OLD_P[0,1:-1,0:-2]  
                                            - Source_term[0,1:-1,1:-1])
            
            p_P[-1, 1:-1,1:-1] = (1.0/6.0)*( p_OLD_P[0,1:-1,1:-1] + p_OLD_P[-2,1:-1,1:-1]
                                             + p_OLD_P[-1,2:,1:-1]+ p_OLD_P[-1,0:-2,1:-1] 
                                             + p_OLD_P[-1,1:-1,2:] + p_OLD_P[-1,1:-1,0:-2]
                                             - Source_term[-1,1:-1,1:-1])  
            
            p_P[1:-1, 0,1:-1] = (1.0/6.0)*( p_OLD_P[2:,0,1:-1] + p_OLD_P[0:-2,0,1:-1]
                                            + p_OLD_P[1:-1,1,1:-1] + p_OLD_P[1:-1,-1,1:-1]
                                            + p_OLD_P[1:-1,0,2:] + p_OLD_P[1:-1,0,0:-2]
                                            - Source_term[1:-1,0,1:-1])
            
            p_P[1:-1, -1,1:-1] = (1.0/6.0)*( p_OLD_P[2:,-1,1:-1] + p_OLD_P[0:-2,-1,1:-1]
                                             + p_OLD_P[1:-1,0,1:-1] + p_OLD_P[1:-1,-2,1:-1]
                                             + p_OLD_P[1:-1,-1,2:] + p_OLD_P[1:-1,-1,0:-2]
                                             - Source_term[1:-1,-1,1:-1])            
            
            p_P[1:-1, 1:-1,0] = (1.0/6.0)*( p_OLD_P[2:,1:-1,0] + p_OLD_P[0:-2,1:-1,0]
                                            + p_OLD_P[1:-1,2:,0] + p_OLD_P[1:-1,0:-2,0]
                                            + p_OLD_P[1:-1,1:-1,1] + p_OLD_P[1:-1,1:-1,-1]
                                            - Source_term[1:-1,1:-1,0])
            
            p_P[1:-1, 1:-1,-1] = (1.0/6.0)*( p_OLD_P[2:,1:-1,-1] + p_OLD_P[0:-2,1:-1,-1]
                                             + p_OLD_P[1:-1,2:,-1] + p_OLD_P[1:-1,0:-2,-1]
                                             + p_OLD_P[1:-1,1:-1,0] + p_OLD_P[1:-1,1:-1,-2]
                                             - Source_term[1:-1,1:-1,-1])            
            
            #Corners of domain
            p_P[0, 0,0] = (1.0/6.0)*( p_OLD_P[1,0,0] + p_OLD_P[-1,0,0]
                                      + p_OLD_P[0,1,0] + p_OLD_P[0,-1,0] 
                                      + p_OLD_P[0,0,1] + p_OLD_P[0,0,-1]
                                      - Source_term[0,0,0] )
            
            p_P[0, 0,-1] = (1.0/6.0)*( p_OLD_P[1,0,-1] + p_OLD_P[-1,0,-1]
                                       + p_OLD_P[0,1,-1] + p_OLD_P[0,-1,-1] 
                                       + p_OLD_P[0,0,0] + p_OLD_P[0,0,-2]
                                       - Source_term[0,0,-1])    
            
            p_P[0, -1,0] = (1.0/6.0)*( p_OLD_P[1,-1,0] + p_OLD_P[-1,-1,0]
                                       + p_OLD_P[0,0,0] + p_OLD_P[0,-2,0] 
                                       + p_OLD_P[0,-1,1] + p_OLD_P[0,-1,-1]
                                       - Source_term[0,-1,0])
            
            p_P[0, -1,-1] = (1.0/6.0)*( p_OLD_P[1,-1,-1] + p_OLD_P[-1,-1,-1]
                                        + p_OLD_P[0,0,-1] + p_OLD_P[0,-2,-1] 
                                        + p_OLD_P[0,-1,0] + p_OLD_P[0,-1,-2]
                                        - Source_term[0,-1,-1])    
            
            
            p_P[-1, 0,0] = (1.0/6.0)*( p_OLD_P[0,0,0] + p_OLD_P[-2,0,0]
                                       + p_OLD_P[-1,1,0] + p_OLD_P[-1,-1,0]
                                       + p_OLD_P[-1,0,1] + p_OLD_P[-1,0,-1]
                                       - Source_term[-1,0,0])
            
            p_P[-1, 0,-1] = (1.0/6.0)*( p_OLD_P[0,0,-1] + p_OLD_P[-2,0,-1]
                                        + p_OLD_P[-1,1,-1] + p_OLD_P[-1,-1,-1]
                                        + p_OLD_P[-1,0,0] + p_OLD_P[-1,0,-2]
                                        - Source_term[-1,0,-1])              
            
            #Need to still work on this...!
            p_P[-1, -1,0] = (1.0/6.0)*( p_OLD_P[0,-1,0] + p_OLD_P[-2,-1,0]
                                        + p_OLD_P[-1,0,0] + p_OLD_P[-1,-2,0] 
                                        + p_OLD_P[-1,-1,1] + p_OLD_P[-1,-1,-1]
                                        - Source_term[-1,-1,0])
            
            p_P[-1, -1,-1] = (1.0/6.0)*( p_OLD_P[0,-1,-1] + p_OLD_P[-2,-1,-1]
                                         + p_OLD_P[-1,0,-1] + p_OLD_P[-1,-2,-1]
                                         + p_OLD_P[-1,-1,0] + p_OLD_P[-1,-1,-2]
                                         - Source_term[-1,-1,-1])   

            
        #pressure gradients
        p_x_P[1:-1,:,:] = (p_P[2:,:, :] - p_P[0:-2,:,:])/(2*dx)
        p_x_P[-1,:,:] = (p_P[0,:, :] - p_P[-2,:,:])/(2*dx)
        p_x_P[0,:,:] = (p_P[1,:, :] - p_P[-1,:,:])/(2*dx)
        
        p_y_P[:,1:-1,:] = (p_P[:,2:,:] - p_P[:,0:-2,:])/(2*dx)
        p_y_P[:,-1,:] = (p_P[:,0,:] - p_P[:,-2,:])/(2*dx)
        p_y_P[:,0,:] = (p_P[:,1,:] - p_P[:,-1,:])/(2*dx)
        
        p_z_P[:,:,1:-1] = (p_P[:,:,2:] - p_P[:,:,0:-2])/(2*dx)
        p_z_P[:,:,-1] = (p_P[:,:,0] - p_P[:,:,-2])/(2*dx)
        p_z_P[:,:,0] = (p_P[:,:,1] - p_P[:,:,-1])/(2*dx)
       
        
        # ----------------------------------------------------
        # Velocity correction step
        # ----------------------------------------------------
        #u_P = u_P - p_x_P/rho
        #v_P = v_P - p_y_P/rho
        #w_P = w_P - p_z_P/rho

        
        # ----------------------------------------------------
        # Post-step
        # ----------------------------------------------------
        # Compute stats
        maxU = torch.max(state_u_P.var)
        maxV = torch.max(state_v_P.var)
        maxW = torch.max(state_w_P.var)
        maxCFL = max((maxU/dx,maxV/dy,maxW/dz))*simDt

        # Update the time
        itCount += 1
        simTime += simDt
        
        # Write stats
        lineStr = "  {:10d}\t{:10.6E}\t{:10.6E}\t{:10.6E}\t{:10.6E}\t{:10.6E}"
        print(lineStr.format(itCount,simTime,maxCFL,maxU,maxV,maxW))

        # Update the figures
        #dr.plotData(state_u_P.var[:,:,int(Nz/2)].cpu().numpy(),"state_U_"+str(itCount))
        #dr.plotData(state_v_P.var[:,:,int(Nz/2)].cpu().numpy(),"state_V_"+str(itCount))
        #dr.plotData(state_w_P.var[:,:,int(Nz/2)].cpu().numpy(),"state_W_"+str(itCount))

        ## END OF ITERATION LOOP




        
    # ----------------------------------------------------
    # Post-simulation tasks
    # ----------------------------------------------------
        #Diff = state_u_P.var - Variable( torch.FloatTensor( np.matrix( u_DNS_downsamples[T_factor*(i+1)]).T ) )
    if (useTargetData):
        Diff = state_u_P.var - target_P
        Loss_i = torch.mean( torch.abs( Diff ) )
        Loss = Loss + Loss_i
        error = np.mean(np.abs( state_u_P.var.cpu().numpy() -  target_P.cpu().numpy() ) )
    
    #Loss_np = Loss.cpu().numpy()
    
    time2 = time.time()
    time_elapsed = time2 - time1
    
    test = torch.mean( state_u_P.var)
        
    # Compute the final energy
    finalEnergy = torch.sum(state_u_P.var**2 + state_v_P.var**2 + state_w_P.var**2)
    
    if (useTargetData):
        print(iterations,test,error,time_elapsed)
    else:
        print("it={}, test={}, elapsed={}, energy init/final={}".format(iterations,test,time_elapsed,
                                                                        initEnergy/finalEnergy))

    # Print a pretty picture
    dr.plotData(state_u_P.var[:,:,int(Nz/2)].cpu().numpy(),"state_U_"+str(itCount))
    dr.plotData(state_v_P.var[:,:,int(Nz/2)].cpu().numpy(),"state_V_"+str(itCount))
    dr.plotData(state_w_P.var[:,:,int(Nz/2)].cpu().numpy(),"state_W_"+str(itCount))
