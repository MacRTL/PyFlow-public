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
sys.path.append("../geometry")
import geometry as geo
#
sys.path.append("../metric")
import metric_staggered
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
#   Options: restart, periodicGaussian, uniform, notAchannel
configName = "restart"
#configName = "periodicGaussian"
#configName = "uniform"
#configLx   = 1.0
#configLy   = 1.0
#configLz   = 1.0
#configName = "notAchannel"
configNx   = 256
configNy   = 128
configNz   = 256
configLx   = 0.008
configLy   = 0.004
configLz   = 0.008

# Restart and target files to read
if (configName=='restart'):
    #dataFileStr = '../../examples/filtered_vol_dnsbox_1024_Lx0.045_NR_00000020_coarse'
    #dataFileType = 'volume'
    
    configFileStr = '../../examples/config_dnsbox_128_Lx0.0056'
    dataFileStr   = '../../examples/data_dnsbox_128_Lx0.0056.1_2.50000E-04'
    dataFileType  = 'restart'

# Model constants
mu  = 1.8678e-5
rho = 1.2

# Time step
simDt    = 2.5e-6
numIt    = 50
numItOut = 2
startTime= 0.0

# SFS model
SFSModel = False

# Solver settings
#   solverName options:   Euler, RK4
#   equationMode options: scalar, NS
solverName   = "RK4"
equationMode = "NS"
genericOrder = 2
plotState    = True
precision    = torch.float32
Num_pressure_iterations = 500

# Comparison options
useTargetData = False
if (useTargetData):
    targetFileStr  = restartFileStr



# ----------------------------------------------------
# Configure initial conditions
# ----------------------------------------------------
if (configName=='restart'):
    if (dataFileType=='restart'):
        # Read grid data from an NGA config file
        xGrid,yGrid,zGrid,xper,yper,zper = dr.readNGAconfig(configFileStr)
        configNx = len(xGrid)-1
        configNy = len(yGrid)-1
        configNz = len(zGrid)-1
        
        # Read state data from an NGA restart file
        names,startTime,data = dr.readNGArestart(dataFileStr)

    elif (dataFileType=='volume'):
        # Read grid and state data from a volume-format file
        xGrid,yGrid,zGrid,names,dataTime,data = dr.readNGA(dataFileStr)
        configNx = len(xGrid)
        configNy = len(yGrid)
        configNz = len(zGrid)

        # Interpolate grid and state data to cell faces
        # [JFM] NEED TO IMPLEMENT

    # Extract the initial conditions
    data_IC = data[:,:,:,0:4]
    
    # Clean up
    del data
    
elif (configName=='periodicGaussian'):
    # Initialize the uniform grid
    xGrid = np.linspace(-0.5*configLx,0.5*configLx,configNx+1)
    yGrid = np.linspace(-0.5*configLy,0.5*configLy,configNy+1)
    zGrid = np.linspace(-0.5*configLz,0.5*configLz,configNz+1)
    
    # Initial condition
    uMax = 2.0
    vMax = 2.0
    wMax = 2.0
    stdDev = 0.1
    gaussianBump = ( np.exp(  -0.5*(xGrid[:-1,np.newaxis,np.newaxis]/stdDev)**2)
                     * np.exp(-0.5*(yGrid[np.newaxis,:-1,np.newaxis]/stdDev)**2)
                     * np.exp(-0.5*(zGrid[np.newaxis,np.newaxis,:-1]/stdDev)**2) )
    data_IC = np.zeros((configNx,configNy,configNz,4),dtype='float64')
    data_IC[:,:,:,0] = uMax * gaussianBump
    data_IC[:,:,:,1] = vMax * gaussianBump
    data_IC[:,:,:,2] = wMax * gaussianBump
    data_IC[:,:,:,3] = 0.0
    del gaussianBump

elif (configName=='uniform'):
    # Uniform flow
    # Initialize the uniform grid
    xGrid = np.linspace(-0.5*configLx,0.5*configLx,configNx+1)
    yGrid = np.linspace(-0.5*configLy,0.5*configLy,configNy+1)
    zGrid = np.linspace(-0.5*configLz,0.5*configLz,configNz+1)
    
    uMax = 2.0
    vMax = 2.0
    wMax = 2.0
    data_IC = np.zeros((configNx,configNy,configNz,4),dtype='float64')
    data_IC[:,:,:,0] = uMax
    data_IC[:,:,:,1] = vMax
    data_IC[:,:,:,2] = wMax
    data_IC[:,:,:,3] = 0.0

elif (configName=='notAchannel'):
    # Not a channel; periodic BCs at +/- y; no walls
    # Initialize the uniform grid
    xGrid = np.linspace(-0.5*configLx,0.5*configLx,configNx+1)
    yGrid = np.linspace(-0.5*configLy,0.5*configLy,configNy+1)
    zGrid = np.linspace(-0.5*configLz,0.5*configLz,configNz+1)
    
    uMax = 2.0
    vMax = 0.0
    wMax = 0.0
    amp  = 0.4
    print("Bulk Re={:7f}".format(rho*uMax*configLy/mu))
    parabolaX = ( 6.0*(yGrid[np.newaxis,:-1,np.newaxis] + 0.5*configLy)
                  *(0.5*configLy - yGrid[np.newaxis,:-1,np.newaxis])
                  /configLy**2 )
    Unorm = np.sqrt(uMax**2 + wMax**2)
    data_IC = np.zeros((configNx,configNy,configNz,4),dtype='float64')
    data_IC[:,:,:,0] = (uMax * parabolaX
                        + amp*Unorm*np.cos(16.0*3.1415926*xGrid[:-1,np.newaxis,np.newaxis]/configLx))
    data_IC[:,:,:,1] = 0.0
    data_IC[:,:,:,2] = (wMax * parabolaX
                        + amp*Unorm*np.cos(16.0*3.1415926*zGrid[np.newaxis,np.newaxis,:-1]/configLz))
    data_IC[:,:,:,3] = 0.0
    del parabolaX



# ----------------------------------------------------
# Configure PyTorch
# ----------------------------------------------------

# Offload to GPUs if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    
# ----------------------------------------------------
# Configure simulation
# ----------------------------------------------------

# Initialize the geometry
#   --> Geometry object also provides parallel decomp/offload settings
geometry = geo.uniform(xGrid,yGrid,zGrid,device,precision)

# Local grid sizes
nx_ = geometry.nx_
ny_ = geometry.ny_
nz_ = geometry.nz_
imin_ = geometry.imin_; imax_ = geometry.imax_
jmin_ = geometry.jmin_; jmax_ = geometry.jmax_
kmin_ = geometry.kmin_; kmax_ = geometry.kmax_

# Initialize the metrics
metric = metric_staggered.metric_uniform(geometry)



# ----------------------------------------------------
# Set up target data state
# ----------------------------------------------------
if (useTargetData):
    # Read target data file
    xGrid_t,yGrid_t,zGrid_t,names_t,dataTime_t,data_t = dr.readNGA(targetFileStr)
    # Just save U for now
    data_target10 = data_t[:,:,:,0]
    
    # Clean up
    del data_t
    
    x_max_P  = torch.FloatTensor( const.x_max16 ).to(device)
    target_P = torch.FloatTensor( data_target10 ).to(device)
        
    # Clean up
    del data_target10



# ----------------------------------------------------
# Set up initial condition
# ----------------------------------------------------
IC_u_np = data_IC[:,:,:,0]
IC_v_np = data_IC[:,:,:,1]
IC_w_np = data_IC[:,:,:,2]
IC_p_np = data_IC[:,:,:,3]

IC_zeros_np = np.zeros( (configNx,configNy,configNz) )
IC_ones_np  = np.ones ( (configNx,configNy,configNz) )

    
# ----------------------------------------------------
# Initialize closure model(s)
# ----------------------------------------------------
# Initialize the neural network closure model
model = sfsmodel_nn.ClosureModel2()
model_name = 'LES_model_NR_March2019'
#model.load_state_dict(torch.load(model_name))
#model.to(device)


# ----------------------------------------------------
# Allocate memory for state data
# ----------------------------------------------------

# Allocate state data using PyTorch variables
state_u_P = state.data_P(geometry,IC_u_np)
state_v_P = state.data_P(geometry,IC_v_np)
state_w_P = state.data_P(geometry,IC_w_np)
state_p_P = state.data_P(geometry,IC_p_np)
state_pOld_P = state.data_P(geometry,IC_p_np)

# Need a temporary velocity state for RK solvers
if (solverName[:-1]=="RK"):
    state_uTmp_P = state.data_P(geometry,IC_u_np)
    state_vTmp_P = state.data_P(geometry,IC_v_np)
    state_wTmp_P = state.data_P(geometry,IC_w_np)

# Allocate workspace arrays
Closure_u_P = torch.FloatTensor( IC_zeros_np ).to(device)
Closure_v_P = torch.FloatTensor( IC_zeros_np ).to(device)
Closure_w_P = torch.FloatTensor( IC_zeros_np ).to(device)    
source_P    = torch.zeros(nx_,ny_,nz_,dtype=precision).to(device)


# Allocate RHS objects
print(' ')
if (equationMode=='scalar'):
    print("Solving scalar advection-diffusion equation")
    rhs1 = velocity.rhs_scalar(geometry,uMax,vMax,wMax)
    if (solverName[:-1]=="RK"):
        rhs2 = velocity.rhs_scalar(geometry,uMax,vMax,wMax)
        rhs3 = velocity.rhs_scalar(geometry,uMax,vMax,wMax)
        rhs4 = velocity.rhs_scalar(geometry,uMax,vMax,wMax)
elif (equationMode=='NS'):
    print("Solving Navier-Stokes equations")
    rhs1 = velocity.rhs_NavierStokes(geometry)
    if (solverName[:-1]=="RK"):
        rhs2 = velocity.rhs_NavierStokes(geometry)
        rhs3 = velocity.rhs_NavierStokes(geometry)
        rhs4 = velocity.rhs_NavierStokes(geometry)
else:
    print("Equation setting not recognized; consequences unknown...")
    
# Clean up
del IC_u_np
del IC_v_np
del IC_w_np
del IC_p_np
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
    itCount  = 0
    simTime  = startTime
    stopTime = startTime + numIt*simDt

    # Write the stdout header
    if (equationMode=='NS'):
        headStr = "  {:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}"
        print(headStr.format("Step","Time","max CFL","max U","max V","max W","int RP","max res_P"))
    else:
        headStr = "  {:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}"
        print(headStr.format("Step","Time","max CFL","max U","max V","max W"))
    
    # Write initial condition stats
    maxU = torch.max(state_u_P.var)
    maxV = torch.max(state_v_P.var)
    maxW = torch.max(state_w_P.var)
    maxCFL = max((maxU/geometry.dx,maxV/geometry.dy,maxW/geometry.dz))*simDt
    lineStr = "  {:10d}\t{:10.6E}\t{:10.6E}\t{:10.6E}\t{:10.6E}\t{:10.6E}"
    print(lineStr.format(itCount,simTime,maxCFL,maxU,maxV,maxW))

    if (plotState):
        # Plot the initial state
        dr.plotData(state_u_P.var[:,:,int(geometry.Nz/2)].cpu().numpy(),"state_U_"+str(itCount))
        dr.plotData(state_v_P.var[:,:,int(geometry.Nz/2)].cpu().numpy(),"state_V_"+str(itCount))
        dr.plotData(state_w_P.var[:,:,int(geometry.Nz/2)].cpu().numpy(),"state_W_"+str(itCount))
        dr.plotData(state_p_P.var[:,:,int(geometry.Nz/2)].cpu().numpy(),"state_p_"+str(itCount))

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
            Closure_u_P, Closure_v_P, Closure_w_P = sfsmodel_smagorinsky.eval(geometry.dx, state_u_P,state_v_P,state_w_P,
                                                                              Closure_u_P, Closure_v_P, Closure_w_P, metric)
        else:
            Closure_u_P = 0.0
            Closure_v_P = 0.0
            Closure_w_P = 0.0
        
        # Compute velocity prediction
        if (solverName=="Euler"):
            # rhs
            rhs1.evaluate(state_u_P,state_v_P,state_w_P,mu,rho,metric)

            # Update the state using explicit Euler
            state_u_P.var = state_u_P.var + ( rhs1.rhs_u - Closure_u_P )*simDt
            state_v_P.var = state_v_P.var + ( rhs1.rhs_v - Closure_v_P )*simDt
            state_w_P.var = state_w_P.var + ( rhs1.rhs_w - Closure_w_P )*simDt

        elif (solverName=="RK4"):
            
            # [JFM] needs turbulence models
            
            # Stage 1
            rhs1.evaluate(state_u_P,state_v_P,state_w_P,mu,rho,metric)
            
            # Stage 2
            state_uTmp_P.ZAXPY(0.5*simDt,rhs1.rhs_u,state_u_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1])
            state_vTmp_P.ZAXPY(0.5*simDt,rhs1.rhs_v,state_v_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1])
            state_wTmp_P.ZAXPY(0.5*simDt,rhs1.rhs_w,state_w_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1])
            rhs2.evaluate(state_uTmp_P,state_vTmp_P,state_wTmp_P,mu,rho,metric)

            # Stage 3
            state_uTmp_P.ZAXPY(0.5*simDt,rhs2.rhs_u,state_u_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1])
            state_vTmp_P.ZAXPY(0.5*simDt,rhs2.rhs_v,state_v_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1])
            state_wTmp_P.ZAXPY(0.5*simDt,rhs2.rhs_w,state_w_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1])
            rhs3.evaluate(state_uTmp_P,state_vTmp_P,state_wTmp_P,mu,rho,metric)

            # Stage 4
            state_uTmp_P.ZAXPY(simDt,rhs3.rhs_u,state_u_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1])
            state_vTmp_P.ZAXPY(simDt,rhs3.rhs_v,state_v_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1])
            state_wTmp_P.ZAXPY(simDt,rhs3.rhs_w,state_w_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1])
            rhs4.evaluate(state_uTmp_P,state_vTmp_P,state_wTmp_P,mu,rho,metric)

            # Update the state
            state_u_P.update( state_u_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1]
                              + simDt/6.0*( rhs1.rhs_u + 2.0*rhs2.rhs_u + 2.0*rhs3.rhs_u + rhs4.rhs_u ) )
            state_v_P.update( state_v_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1]
                              + simDt/6.0*( rhs1.rhs_v + 2.0*rhs2.rhs_v + 2.0*rhs3.rhs_v + rhs4.rhs_v ) )
            state_w_P.update( state_w_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1]
                              + simDt/6.0*( rhs1.rhs_w + 2.0*rhs2.rhs_w + 2.0*rhs3.rhs_w + rhs4.rhs_w ) )
            

        
        # ----------------------------------------------------
        # Pressure Poisson equation
        # ----------------------------------------------------

        # [JFM] move this to its own module
        
        # 1. Currently using Chorin's original fractional step method
        #   (essentially Lie splitting); unclear interpretation of
        #   predictor step RHS w/o pressure. Modern fractional-step
        #   (based on midpoint method) would be better.

        # 2. Boundary conditions: zero normal gradient. Note: only
        #   satisfies local mass conservation; global mass
        #   conservation needs to be enforced in open systems before
        #   solving Poisson equation, e.g., by rescaling source_P.

        if (equationMode=='NS'):
            # Divergence of the predicted velocity field
            metric.div_vel(state_u_P,state_v_P,state_w_P,source_P)
            
            # Integral of the Poisson eqn RHS
            int_RP = torch.sum(source_P)
            
            # Poisson equation source term
            source_P *= rho/simDt*geometry.dx**2
            
            #source_P = rho/simDt * (state_u_P.grad_x + state_v_P.grad_y + state_w_P.grad_z)
            #source_P = rho*geometry.dx*geometry.dx*(state_u_P.grad_x + state_v_P.grad_y + state_w_P.grad_z)
            
            # Pressure iteration residual
            max_res_P = 0.0
            
            # Matrix equation (3D)
            DInv  = -1.0/6.0
            DInvF = -0.2
            DInvC = -1.0/3.0
            #DInv  = -geometry.dx**2/6.0 # Centers - uniform grid
            #DInvF = -geometry.dx**2*0.2 # Faces - uniform grid
            #DInvC = -geometry.dx**2/3.0 # Corners - uniform grid
            #EFacX = -1.0/geometry.dx**2
            #EFacY = -1.0/geometry.dy**2
            #EFacZ = -1.0/geometry.dz**2
            #FFacX = -1.0/geometry.dx**2
            #FFacY = -1.0/geometry.dy**2
            #FFacZ = -1.0/geometry.dz**2
        
            for j in range( Num_pressure_iterations ):
                # Initial guess is p from previous time step
                state_pOld_P.update(state_p_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1])

                # [JFM] update this to the Laplacian operator
                #  --> Needed for more general solution than Jacobi iteration
                #
                # Compute gradient of the old pressure field (can just copy and save)
                #metric.grad_P(state_pOld_P)
                # Update the state
                #state_p_P.update(( -state_pOld_P.grad_x[1:,:-1,:-1] - state_pOld_P.grad_x[:-1,:-1,:-1]
                #                   -state_pOld_P.grad_y[:-1,1:,:-1] - state_pOld_P.grad_y[:-1,:-1,:-1]
                #                   -state_pOld_P.grad_z[:-1,:-1,1:] - state_pOld_P.grad_z[:-1,:-1,:-1]
                #                   +source_P )*DInv)
            
                # Jacobi iteration
                state_p_P.update(( -state_pOld_P.var[imin_+1:imax_+2,jmin_:jmax_+1,kmin_:kmax_+1]
                                   -state_pOld_P.var[imin_-1:imax_  ,jmin_:jmax_+1,kmin_:kmax_+1]
                                   -state_pOld_P.var[imin_:imax_+1,jmin_+1:jmax_+2,kmin_:kmax_+1]
                                   -state_pOld_P.var[imin_:imax_+1,jmin_-1:jmax_  ,kmin_:kmax_+1]
                                   -state_pOld_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_+1:kmax_+2]
                                   -state_pOld_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_-1:kmax_  ]
                                   +source_P )*DInv)
                
            # Compute max pressure residual
            max_res_P = torch.max(torch.abs(state_p_P.var - state_pOld_P.var))

            # Compute pressure gradients
            metric.grad_P(state_p_P)

        
            # ----------------------------------------------------
            # Velocity correction step
            # ----------------------------------------------------
            state_u_P.vel_corr(state_p_P.grad_x,simDt/rho)
            state_v_P.vel_corr(state_p_P.grad_y,simDt/rho)
            state_w_P.vel_corr(state_p_P.grad_z,simDt/rho)
                           
            #state_u_P.var = state_u_P.var - state_p_P.grad_x/rho*simDt
            #state_v_P.var = state_v_P.var - state_p_P.grad_y/rho*simDt
            #state_w_P.var = state_w_P.var - state_p_P.grad_z/rho*simDt

        
        # ----------------------------------------------------
        # Post-step
        # ----------------------------------------------------
        # Compute stats
        maxU = torch.max(state_u_P.var)
        maxV = torch.max(state_v_P.var)
        maxW = torch.max(state_w_P.var)
        maxCFL = max((maxU/geometry.dx,maxV/geometry.dy,maxW/geometry.dz))*simDt

        # Update the time
        itCount += 1
        simTime += simDt
        
        # Write stats
        if (equationMode=='NS'):
            lineStr = "  {:10d}\t{:10.6E}\t{:10.6E}\t{:10.6E}\t{:10.6E}\t{:10.6E}\t{: 10.6E}\t{: 10.6E}"
            print(lineStr.format(itCount,simTime,maxCFL,maxU,maxV,maxW,int_RP,max_res_P))
        else:
            lineStr = "  {:10d}\t{:10.6E}\t{:10.6E}\t{:10.6E}\t{:10.6E}\t{:10.6E}"
            print(lineStr.format(itCount,simTime,maxCFL,maxU,maxV,maxW))

        # Update the figures
        if (np.mod(itCount,numItOut)==0 and plotState):
            dr.plotData(state_u_P.var[:,:,int(geometry.Nz/2)].cpu().numpy(),"state_U_"+str(itCount))
            dr.plotData(state_v_P.var[:,:,int(geometry.Nz/2)].cpu().numpy(),"state_V_"+str(itCount))
            dr.plotData(state_w_P.var[:,:,int(geometry.Nz/2)].cpu().numpy(),"state_W_"+str(itCount))
            dr.plotData(state_p_P.var[:,:,int(geometry.Nz/2)].cpu().numpy(),"state_p_"+str(itCount))

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

    if (plotState):
        # Print a pretty picture
        dr.plotData(state_u_P.var[:,:,int(geometry.Nz/2)].cpu().numpy(),"state_U_"+str(itCount))
        dr.plotData(state_v_P.var[:,:,int(geometry.Nz/2)].cpu().numpy(),"state_V_"+str(itCount))
        dr.plotData(state_w_P.var[:,:,int(geometry.Nz/2)].cpu().numpy(),"state_W_"+str(itCount))
        dr.plotData(state_p_P.var[:,:,int(geometry.Nz/2)].cpu().numpy(),"state_p_"+str(itCount))
