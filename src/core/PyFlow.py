# ------------------------------------------------------------------------
#
# PyFlow: A GPU-accelerated CFD platform written in Python
#
# @authors:
#    Justin A. Sirignano
#    Jonathan F. MacArt
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
import pressure
#
sys.path.append("../sfsmodel")
import sfsmodel_nn
import sfsmodel_smagorinsky


####### TODO
#  1. Non-periodic BCs
#  2. Midpoint fractional-step
#  3. RK3
#  4. Krylov - GPU
#  5. Non-uniform grid


# ----------------------------------------------------
# User-specified parameters
# ----------------------------------------------------

# Simulation geometry
#   Options: restart, periodicGaussian, uniform, notAchannel
#configName = "restart"
configName = "periodicGaussian"
#configName = "uniform"
#configName = "notAchannel"
configNx   = 64
configNy   = 64
configNz   = 64
configLx   = 1.0
configLy   = 1.0
configLz   = 1.0
isper = [1,1,1]

# Data and config files to read
if (configName=='restart'):
    #dataFileStr = '../../examples/filtered_vol_dnsbox_1024_Lx0.045_NR_00000020_coarse'
    #dataFileType = 'volume'
    
    configFileStr = '../../examples/config_dnsbox_128_Lx0.0056'
    dataFileStr   = '../../examples/data_dnsbox_128_Lx0.0056.1_2.50000E-04'
    dataFileType  = 'restart'

# Data file to write
fNameOut     = 'data_dnsbox_128_Lx0.0056.PF'
numItDataOut = 500

# Parallel decomposition
nproc_x = 1
nproc_y = 1
nproc_z = 1

# Model constants
mu  = 1.8678e-5
rho = 1.2

# Time step info
simDt        = 2.5e-6
numIt        = 10
startTime    = 0.0

# SFS model
#   SFSmodel options: none, Smagorinsky, nn
SFSmodel = 'none' #'Smagorinsky'

# Solver settings
#   solverName options:   Euler, RK4
#   equationMode options: scalar, NS
#   pSolverMode options:  Jacobi, bicgstab
#
solverName   = "RK4"
equationMode = "NS"
genericOrder = 2
precision    = torch.float32
pSolverMode  = "Jacobi"
Num_pressure_iterations = 800
#pSolverMode  = "bicgstab"
#Num_pressure_iterations = 300

# Output options
plotState    = True
numItPlotOut = 50

# Comparison options
useTargetData = False
if (useTargetData):
    targetFileStr  = restartFileStr




# ----------------------------------------------------
# Configure PyTorch
# ----------------------------------------------------

# Offload to GPUs if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    
# ----------------------------------------------------
# Configure parallel decomposition
# ----------------------------------------------------
nproc  = [nproc_x,nproc_y,nproc_z]
decomp = geo.decomp(configNx,configNy,configNz,nproc,isper)

# Check the decomp
if (decomp.size!=nproc_x*nproc_y*nproc_z):
    raise Exception('\nNumber of processors does not match specified domain decomposition\n')

# Local grid sizes
nx_ = decomp.nx_
ny_ = decomp.ny_
nz_ = decomp.nz_
imin_ = decomp.imin_; imax_ = decomp.imax_
jmin_ = decomp.jmin_; jmax_ = decomp.jmax_
kmin_ = decomp.kmin_; kmax_ = decomp.kmax_
imin_loc = decomp.imin_loc; imax_loc = decomp.imax_loc
jmin_loc = decomp.jmin_loc; jmax_loc = decomp.jmax_loc
kmin_loc = decomp.kmin_loc; kmax_loc = decomp.kmax_loc
nxo_ = decomp.nxo_
nyo_ = decomp.nyo_
nzo_ = decomp.nzo_



# ----------------------------------------------------
# Configure initial conditions
# ----------------------------------------------------
# Default state variable names
names = ['U','V','W','P']

# Process initial condition type
if (configName=='restart'):
    if (dataFileType=='restart'):
        # Read grid data from an NGA config file
        xGrid,yGrid,zGrid,xper,yper,zper = dr.readNGAconfig(configFileStr)
        configNx = len(xGrid)-1
        configNy = len(yGrid)-1
        configNz = len(zGrid)-1
        isper = [xper,yper,zper]

        # BCAST these from rank=0
        
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
        # Need periodicity information

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
    gaussianBump = ( np.exp(  -0.5*(xGrid[imin_loc:imax_loc+1,np.newaxis,np.newaxis]/stdDev)**2)
                     * np.exp(-0.5*(yGrid[np.newaxis,jmin_loc:jmax_loc+1,np.newaxis]/stdDev)**2)
                     * np.exp(-0.5*(zGrid[np.newaxis,np.newaxis,kmin_loc:kmax_loc+1]/stdDev)**2) )
    data_IC = np.zeros((nx_,ny_,nz_,4),dtype='float64')
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
    data_IC = np.zeros((nx_,ny_,nz_,4),dtype='float64')
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
    if (decomp.rank==0):
        print("Bulk Re={:7f}".format(rho*uMax*configLy/mu))
    parabolaX = ( 6.0*(yGrid[np.newaxis,jmin_loc:jmax_loc+1,np.newaxis] + 0.5*configLy)
                  *(0.5*configLy - yGrid[np.newaxis,jmin_loc:jmax_loc+1,np.newaxis])
                  /configLy**2 )
    Unorm = np.sqrt(uMax**2 + wMax**2)
    data_IC = np.zeros((nx_,ny_,nz_,4),dtype='float64')
    data_IC[:,:,:,0] = (uMax * parabolaX
                        + amp*Unorm*np.cos(16.0*3.1415926*xGrid[imin_loc:imax_loc+1,np.newaxis,np.newaxis]/configLx))
    data_IC[:,:,:,1] = 0.0
    data_IC[:,:,:,2] = (wMax * parabolaX
                        + amp*Unorm*np.cos(16.0*3.1415926*zGrid[np.newaxis,np.newaxis,kmin_loc:kmax_loc+1]/configLz))
    data_IC[:,:,:,3] = 0.0
    del parabolaX


    
# ----------------------------------------------------
# Configure geometry
# ----------------------------------------------------
geometry = geo.uniform(xGrid,yGrid,zGrid,decomp,device,precision)

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

IC_zeros_np = np.zeros( (nx_,ny_,nz_) )
IC_ones_np  = np.ones ( (nx_,ny_,nz_) )

    
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
state_u_P = state.state_P(geometry,IC_u_np)
state_v_P = state.state_P(geometry,IC_v_np)
state_w_P = state.state_P(geometry,IC_w_np)
state_p_P = state.state_P(geometry,IC_p_np)
state_pOld_P = state.state_P(geometry,IC_p_np)

# Set up a Numpy mirror to the PyTorch state
#  --> Used for file I/O
state_data_all = (state_u_P, state_v_P, state_w_P, state_p_P)
data_all_CPU   = state.data_all_CPU(geometry,startTime,simDt,names,state_data_all)

# Need a temporary velocity state for RK solvers
if (solverName[:-1]=="RK"):
    state_uTmp_P = state.state_P(geometry,IC_u_np)
    state_vTmp_P = state.state_P(geometry,IC_v_np)
    state_wTmp_P = state.state_P(geometry,IC_w_np)

# Allocate workspace arrays
source_P = torch.zeros(nx_,ny_,nz_,dtype=precision).to(device)
VISC_P   = torch.ones(nxo_,nyo_,nzo_,dtype=precision).to(device)
VISC_P.mul_(mu)

# SFS model
if (SFSmodel=='Smagorinsky'):
    use_SFSmodel = True
    sfsmodel = sfsmodel_smagorinsky.stress_constCs(geometry,metric)
else:
    # Construct a blank SFSmodel object
    use_SFSmodel = False
    sfsmodel = sfsmodel_smagorinsky.stress_constCs(geometry,metric)

# Save the molecular viscosity
if (use_SFSmodel):
    muMolec = mu

# Allocate RHS objects
if (equationMode=='scalar'):
    if (decomp.rank==0):
        print("\nSolving scalar advection-diffusion equation")
    rhs1 = velocity.rhs_scalar(geometry,uMax,vMax,wMax)
    if (solverName[:-1]=="RK"):
        rhs2 = velocity.rhs_scalar(geometry,uMax,vMax,wMax)
        rhs3 = velocity.rhs_scalar(geometry,uMax,vMax,wMax)
        rhs4 = velocity.rhs_scalar(geometry,uMax,vMax,wMax)
        
elif (equationMode=='NS'):
    if (decomp.rank==0):
        print("\nSolving Navier-Stokes equations")
        print("Solver settings: advancer={}, pressure={}".format(solverName,pSolverMode))
    
    rhs1 = velocity.rhs_NavierStokes(geometry)
    if (solverName[:-1]=="RK"):
        rhs2 = velocity.rhs_NavierStokes(geometry)
        rhs3 = velocity.rhs_NavierStokes(geometry)
        rhs4 = velocity.rhs_NavierStokes(geometry)
        
    # Pressure solver
    if (pSolverMode=='Jacobi'):
        poisson = pressure.solver_jacobi(geometry,rho,simDt,Num_pressure_iterations)
    elif (pSolverMode=='bicgstab'):
        if (nproc_x>1 or nproc_y>1 or nproc_z>1):
            if (decomp.rank==0):
                raise Exception('\nbicgstab solver currently not implemented in parallel\n')
        else:
            poisson = pressure.solver_bicgstab_serial(geometry,metric,rho,simDt,Num_pressure_iterations)
        
else:
    if (decomp.rank==0):
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
        if (decomp.rank==0):
            headStr = "  {:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}"
            print(headStr.format("Step","Time","max CFL","max U","max V","max W","int RP","max res_P"))
    else:
        if (decomp.rank==0):
            headStr = "  {:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}"
            print(headStr.format("Step","Time","max CFL","max U","max V","max W"))

    # Write the initial data file
    timeStr = "{:12.7E}".format(simTime)
    #dr.writeNGArestart(fNameOut+'_'+timeStr,data_all_CPU,False)
    
    # Write initial condition stats - JFM make this parallel max
    maxU = torch.max(state_u_P.var)
    maxV = torch.max(state_v_P.var)
    maxW = torch.max(state_w_P.var)
    maxCFL = max((maxU/geometry.dx,maxV/geometry.dy,maxW/geometry.dz))*simDt
    if (decomp.rank==0):
        lineStr = "  {:10d}\t{:10.6E}\t{:10.6E}\t{:10.6E}\t{:10.6E}\t{:10.6E}"
        print(lineStr.format(itCount,simTime,maxCFL,maxU,maxV,maxW))

    if (plotState):
        timeStr = "{:12.7E}".format(simTime)
        # Plot the initial state
        dr.plotData(state_u_P.var[:,:,int(geometry.Nz/2)].cpu().numpy(),"state_U_"+str(itCount)+"_"+timeStr)
        dr.plotData(state_v_P.var[:,:,int(geometry.Nz/2)].cpu().numpy(),"state_V_"+str(itCount)+"_"+timeStr)
        dr.plotData(state_w_P.var[:,:,int(geometry.Nz/2)].cpu().numpy(),"state_W_"+str(itCount)+"_"+timeStr)
        dr.plotData(state_p_P.var[:,:,int(geometry.Nz/2)].cpu().numpy(),"state_p_"+str(itCount)+"_"+timeStr)

    # Compute the initial energy
    initEnergy = torch.sum(state_u_P.var**2 + state_v_P.var**2 + state_w_P.var**2)
    
    # Main iteration loop
    while (simTime < stopTime):

        # [JFM] need new sub-iteration loop
        
        # ----------------------------------------------------
        # Velocity prediction step
        # ----------------------------------------------------
        
        #model_output = model( state_u_P.var)
        #model_output2 = model_output.cpu()

        # Evaluate any SFS models
        if (use_SFSmodel):
            if (sfsmodel.modelType=='eddyVisc'):
                muEddy = sfsmodel.eddyVisc(state_u_P,state_v_P,state_w_P,rho,metric)
                VISC_P.copy_( muMolec + muEddy )
            elif (sfsmodel.modelType=='tensor'):
                if (decomp.rank==0):
                    print(' --> SFS model type not implemented')
            else:
                if (decomp.rank==0):
                    print(' --> SFS model type not implemented')
                
        # Compute velocity prediction
        if (solverName=="Euler"):
            # rhs
            rhs1.evaluate(state_u_P,state_v_P,state_w_P,VISC_P,rho,metric)

            # Update the state using explicit Euler
            state_u_P.var = state_u_P.var + rhs1.rhs_u*simDt
            state_v_P.var = state_v_P.var + rhs1.rhs_v*simDt
            state_w_P.var = state_w_P.var + rhs1.rhs_w*simDt

        elif (solverName=="RK4"):
            
            # [JFM] need to pass tensor-type turbulence models into the RHS
            
            # Stage 1
            rhs1.evaluate(state_u_P,state_v_P,state_w_P,VISC_P,rho,metric)
            
            # Stage 2
            state_uTmp_P.ZAXPY(0.5*simDt,rhs1.rhs_u,state_u_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1])
            state_vTmp_P.ZAXPY(0.5*simDt,rhs1.rhs_v,state_v_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1])
            state_wTmp_P.ZAXPY(0.5*simDt,rhs1.rhs_w,state_w_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1])
            rhs2.evaluate(state_uTmp_P,state_vTmp_P,state_wTmp_P,VISC_P,rho,metric)

            # Stage 3
            state_uTmp_P.ZAXPY(0.5*simDt,rhs2.rhs_u,state_u_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1])
            state_vTmp_P.ZAXPY(0.5*simDt,rhs2.rhs_v,state_v_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1])
            state_wTmp_P.ZAXPY(0.5*simDt,rhs2.rhs_w,state_w_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1])
            rhs3.evaluate(state_uTmp_P,state_vTmp_P,state_wTmp_P,VISC_P,rho,metric)

            # Stage 4
            state_uTmp_P.ZAXPY(simDt,rhs3.rhs_u,state_u_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1])
            state_vTmp_P.ZAXPY(simDt,rhs3.rhs_v,state_v_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1])
            state_wTmp_P.ZAXPY(simDt,rhs3.rhs_w,state_w_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1])
            rhs4.evaluate(state_uTmp_P,state_vTmp_P,state_wTmp_P,VISC_P,rho,metric)

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

            # Solve the Poisson equation
            poisson.solve(state_pOld_P,state_p_P,source_P)
                
            # Max pressure residual
            max_res_P = torch.max(torch.abs(state_p_P.var - state_pOld_P.var))

        
            # ----------------------------------------------------
            # Velocity correction step
            # ----------------------------------------------------

            # Compute pressure gradients
            metric.grad_P(state_p_P)

            # Update the velocity correction
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
            if (decomp.rank==0):
                lineStr = "  {:10d}\t{:10.6E}\t{:10.6E}\t{:10.6E}\t{:10.6E}\t{:10.6E}\t{: 10.6E}\t{: 10.6E}"
                print(lineStr.format(itCount,simTime,maxCFL,maxU,maxV,maxW,int_RP,max_res_P))
        else:
            if (decomp.rank==0):
                lineStr = "  {:10d}\t{:10.6E}\t{:10.6E}\t{:10.6E}\t{:10.6E}\t{:10.6E}"
                print(lineStr.format(itCount,simTime,maxCFL,maxU,maxV,maxW))

        # Write output
        if (np.mod(itCount,numItDataOut)==0):
            # Write data to disk
            data_all_CPU.time = simTime
            data_all_CPU.dt   = simDt
            timeStr = "{:12.7E}".format(simTime)
            #dr.writeNGArestart(fNameOut+'_'+timeStr,data_all_CPU,False)

        if (plotState and np.mod(itCount,numItPlotOut)==0):
            timeStr = "{:12.7E}".format(simTime)
            dr.plotData(state_u_P.var[:,:,int(geometry.Nz/2)].cpu().numpy(),"state_U_"+str(itCount)+"_"+timeStr)
            dr.plotData(state_v_P.var[:,:,int(geometry.Nz/2)].cpu().numpy(),"state_V_"+str(itCount)+"_"+timeStr)
            dr.plotData(state_w_P.var[:,:,int(geometry.Nz/2)].cpu().numpy(),"state_W_"+str(itCount)+"_"+timeStr)
            dr.plotData(state_p_P.var[:,:,int(geometry.Nz/2)].cpu().numpy(),"state_p_"+str(itCount)+"_"+timeStr)

        ## END OF ITERATION LOOP




        
    # ----------------------------------------------------
    # Post-simulation tasks
    # ----------------------------------------------------
    
    # Write the final state to disk
    data_all_CPU.time = simTime
    data_all_CPU.dt   = simDt
    timeStr = "{:12.7E}".format(simTime)
    #dr.writeNGArestart(fNameOut+'_'+timeStr,data_all_CPU,False)
            
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
        if (decomp.rank==0):
            print(iterations,test,error,time_elapsed)
    else:
        if (decomp.rank==0):
            print("it={}, test={}, elapsed={}, energy final/initial={}".format(iterations,test,time_elapsed,
                                                                           finalEnergy/initEnergy))

    if (plotState and decomp.rank==0):
        # Print a pretty picture
        timeStr = "{:12.7E}".format(simTime)
        dr.plotData(state_u_P.var[:,:,int(geometry.Nz/2)].cpu().numpy(),"state_U_"+str(itCount)+"_"+timeStr)
        dr.plotData(state_v_P.var[:,:,int(geometry.Nz/2)].cpu().numpy(),"state_V_"+str(itCount)+"_"+timeStr)
        dr.plotData(state_w_P.var[:,:,int(geometry.Nz/2)].cpu().numpy(),"state_W_"+str(itCount)+"_"+timeStr)
        dr.plotData(state_p_P.var[:,:,int(geometry.Nz/2)].cpu().numpy(),"state_p_"+str(itCount)+"_"+timeStr)
