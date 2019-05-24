# ------------------------------------------------------------------------
#
# PyFlow: A GPU-accelerated CFD platform written in Python
#
# @authors:
#    Jonathan F. MacArt
#    Justin A. Sirignano
#    Jonathan B. Freund
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
import sys
sys.path.insert(0, "/mnt/bwpy/single/usr/lib/python3.5/site-packages")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
sys.path.pop(0)

import time
import copy
import os

# Load PyFlow modules
sys.path.append("../data")
import state
import dataReader as dr
import constants as const
#
sys.path.append("../library")
import parallel
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
import sfsmodel_gradient


####### TODO
#  1. Non-periodic BCs
#  2. Midpoint fractional-step
#  3. RK3
#  4. Krylov - GPU
#  5. Non-uniform grid


# ----------------------------------------------------
# User-specified parameters
# ----------------------------------------------------

# [JFM] move this to an input file passed as a command-line arg

# Simulation geometry
#   Options: restart, periodicGaussian, uniform, notAchannel
configName = "restart"
#configName = "periodicGaussian"
#configName = "uniform"
#configName = "notAchannel"
#configNx   = 2
#configNy   = 2
#configNz   = 2
#configNx   = 64
#configNy   = 64
#configNz   = 64
#configLx   = 1.0
#configLy   = 1.0
#configLz   = 1.0
#isper = [1,1,1]

# Data and config files to read
if (configName=='restart'):

    # Example case in git repository
    inFileDir     = '../../examples/'
    configFileStr = inFileDir+'config_dnsbox_128_Lx0.0056'
    dataFileBStr  = inFileDir+'dnsbox_128_128_Lx0.0056.1_2.50000E-04'
    dataFileStr   = inFileDir+'data_dnsbox_128_Lx0.0056.1_2.50000E-04'
    dataFileType  = 'restart'

    # Isotropic 128^3 DNS - verification vs. NGA
    #inFileDir     = '../../verification/dnsbox_128_Lx0.0056_NR/test_input_files/'
    #configFileStr = inFileDir+'config_dnsbox_128_Lx0.0056'
    #dataFileBStr  = inFileDir+'dnsbox_128_128_Lx0.0056.1_5.00000E-04'
    #dataFileStr   = inFileDir+'data_dnsbox_128_Lx0.0056.1_5.00000E-04'
    #dataFileType  = 'restart'

    # Downsampled 1024^3 DNS - needs SGS model
    #inFileDir     = '../../input/downsampled_LES_restart/dnsbox_1024_Lx0.045_NR_run_2/restart_1024_Lx0.045_NR_Delta16_Down16/'
    #configFileStr = inFileDir+'config_dnsbox_1024_Lx0.045_NR_Delta16_Down16_0000'
    #dataFileBStr  = inFileDir+'dnsbox_1024_Lx0.045_NR_Delta16_Down16_000000'
    #startFileIt   = 20
    #dataFileStr   = dataFileBStr + str(startFileIt)
    #dataFileType  = 'restart'

# Data file to write
fNameOut     = 'data_dnsbox_128_Lx0.0056'
#fNameOut     = 'dnsbox_1024_Lx0.045_NR_Delta16_Down16_00000020'
numItDataOut = 20

# Parallel decomposition
nproc_x = 1
nproc_y = 1
nproc_z = 1

# Physical constants
mu  = 1.8678e-5
rho = 1.2

# Time step info
simDt        = 1.0e-6
numIt        = 50
startTime    = 0.0

# SFS model settings
#   SFSmodel options: none, Smagorinsky, gradient, nn
SFSmodel = 'none'
#SFSmodel = 'Smagorinsky'; Cs = 0.18; expFilterFac = 1.0;

# Solver settings
#   advancerName options: Euler, RK4
#   equationMode options: scalar, NS
#   pSolverMode options:  Jacobi, bicgstab (Krylov subspace)
#
advancerName = "RK4"
equationMode = "NS"
#
# Pressure solver settings
pSolverMode             = "bicgstab"
min_pressure_residual   = 1e-9
max_pressure_iterations = 150
#
# Accuracy and precision settings
genericOrder = 2
dtypeTorch   = torch.float64
dtypeNumpy   = np.float64

# Output options
plotState    = False
numItPlotOut = 20

# Comparison options
useTargetData = False
if (useTargetData):
    targetFileBaseStr  = dataFileBStr
    numItTargetComp = 50




# ----------------------------------------------------
# Configure PyTorch
# ----------------------------------------------------

# Offload to GPUs if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    
# ----------------------------------------------------
# Configure simulation domain
# ----------------------------------------------------
# Basic parallel operations
comms = parallel.comms(dtypeNumpy)

# Domain sizes if performing a restart
if (configName=='restart'):
    if (dataFileType=='restart'):
        # Read grid data from an NGA config file
        xGrid,yGrid,zGrid,xper,yper,zper = dr.readNGAconfig(configFileStr)
        configNx = len(xGrid)-1
        configNy = len(yGrid)-1
        configNz = len(zGrid)-1
        isper = [xper,yper,zper]

    elif (dataFileType=='volume'):
        # Read grid and state data from a volume-format file
        xGrid,yGrid,zGrid,names,dataTime,data = dr.readNGA(dataFileStr)
        configNx = len(xGrid)
        configNy = len(yGrid)
        configNz = len(zGrid)

# Domain decomposition
nproc  = [nproc_x,nproc_y,nproc_z]
decomp = parallel.decomp(configNx,configNy,configNz,nproc,isper,device,
                         dtypeTorch,dtypeNumpy)

if ((nproc_x>1 and nproc_y>1) or (nproc_z>1)):
    plotState = False

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
        # Read state information from an NGA restart file
        names,startTime = dr.readNGArestart(dataFileStr)

    elif (dataFileType=='volume'):
        # Read grid and state data from a volume-format file
        xGrid,yGrid,zGrid,names,dataTime,data = dr.readNGA(dataFileStr)
        configNx = len(xGrid)
        configNy = len(yGrid)
        configNz = len(zGrid)

        # Interpolate grid and state data to cell faces
        raise Exception('\nPyFlow: Volume file interpolation not implemented\n')
        # Need periodicity information

    # Read restart data later
    data_IC = np.zeros((nx_,ny_,nz_,4),dtype=dtypeNumpy)
    
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
    data_IC = np.zeros((nx_,ny_,nz_,4),dtype=dtypeNumpy)
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
    data_IC = np.zeros((nx_,ny_,nz_,4),dtype=dtypeNumpy)
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
    data_IC = np.zeros((nx_,ny_,nz_,4),dtype=dtypeNumpy)
    data_IC[:,:,:,0] = (uMax * parabolaX
                        + amp*Unorm*np.cos(16.0*3.1415926*xGrid[imin_loc:imax_loc+1,np.newaxis,np.newaxis]/configLx))
    data_IC[:,:,:,1] = 0.0
    data_IC[:,:,:,2] = (wMax * parabolaX
                        + amp*Unorm*np.cos(16.0*3.1415926*zGrid[np.newaxis,np.newaxis,kmin_loc:kmax_loc+1]/configLz))
    data_IC[:,:,:,3] = 0.0
    del parabolaX


    
# ----------------------------------------------------
# Configure geometry and metrics
# ----------------------------------------------------
geometry = geo.uniform(xGrid,yGrid,zGrid,decomp)

# Initialize the metrics
metric = metric_staggered.metric_uniform(geometry)



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
#model = sfsmodel_nn.ClosureModel2()
#model_name = 'LES_model_NR_March2019'
#model.load_state_dict(torch.load(model_name))
#model.to(device)


# ----------------------------------------------------
# Allocate memory for state data
# ----------------------------------------------------

# Allocate state data using PyTorch variables
state_u_P    = state.state_P(decomp,IC_u_np)
state_v_P    = state.state_P(decomp,IC_v_np)
state_w_P    = state.state_P(decomp,IC_w_np)
state_p_P    = state.state_P(decomp,IC_p_np)
state_DP_P = state.state_P(decomp,IC_p_np)

# Set up a Numpy mirror to the PyTorch state
#  --> Used for file I/O
state_data_all = (state_u_P, state_v_P, state_w_P, state_p_P)
data_all_CPU   = state.data_all_CPU(decomp,startTime,simDt,names[0:4],state_data_all)

# Need a temporary velocity state for RK solvers
if (advancerName[:-1]=="RK"):
    state_uTmp_P = state.state_P(decomp,IC_u_np)
    state_vTmp_P = state.state_P(decomp,IC_v_np)
    state_wTmp_P = state.state_P(decomp,IC_w_np)

# Allocate pressure source term and local viscosity
source_P = torch.zeros(nx_,ny_,nz_,dtype=dtypeTorch).to(device)
VISC_P   = torch.ones(nxo_,nyo_,nzo_,dtype=dtypeTorch).to(device)
VISC_P.mul_(mu)

# SFS model
if (SFSmodel=='Smagorinsky'):
    use_SFSmodel = True
    sfsmodel = sfsmodel_smagorinsky.stress_constCs(geometry,metric,Cs,expFilterFac)
elif (SFSmodel=='gradient'):
    use_SFSmodel = True
    sfsmodel = sfsmodel_gradient.residual_stress(decomp,geometry,metric)
else:
    # Construct a blank SFSmodel object
    use_SFSmodel = False
    sfsmodel = sfsmodel_smagorinsky.stress_constCs(geometry,metric)

# Save the molecular viscosity
if (sfsmodel.modelType=='eddyVisc'):
    muMolec = mu

# Switch for different solver types
if (equationMode=='scalar'):
    # Scalar advection-diffusion equations
    if (decomp.rank==0):
        print("\nSolving scalar advection-diffusion equation")

    # Allocate RHS objects
    rhs1 = velocity.rhs_scalar(decomp,uMax,vMax,wMax)
    if (advancerName[:-1]=="RK"):
        rhs2 = velocity.rhs_scalar(decomp,uMax,vMax,wMax)
        rhs3 = velocity.rhs_scalar(decomp,uMax,vMax,wMax)
        rhs4 = velocity.rhs_scalar(decomp,uMax,vMax,wMax)
        
elif (equationMode=='NS'):
    # Navier-Stokes equations
    if (decomp.rank==0):
        print("\nSolving Navier-Stokes equations")
        print("Solver settings: advancer={}, pressure={}".format(advancerName,pSolverMode))

    # Allocate RHS objects    
    rhs1 = velocity.rhs_NavierStokes(decomp)
    if (advancerName[:-1]=="RK"):
        rhs2 = velocity.rhs_NavierStokes(decomp)
        rhs3 = velocity.rhs_NavierStokes(decomp)
        rhs4 = velocity.rhs_NavierStokes(decomp)
        
    # Initialize pressure solver
    if (pSolverMode=='Jacobi'):
        poisson = pressure.solver_jacobi(comms,decomp,metric,
                                         geometry,rho,simDt,max_pressure_iterations)
    elif (pSolverMode=='bicgstab'):
        poisson = pressure.solver_bicgstab(comms,decomp,metric,geometry,rho,simDt,
                                           min_pressure_residual,max_pressure_iterations)
    if (pSolverMode=='RedBlackGS'):
        #poisson = pressure.solver_GS_redblack(geometry,rho,simDt,max_pressure_iterations)
        raise Exception('\Red-black GS not yet implemented\n')
        
else:
    if (decomp.rank==0):
        raise Exception("Equation setting not recognized; consequences unknown...")

# Read restart state data in parallel
if (configName=='restart'):
    if (dataFileType=='restart'):
        dr.readNGArestart_parallel(dataFileStr,data_all_CPU)



# ----------------------------------------------------
# Allocate memory for target state data
# ----------------------------------------------------
if (useTargetData):
    state_u_T    = state.state_P(decomp,IC_zeros_np)
    state_v_T    = state.state_P(decomp,IC_zeros_np)
    state_w_T    = state.state_P(decomp,IC_zeros_np)
    
    #  Set up a Numpy mirror to the target state data
    target_data_all = (state_u_T, state_v_T, state_w_T)
    target_data_all_CPU = state.data_all_CPU(decomp,startTime,simDt,names[0:3],target_data_all)
    dr.readNGArestart_parallel(dataFileStr,target_data_all_CPU)
    print(target_data_all_CPU.read(0)[0,0,0])

    
#    # Read target data file
#    xGrid_t,yGrid_t,zGrid_t,names_t,dataTime_t,data_t = dr.readNGA(targetFileStr)
#    # Just save U for now
#    data_target10 = data_t[:,:,:,0]
#    
#    # Clean up
#    del data_t
#    
#    x_max_P  = torch.FloatTensor( const.x_max16 ).to(device)
#    target_P = torch.FloatTensor( data_target10 ).to(device)
#        
#    # Clean up
#    del data_target10

    
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
            headStr = "  {:10s}   {:9s}   {:9s}   {:9s}   {:9s}   {:9s}   {:9s}   {:9s}   {:9s}"
            print(headStr.format("Step","Time","max CFL","max U","max V","max W","TKE","divergence","max res_P"))
    else:
        if (decomp.rank==0):
            headStr = "  {:10s}   {:9s}   {:9s}   {:9s}   {:9s}   {:9s}"
            print(headStr.format("Step","Time","max CFL","max U","max V","max W"))

    # Synchronize the overlap cells before stepping
    state_u_P.update_border()
    state_v_P.update_border()
    state_w_P.update_border()
    state_p_P.update_border()

    # Write the initial data file
    timeStr = "{:12.7E}".format(simTime)
    if (decomp.rank==0):
        dr.writeNGArestart(fNameOut+'_'+timeStr,data_all_CPU,True)
    dr.writeNGArestart_parallel(fNameOut+'_'+timeStr,data_all_CPU)

    # Compute resolved kinetic energy and velocity rms
    initEnergy = comms.parallel_sum(np.sum( data_all_CPU.read(0)**2 +
                                            data_all_CPU.read(1)**2 +
                                            data_all_CPU.read(2)**2 ))
    #rmsVel = np.sqrt(initEnergy/decomp.N)
    if (equationMode=='NS'):
        # Compute the initial divergence
        metric.div_vel(state_u_P,state_v_P,state_w_P,source_P)
        maxDivg = comms.parallel_max(torch.max(torch.abs(source_P)).cpu().numpy())
    
    # Write initial condition stats
    maxU = comms.parallel_max(data_all_CPU.absmax(0))
    maxV = comms.parallel_max(data_all_CPU.absmax(1))
    maxW = comms.parallel_max(data_all_CPU.absmax(2))
    TKE  = comms.parallel_sum(np.sum( data_all_CPU.read(0)**2 +
                                      data_all_CPU.read(1)**2 +
                                      data_all_CPU.read(2)**2 ))*0.5/float(decomp.N)
    if (decomp.rank==0):
        maxCFL = max((maxU/geometry.dx,maxV/geometry.dy,maxW/geometry.dz))*simDt
        lineStr = "  {:10d}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}"
        print(lineStr.format(itCount,simTime,maxCFL,maxU,maxV,maxW,TKE,maxDivg))

    if (plotState):
        timeStr = "{:12.7E}_{}".format(simTime,decomp.rank)
        # Plot the initial state
        decomp.plot_fig_root(dr,state_u_P.var,"state_U_"+str(itCount)+"_"+timeStr)
    
    # Main iteration loop
    while (simTime < stopTime and itCount < numIt):

        # [JFM] need new sub-iteration loop
        
        # ----------------------------------------------------
        # Velocity prediction step
        # ----------------------------------------------------

        # Evaluate any SFS models
        if (use_SFSmodel):
            if (sfsmodel.modelType=='eddyVisc'):
                muEddy = sfsmodel.eddyVisc(state_u_P,state_v_P,state_w_P,rho,metric)
                VISC_P.copy_( muMolec + muEddy )
            elif (sfsmodel.modelType=='tensor'):
                sfsmodel.update(state_u_P,state_v_P,state_w_P,metric)
            else:
                if (decomp.rank==0):
                    raise Exception('PyFlow: SFS model type not implemented')
                
        # Compute velocity prediction
        if (advancerName=="Euler"):
            # rhs
            rhs1.evaluate(state_u_P,state_v_P,state_w_P,VISC_P,rho,sfsmodel,metric)

            # Update the state
            state_u_P.var = state_u_P.var + rhs1.rhs_u*simDt
            state_v_P.var = state_v_P.var + rhs1.rhs_v*simDt
            state_w_P.var = state_w_P.var + rhs1.rhs_w*simDt

        elif (advancerName=="RK4"):
            
            # Stage 1
            rhs1.evaluate(state_u_P,state_v_P,state_w_P,VISC_P,rho,sfsmodel,metric)
            
            # Stage 2
            state_uTmp_P.ZAXPY(0.5*simDt,rhs1.rhs_u,state_u_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1])
            state_vTmp_P.ZAXPY(0.5*simDt,rhs1.rhs_v,state_v_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1])
            state_wTmp_P.ZAXPY(0.5*simDt,rhs1.rhs_w,state_w_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1])
            rhs2.evaluate(state_uTmp_P,state_vTmp_P,state_wTmp_P,VISC_P,rho,sfsmodel,metric)

            # Stage 3
            state_uTmp_P.ZAXPY(0.5*simDt,rhs2.rhs_u,state_u_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1])
            state_vTmp_P.ZAXPY(0.5*simDt,rhs2.rhs_v,state_v_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1])
            state_wTmp_P.ZAXPY(0.5*simDt,rhs2.rhs_w,state_w_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1])
            rhs3.evaluate(state_uTmp_P,state_vTmp_P,state_wTmp_P,VISC_P,rho,sfsmodel,metric)

            # Stage 4
            state_uTmp_P.ZAXPY(simDt,rhs3.rhs_u,state_u_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1])
            state_vTmp_P.ZAXPY(simDt,rhs3.rhs_v,state_v_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1])
            state_wTmp_P.ZAXPY(simDt,rhs3.rhs_w,state_w_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1])
            rhs4.evaluate(state_uTmp_P,state_vTmp_P,state_wTmp_P,VISC_P,rho,sfsmodel,metric)

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
            #int_RP = comms.parallel_sum(torch.sum(source_P).cpu().numpy())

            # Solve the Poisson equation
            max_resP = poisson.solve(state_DP_P,state_p_P,source_P)

        
            # ----------------------------------------------------
            # Velocity correction step
            # ----------------------------------------------------

            # Compute pressure gradients
            metric.grad_P(state_DP_P)

            # Update the velocity correction
            state_u_P.vel_corr(state_DP_P.grad_x,simDt/rho)
            state_v_P.vel_corr(state_DP_P.grad_y,simDt/rho)
            state_w_P.vel_corr(state_DP_P.grad_z,simDt/rho)

        
        # ----------------------------------------------------
        # Post-step
        # ----------------------------------------------------
        # Compute stats
        maxU = comms.parallel_max(data_all_CPU.absmax(0))
        maxV = comms.parallel_max(data_all_CPU.absmax(1))
        maxW = comms.parallel_max(data_all_CPU.absmax(2))
        maxCFL = max((maxU/geometry.dx,maxV/geometry.dy,maxW/geometry.dz))*simDt
        TKE  = comms.parallel_sum(np.sum( data_all_CPU.read(0)**2 +
                                          data_all_CPU.read(1)**2 +
                                          data_all_CPU.read(2)**2 ))*0.5/float(decomp.N)
        #rmsVel = comms.parallel_sum(np.sum( data_all_CPU.read(0)**2 +
        #                                    data_all_CPU.read(1)**2 +
        #                                    data_all_CPU.read(2)**2 ))
        #rmsVel = np.sqrt(rmsVel/decomp.N)
        if (equationMode=='NS'):
            # Compute the final divergence
            metric.div_vel(state_u_P,state_v_P,state_w_P,source_P)
            maxDivg = comms.parallel_max(torch.max(torch.abs(source_P)).cpu().numpy())

        # Update the time
        itCount += 1
        simTime += simDt
        
        # Write stats
        if (equationMode=='NS'):
            if (decomp.rank==0):
                lineStr = "  {:10d}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}   {: 8.3E}   {: 8.3E}"
                print(lineStr.format(itCount,simTime,maxCFL,maxU,maxV,maxW,TKE,maxDivg,max_resP))
        else:
            if (decomp.rank==0):
                lineStr = "  {:10d}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}"
                print(lineStr.format(itCount,simTime,maxCFL,maxU,maxV,maxW))

        # Write output
        if (np.mod(itCount,numItDataOut)==0):
            # Write data to disk
            data_all_CPU.time = simTime
            data_all_CPU.dt   = simDt
            timeStr = "{:12.7E}".format(simTime)
            if (decomp.rank==0):
                dr.writeNGArestart(fNameOut+'_'+timeStr,data_all_CPU,True)
            dr.writeNGArestart_parallel(fNameOut+'_'+timeStr,data_all_CPU)

        if (plotState and np.mod(itCount,numItPlotOut)==0):
            timeStr = "{:12.7E}_{}".format(simTime,decomp.rank)
            decomp.plot_fig_root(dr,state_u_P.var,"state_U_"+str(itCount)+"_"+timeStr)


        # Compare to target DNS data
        if (useTargetData and np.mod(itCount,numItTargetComp)==0):
            # Only on root processor for now
            targetFileIt  = startFileIt+itCount
            targetFileStr = targetFileBaseStr + str(targetFileIt)

            # Check to make sure we read at the right time
            if (decomp.rank==0):
                names_t,simTime_t = dr.readNGArestart(targetFileStr,printOut=False)
                if (False): #(simTime!=simTime_t):
                    raise Exception("\nPyFlow: target file not at same time as simulation\n")
                else:
                    print(" --> Comparing to target data file {} at time {:10.5E}".format(targetFileIt,simTime_t))

            # Read the target state data
            #dr.readNGArestart_parallel(targetFileStr,target_data_all_CPU,ivar_read_start=0,nvar_read=3)
            #names_t,simTime_t,data_t = dr.readNGArestart(targetFileStr,headerOnly=False,printOut=False)
            #target_data_all_CPU.append(0,data_t[:,:,:,0])

            # L1 error of x-velocity field
            print(data_all_CPU.read(0)[0,0,0])
            print(np.max(target_data_all_CPU.read(0)))
            print(target_data_all_CPU.read(0)[0,0,0])
            maxU_sim = comms.parallel_max(np.max(np.abs( data_all_CPU.read(0) )))
            maxU_t   = comms.parallel_max(np.max(np.abs( target_data_all_CPU.read(0) )))
            L1_error = (np.mean(np.abs( data_all_CPU.read(0) -
                                                         target_data_all_CPU.read(0) )))#/(geometry.Nx*geometry.Ny*geometry.Nz)
                                          
            if (decomp.rank==0):
                print("     Max(U) sim: {:10.5E}, Max(U) target: {:10.5E}".format(maxU_sim,maxU_t))
                print("     L1 error  : {:10.5E}".format(L1_error))
                
        ## END OF ITERATION LOOP




        
    # ----------------------------------------------------
    # Post-simulation tasks
    # ----------------------------------------------------
    
    # Write the final state to disk
    data_all_CPU.time = simTime
    data_all_CPU.dt   = simDt
    timeStr = "{:12.7E}".format(simTime)
    if (decomp.rank==0):
        dr.writeNGArestart(fNameOut+'_'+timeStr,data_all_CPU,True)
    dr.writeNGArestart_parallel(fNameOut+'_'+timeStr,data_all_CPU)
            
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
    finalEnergy = comms.parallel_sum(np.sum( data_all_CPU.read(0)**2 +
                                             data_all_CPU.read(1)**2 +
                                             data_all_CPU.read(2)**2 ))*0.5
    
    if (useTargetData):
        if (decomp.rank==0):
            print(iterations,test,error,time_elapsed)
    else:
        if (decomp.rank==0):
            print("it={}, test={:10.5E}, elapsed={}".format(iterations,test,time_elapsed))
            print("Energy initial={:10.5E}, final={:10.5E}, ratio={:10.5E}".format(initEnergy,finalEnergy,
                                                                                   finalEnergy/initEnergy))

    if (plotState):
        # Print a pretty picture
        timeStr = "{:12.7E}_{}".format(simTime,decomp.rank)
        decomp.plot_fig_root(dr,state_u_P.var,"state_U_"+str(itCount)+"_"+timeStr)
