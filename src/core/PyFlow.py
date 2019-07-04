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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import time
import copy
import os

import resource

# Load PyFlow modules
sys.path.append("../data")
import state
import dataReader as dr
import constants as const
import initial_conditions
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
import adjoint
#
sys.path.append("../sfsmodel")
import sfsmodel_ML
import sfsmodel_smagorinsky
import sfsmodel_gradient


####### TODO
#  1. Non-periodic BCs
#  2. Midpoint fractional-step
#  3. RK3
#  4. Non-uniform grid


# ----------------------------------------------------
# User-specified parameters
# ----------------------------------------------------

# [JFM] move this to an input file passed as a command-line arg
class inputConfigClass:
    def __init__(self):
        # Simulation geometry
        #   Options: restart, periodicGaussian, uniform, notAchannel
        self.configName = "restart"
        #configName = "periodicGaussian"
        #configName = "uniform"
        #configName = "notAchannel"
        #self.Nx   = 2
        #self.Ny   = 2
        #self.Nz   = 2
        #self.Nx   = 64
        #self.Ny   = 64
        #self.Nz   = 64
        #self.Lx   = 1.0
        #self.Ly   = 1.0
        #self.Lz   = 1.0
        #isper = [1,1,1]
        
        # Example case in git repository
        #inFileDir     = '../../examples/'
        #configFileStr = inFileDir+'config_dnsbox_128_Lx0.0056'
        #dataFileBStr  = inFileDir+'dnsbox_128_128_Lx0.0056.1_2.50000E-04'
        #dataFileStr   = inFileDir+'data_dnsbox_128_Lx0.0056.1_2.50000E-04'
        #dataFileType  = 'restart'
        
        # Isotropic 128^3 DNS - verification vs. NGA
        #inFileDir     = '../../verification/dnsbox_128_Lx0.0056_NR/test_input_files/'
        #configFileStr = inFileDir+'config_dnsbox_128_Lx0.0056'
        #dataFileBStr  = inFileDir+'dnsbox_128_128_Lx0.0056.1_5.00000E-04'
        #dataFileStr   = inFileDir+'data_dnsbox_128_Lx0.0056.1_5.00000E-04'
        #dataFileType  = 'restart'
        
        # Downsampled 1024^3 DNS - needs SGS model
        self.inFileBase    = '../../verification/'
        self.inFileDir     = self.inFileBase+'downsampled_LES_restart/dnsbox_1024_Lx0.045_NR_run_2/restart_1024_Lx0.045_NR_Delta16_Down16/test_input_files/'
        self.configFileStr = self.inFileDir+'config_dnsbox_1024_Lx0.045_NR_Delta16_Down16_0000'
        self.dataFileBStr  = self.inFileDir+'dnsbox_1024_Lx0.045_NR_Delta16_Down16_'
        self.startFileIt   = 20
        self.dataFileStr   = self.dataFileBStr + '{:08d}'.format(self.startFileIt)
        self.dataFileType  = 'restart'
        
        # Data file to write
        #self.fNameOut     = 'data_dnsbox_128_Lx0.0056'
        self.fNameOut     = 'dnsbox_1024_Lx0.045_NR_Delta16_Down16_00000020'
        self.numItDataOut = 20

        # Parallel decomposition
        self.nproc_x = 2
        self.nproc_y = 1
        self.nproc_z = 1

        # Physical constants
        self.mu  = 1.8678e-5
        self.rho = 1.2

        # Time step info
        self.simDt        = 1.0e-6
        self.numIt        = 50
        self.startTime    = 0.0
        
        # SFS model settings
        #   SFSmodel options: none, Smagorinsky, gradient, ML
        #self.SFSmodel = 'none'
        #self.SFSmodel = 'Smagorinsky'; self.Cs = 0.18; self.expFilterFac = 1.0;
        # ML model input
        self.SFSmodel      = 'ML';
        self.modelDictName = 'test_model.dict'
        self.modelDictSave = 'save_model.dict'
        self.loadModel     = False
        self.saveModel     = True

        # Adjoint training settings
        #   PyFlow will look for a target data file every numCheckpointIt
        self.adjointTraining = True
        self.numCheckpointIt = 5

        # Solver settings
        #   advancerName options: Euler, RK4
        #   equationMode options: scalar, NS
        #   pSolverMode options:  Jacobi, bicgstab
        #
        self.advancerName = "RK4"
        self.equationMode = "NS"
        #
        # Pressure solver settings
        self.pSolverMode             = "bicgstab"
        self.min_pressure_residual   = 1e-9
        self.max_pressure_iterations = 150
        #
        # Accuracy and precision settings
        self.genericOrder = 2
        self.dtypeTorch   = torch.float64
        self.dtypeNumpy   = np.float64
        
        # Output options
        self.plotState    = False
        self.numItPlotOut = 20
        
        # Comparison options (deprecated)
        self.useTargetData = False
        if (self.useTargetData):
            self.targetFileBaseStr = dataFileBStr
            self.numItTargetComp = 50


inputConfig = inputConfigClass()


# ----------------------------------------------------
# Configure PyTorch
# ----------------------------------------------------
# Offload to GPUs if available
# Needs update for multi-GPU systems
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



# ----------------------------------------------------
# Configure simulation domain
# ----------------------------------------------------
# Basic parallel operations
comms = parallel.comms(inputConfig.dtypeNumpy)

# Configure grid sizes
config = geo.config(inputConfig)

# Domain decomposition
decomp = parallel.decomp(inputConfig,config)
    #configNx,configNy,configNz,nproc,isper,device,
    #                     dtypeTorch,dtypeNumpy)

if ((inputConfig.nproc_x>1 and inputConfig.nproc_y>1) or (inputConfig.nproc_z>1)):
    inputConfig.plotState = False

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

# Global grid sizes
nx = decomp.nx; ny = decomp.ny; nz = decomp.nz

# Time step size
simDt = inputConfig.simDt

# Target data settings
useTargetData = inputConfig.useTargetData

# ----------------------------------------------------
# Generate initial conditions
# ----------------------------------------------------
names,startTime,data_IC,xGrid,yGrid,zGrid = \
    initial_conditions.generate(inputConfig,config,decomp)

    
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
# Allocate memory for state data
# ----------------------------------------------------
# Allocate state data using PyTorch variables
state_u_P    = state.state_P(decomp,IC_u_np)
state_v_P    = state.state_P(decomp,IC_v_np)
state_w_P    = state.state_P(decomp,IC_w_np)
state_p_P    = state.state_P(decomp,IC_p_np)
state_DP_P   = state.state_P(decomp,IC_p_np)

# Set up a Numpy mirror to the PyTorch state
#  --> Used for file I/O
state_data_all = (state_u_P, state_v_P, state_w_P, state_p_P)
data_all_CPU   = state.data_all_CPU(decomp,startTime,simDt,
                                    names[0:4],state_data_all)

# Allocate a temporary velocity state for RK solvers
if (inputConfig.advancerName[:-1]=="RK"):
    state_uTmp_P = state.state_P(decomp,IC_u_np)
    state_vTmp_P = state.state_P(decomp,IC_v_np)
    state_wTmp_P = state.state_P(decomp,IC_w_np)

# Density
mu  = inputConfig.mu
rho = inputConfig.rho

# Allocate pressure source term and local viscosity
source_P = torch.zeros(nx_,ny_,nz_,dtype=inputConfig.dtypeTorch).to(decomp.device)
VISC_P   = torch.ones(nxo_,nyo_,nzo_,dtype=inputConfig.dtypeTorch).to(decomp.device)
VISC_P.mul_(mu)


# ----------------------------------------------------
# Configure adjoint training
# ----------------------------------------------------
if (inputConfig.adjointTraining):
    # Check for a few prerequisites
    if (inputConfig.SFSmodel!="ML"):
        if (decomp.rank==0):
            raise Exception("\nAdjoint training requires ML subfilter model\n")
    elif (inputConfig.equationMode!="NS"):
        if (decomp.rank==0):
            raise Exception("\nAdjoint training requires Navier-Stokes solver\n")
    else:
        numCheckpointIt = inputConfig.numCheckpointIt
        
        # Allocate the adjoint state
        state_u_adj_P = state.state_P(decomp,IC_zeros_np)
        state_v_adj_P = state.state_P(decomp,IC_zeros_np)
        state_w_adj_P = state.state_P(decomp,IC_zeros_np)

        # Set up a Numpy mirror to the PyTorch adjoint state
        adjoint_data_all = (state_u_adj_P, state_v_adj_P, state_w_adj_P)
        data_adj_CPU = state.data_all_CPU(decomp,startTime,simDt,
                                          names[0:3],adjoint_data_all)

        # Initialize the adjoint RHS object
        adj_rhs1 = adjoint.rhs_adjPredictor(decomp)

        # Allocate temporary adjoint states and rhs objects for RK solvers
        if (inputConfig.advancerName[:-1]=="RK"):
            state_uTmp_adj_P = state.state_P(decomp,IC_zeros_np)
            state_vTmp_adj_P = state.state_P(decomp,IC_zeros_np)
            state_wTmp_adj_P = state.state_P(decomp,IC_zeros_np)
            adj_rhs2 = adjoint.rhs_adjPredictor(decomp)
            adj_rhs3 = adjoint.rhs_adjPredictor(decomp)
            adj_rhs4 = adjoint.rhs_adjPredictor(decomp)
        
        # Allocate space for checkpointed solutions
        #  --> Could be moved to adjoint module
        check_u_P = torch.zeros(nxo_,nyo_,nzo_,numCheckpointIt+1,
                                dtype=inputConfig.dtypeTorch).to(decomp.device)
        check_v_P = torch.zeros(nxo_,nyo_,nzo_,numCheckpointIt+1,
                                dtype=inputConfig.dtypeTorch).to(decomp.device)
        check_w_P = torch.zeros(nxo_,nyo_,nzo_,numCheckpointIt+1,
                                dtype=inputConfig.dtypeTorch).to(decomp.device)

        # Set up to use target data
        useTargetData = True
        numItTargetComp = numCheckpointIt


# ----------------------------------------------------
# Configure SFS model
# ----------------------------------------------------
if (inputConfig.SFSmodel=='Smagorinsky'):
    use_SFSmodel = True
    sfsmodel = sfsmodel_smagorinsky.stress_constCs(geometry,metric,inputConfig.Cs,
                                                   inputConfig.expFilterFac)
elif (inputConfig.SFSmodel=='gradient'):
    use_SFSmodel = True
    sfsmodel = sfsmodel_gradient.residual_stress(decomp,geometry,metric)
elif (inputConfig.SFSmodel=='ML'):
    use_SFSmodel = True
    sfsmodel = sfsmodel_ML.residual_stress(decomp,geometry,metric,
                                           inputConfig.loadModel,inputConfig.modelDictName)
else:
    # Construct a blank SFSmodel object
    use_SFSmodel = False
    sfsmodel = sfsmodel_smagorinsky.stress_constCs(geometry,metric)

# Save the molecular viscosity
if (sfsmodel.modelType=='eddyVisc'):
    muMolec = mu

    
# ----------------------------------------------------
# Configure solver
# ----------------------------------------------------
if (inputConfig.equationMode=='scalar'):
    # Scalar advection-diffusion equations
    if (decomp.rank==0):
        print("\nSolving scalar advection-diffusion equation")

    # Allocate RHS objects (uMax etc. deprecated)
    rhs1 = velocity.rhs_scalar(decomp,uMax,vMax,wMax)
    if (advancerName[:-1]=="RK"):
        rhs2 = velocity.rhs_scalar(decomp,uMax,vMax,wMax)
        rhs3 = velocity.rhs_scalar(decomp,uMax,vMax,wMax)
        rhs4 = velocity.rhs_scalar(decomp,uMax,vMax,wMax)
        
elif (inputConfig.equationMode=='NS'):
    # Navier-Stokes equations
    if (decomp.rank==0):
        print("\nSolving Navier-Stokes equations")
        print("Solver settings: advancer={}, pressure={}"
              .format(inputConfig.advancerName,inputConfig.pSolverMode))

    # Allocate RHS objects    
    rhs1 = velocity.rhs_NavierStokes(decomp)
    if (inputConfig.advancerName[:-1]=="RK"):
        rhs2 = velocity.rhs_NavierStokes(decomp)
        rhs3 = velocity.rhs_NavierStokes(decomp)
        rhs4 = velocity.rhs_NavierStokes(decomp)
        
    # Initialize pressure solver
    if (inputConfig.pSolverMode=='Jacobi'):
        poisson = pressure.solver_jacobi(comms,decomp,metric,
                                         geometry,inputConfig.rho,simDt,
                                         inputConfig.max_pressure_iterations)
    elif (inputConfig.pSolverMode=='bicgstab'):
        poisson = pressure.solver_bicgstab(comms,decomp,metric,geometry,inputConfig.rho,simDt,
                                           inputConfig.min_pressure_residual,
                                           inputConfig.max_pressure_iterations)
    if (inputConfig.pSolverMode=='RedBlackGS'):
        #poisson = pressure.solver_GS_redblack(geometry,rho,simDt,max_pressure_iterations)
        raise Exception('\nRed-black GS not yet implemented\n')
        
else:
    if (decomp.rank==0):
        raise Exception("Equation setting not recognized; consequences unknown...")

# Read restart state data in parallel
if (inputConfig.configName=='restart'):
    if (inputConfig.dataFileType=='restart'):
        dr.readNGArestart_parallel(inputConfig.dataFileStr,data_all_CPU)



# ----------------------------------------------------
# Allocate memory for target state data
# ----------------------------------------------------
if (useTargetData):
    state_u_T = state.state_P(decomp,IC_zeros_np,need_gradients=False)
    state_v_T = state.state_P(decomp,IC_zeros_np,need_gradients=False)
    state_w_T = state.state_P(decomp,IC_zeros_np,need_gradients=False)
    
    #  Set up a Numpy mirror to the target state data
    target_data_all = (state_u_T, state_v_T, state_w_T)
    target_data_all_CPU = state.data_all_CPU(decomp,startTime,simDt,names[0:3],target_data_all)

    # Read the target data file
    #   Adjoint training reads target files in outer iteration loop
    if (not inputConfig.adjointTraining):
        targetDataFileStr = targetFileBaseStr + '{:08d}'.format(inputConfig.startFileIt)
        dr.readNGArestart_parallel(targetDataFileStr,target_data_all_CPU)

    # JFM - for SFS model verification
    #dr.readNGArestart_parallel(dataFileStr,target_data_all_CPU)
    #print(target_data_all_CPU.read(0)[0,0,0])

    
#    # Read target data file
#    xGrid_t,yGrid_t,zGrid_t,names_t,dataTime_t,data_t = dr.readNGA(targetFileStr)
#    # Just save U for now
#    data_target10 = data_t[:,:,:,0]
#    
#    # Clean up
#    del data_t
#    
#    x_max_P  = torch.FloatTensor( const.x_max16 ).to(decomp.device)
#    target_P = torch.FloatTensor( data_target10 ).to(decomp.device)
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
# Pre-simulation monitoring tasks 
# ----------------------------------------------------
    
#for param_group in optimizer.param_groups:
#        param_group['lr'] = 0.1*LR
    
#optimizer.zero_grad()

#u_P = Variable( torch.FloatTensor( IC_u_np ) )
#v_P = Variable( torch.FloatTensor( IC_v_np ) )
#
#Loss = 0.0

# Simulation time
simTime  = startTime
stopTime = startTime + inputConfig.numIt*simDt

# Synchronize the overlap cells before stepping
state_u_P.update_border()
state_v_P.update_border()
state_w_P.update_border()
state_p_P.update_border()

# Write the initial data file
timeStr = "{:12.7E}".format(simTime)
# Root process writes the header
if (decomp.rank==0):
    dr.writeNGArestart(inputConfig.fNameOut+'_'+timeStr,data_all_CPU,True)
# All processes write data
dr.writeNGArestart_parallel(inputConfig.fNameOut+'_'+timeStr,data_all_CPU)

# Write the stdout header
if (inputConfig.equationMode=='NS'):
    if (decomp.rank==0):
        headStr = "  {:10s}   {:9s}   {:9s}   {:9s}   {:9s}   {:9s}   {:9s}   {:9s}   {:9s}"
        print(headStr.format("Step","Time","max CFL","max U","max V","max W","TKE","divergence","max res_P"))
else:
    if (decomp.rank==0):
        headStr = "  {:10s}   {:9s}   {:9s}   {:9s}   {:9s}   {:9s}"
        print(headStr.format("Step","Time","max CFL","max U","max V","max W"))

# Compute resolved kinetic energy and velocity rms
initEnergy = comms.parallel_sum(np.sum( data_all_CPU.read(0)**2 +
                                        data_all_CPU.read(1)**2 +
                                        data_all_CPU.read(2)**2 ))
#rmsVel = np.sqrt(initEnergy/decomp.N)
if (inputConfig.equationMode=='NS'):
    # Compute the initial divergence
    metric.div_vel(state_u_P,state_v_P,state_w_P,source_P)
    maxDivg = comms.parallel_max(torch.max(torch.abs(source_P)).cpu().numpy())
    
# Write initial condition stats to screen
maxU = comms.parallel_max(data_all_CPU.absmax(0))
maxV = comms.parallel_max(data_all_CPU.absmax(1))
maxW = comms.parallel_max(data_all_CPU.absmax(2))
TKE  = comms.parallel_sum(np.sum( data_all_CPU.read(0)**2 +
                                  data_all_CPU.read(1)**2 +
                                  data_all_CPU.read(2)**2 ))*0.5/float(decomp.N)
if (decomp.rank==0):
    maxCFL = max((maxU/geometry.dx,maxV/geometry.dy,maxW/geometry.dz))*simDt
    lineStr = "  {:10d}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}"
    print(lineStr.format(0,simTime,maxCFL,maxU,maxV,maxW,TKE,maxDivg))
    
# Plot the initial state
if (inputConfig.plotState):
    timeStr = "{:12.7E}_{}".format(simTime,decomp.rank)
    # Plot the initial state
    decomp.plot_fig_root(dr,state_u_P.var,"state_U_"+str(0)+"_"+timeStr)

    
# ----------------------------------------------------
# Main simulation loop
# ----------------------------------------------------
    
time1 = time.time()

# Total iteration counter
itCount = 0

# Configure the main simulation loop
if (inputConfig.adjointTraining):
    # Adjoint training: divide outer loop into checkpointed inner loops
    numStepsOuter = inputConfig.numIt//numCheckpointIt
    numStepsInner = numCheckpointIt
else:
    # Forward solver only
    numStepsOuter = 1
    numStepsInner = numIt    
    
# Here we go
for itCountOuter in range(numStepsOuter):
    
    # Reset the inner iteration counter
    itCountInner = 0

    # Checkpoint the velocity initial condition
    if (inputConfig.adjointTraining):
        check_u_P[:,:,:,itCountInner].copy_(state_u_P.var)
        check_v_P[:,:,:,itCountInner].copy_(state_v_P.var)
        check_w_P[:,:,:,itCountInner].copy_(state_w_P.var)
    
    # ----------------------------------------------------
    # Forward inner loop
    # ----------------------------------------------------
    while (simTime < stopTime and itCountInner < numStepsInner):

        # [JFM] need new sub-iteration loop
        
        # ----------------------------------------------------
        # Velocity prediction step
        # ----------------------------------------------------

        # Evaluate SFS model
        if (use_SFSmodel):
            if (sfsmodel.modelType=='eddyVisc'):
                muEddy = sfsmodel.eddyVisc(state_u_P,state_v_P,state_w_P,rho,metric)
                VISC_P.copy_( muMolec + muEddy )
            elif (sfsmodel.modelType=='tensor'):
                sfsmodel.update(state_u_P,state_v_P,state_w_P,metric)
            # --> Source-type models: evaluate inside the RHS
            #elif (sfsmodel.modelType=='source'):
            #    sfsmodel.update(state_u_P,state_v_P,state_w_P,metric)
                
        # Compute velocity prediction
        if (inputConfig.advancerName=="Euler"):
            # rhs
            rhs1.evaluate(state_u_P,state_v_P,state_w_P,VISC_P,rho,sfsmodel,metric)

            # Update the state
            state_u_P.var = state_u_P.var + rhs1.rhs_u*simDt
            state_v_P.var = state_v_P.var + rhs1.rhs_v*simDt
            state_w_P.var = state_w_P.var + rhs1.rhs_w*simDt

        elif (inputConfig.advancerName=="RK4"):
            
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

        if (inputConfig.equationMode=='NS'):
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
        # Checkpoint the velocity solution
        # ----------------------------------------------------
        if (inputConfig.adjointTraining):
            check_u_P[:,:,:,itCountInner+1].copy_(state_u_P.var)
            check_v_P[:,:,:,itCountInner+1].copy_(state_v_P.var)
            check_w_P[:,:,:,itCountInner+1].copy_(state_w_P.var)
            
                      
        # ----------------------------------------------------
        # Post-step tasks
        # ----------------------------------------------------
        # Update the counters
        itCountInner += 1
        itCount += 1
        simTime += simDt
        simTimeCheckpoint = simTime
        
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
        if (inputConfig.equationMode=='NS'):
            # Compute the final divergence
            metric.div_vel(state_u_P,state_v_P,state_w_P,source_P)
            maxDivg = comms.parallel_max(torch.max(torch.abs(source_P)).cpu().numpy())
        
        # Write stats
        if (inputConfig.equationMode=='NS'):
            if (decomp.rank==0):
                lineStr = "  {:10d}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}    {:8.3E}"
                print(lineStr.format(itCount,simTime,maxCFL,maxU,maxV,maxW,TKE,maxDivg,max_resP))
        else:
            if (decomp.rank==0):
                lineStr = "  {:10d}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}"
                print(lineStr.format(itCount,simTime,maxCFL,maxU,maxV,maxW))

        # Write output
        if (np.mod(itCount,inputConfig.numItDataOut)==0):
            # Write data to disk
            data_all_CPU.time = simTime
            data_all_CPU.dt   = simDt
            timeStr = "{:12.7E}".format(simTime)
            if (decomp.rank==0):
                dr.writeNGArestart(inputConfig.fNameOut+'_'+timeStr,data_all_CPU,True)
            dr.writeNGArestart_parallel(inputConfig.fNameOut+'_'+timeStr,data_all_CPU)

        if (inputConfig.plotState and np.mod(itCount,inputConfig.numItPlotOut)==0):
            timeStr = "{:12.7E}_{}".format(simTime,decomp.rank)
            decomp.plot_fig_root(dr,state_u_P.var,"state_U_"+str(itCount)+"_"+timeStr)


        # Compare to target DNS data
        if (useTargetData and np.mod(itCount,numItTargetComp)==0 and not inputConfig.adjointTraining):
            # Only on root processor for now
            targetFileIt  = inputConfig.startFileIt+itCount
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
                                        target_data_all_CPU.read(0) )))
            #/(geometry.Nx*geometry.Ny*geometry.Nz)
                                          
            if (decomp.rank==0):
                print("     Max(U) sim: {:10.5E}, Max(U) target: {:10.5E}".format(maxU_sim,maxU_t))
                print("     L1 error  : {:10.5E}".format(L1_error))
            
    ## END OF FORWARD INNER LOOP

        
        
    # ----------------------------------------------------
    # Adjoint inner loop
    # ----------------------------------------------------
    if (inputConfig.adjointTraining):
        itCountInner = numStepsInner
        itCountInnerUp = 0

        # Load target state
        targetDataFileStr = inputConfig.dataFileBStr + '{:08d}'.format(inputConfig.startFileIt+itCount)
        dr.readNGArestart_parallel(targetDataFileStr,target_data_all_CPU)

        # Set the adjoint initial condition to the mean absolute error
        state_u_adj_P.var.copy_( torch.sign(state_u_P.var - state_u_T.var) )
        state_v_adj_P.var.copy_( torch.sign(state_v_P.var - state_v_T.var) )
        state_w_adj_P.var.copy_( torch.sign(state_w_P.var - state_w_T.var) )
        #state_u_adj_P.var.zero_()
        #state_v_adj_P.var.zero_()
        #state_w_adj_P.var.zero_()
        # Normalize
        state_u_adj_P.var.div_ ( nx*ny*nz )
        state_v_adj_P.var.div_ ( nx*ny*nz )
        state_w_adj_P.var.div_ ( nx*ny*nz )

        if (decomp.rank==0):
            print('Starting adjoint iteration')
        
        while (itCountInner > 0):

            # Load the checkpointed velocity solution at time 't'
            #   Overlap cells are already synced in the checkpointed solutions
            # --> JFM: check correct time is loaded??
            state_u_P.var.copy_( check_u_P[:,:,:,itCountInner] )
            state_v_P.var.copy_( check_v_P[:,:,:,itCountInner] )
            state_w_P.var.copy_( check_w_P[:,:,:,itCountInner] )

            
            # ----------------------------------------------------
            # Adjoint 'pressure' iteration
            # ----------------------------------------------------
            # Divergence of the adjoint velocity field
            metric.div_vel(state_u_adj_P,state_v_adj_P,state_w_adj_P,source_P)

            # Solve the Poisson equation
            max_resP = poisson.solve(state_DP_P,state_p_P,source_P)

            
            # ----------------------------------------------------
            # Adjoint corrector step: \hat{u}^*
            # ----------------------------------------------------
            # Compute 'pressure' gradients
            metric.grad_P(state_DP_P)

            # Update the adjoint solution
            #   Note negative sign
            state_u_adj_P.vel_corr(state_DP_P.grad_x, -simDt/rho)
            state_v_adj_P.vel_corr(state_DP_P.grad_y, -simDt/rho)
            state_w_adj_P.vel_corr(state_DP_P.grad_z, -simDt/rho)

            
            # ----------------------------------------------------
            # Adjoint predictor step: \hat{u}^t
            # ----------------------------------------------------
            if (inputConfig.advancerName=="Euler"):
                
                # Adjoint equation rhs
                adj_rhs1.evaluate(state_u_adj_P,state_v_adj_P,state_w_adj_P,
                                  state_u_P,state_v_P,state_w_P,VISC_P,rho,sfsmodel,metric)
                
                # Update the adjoint state
                state_u_adj_P.update( state_u_adj_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1] + simDt*adj_rhs1.rhs_u )
                state_v_adj_P.update( state_v_adj_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1] + simDt*adj_rhs1.rhs_v )
                state_w_adj_P.update( state_w_adj_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1] + simDt*adj_rhs1.rhs_w )

            elif (inputConfig.advancerName=="RK4"):
                
                # Stage 1
                adj_rhs1.evaluate(state_u_adj_P,state_v_adj_P,state_w_adj_P,
                                  state_u_P,state_v_P,state_w_P,VISC_P,rho,sfsmodel,metric)
                # Stage 2
                state_uTmp_adj_P.ZAXPY(0.5*simDt,adj_rhs1.rhs_u,state_u_adj_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1])
                state_vTmp_adj_P.ZAXPY(0.5*simDt,adj_rhs1.rhs_v,state_v_adj_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1])
                state_wTmp_adj_P.ZAXPY(0.5*simDt,adj_rhs1.rhs_w,state_w_adj_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1])
                adj_rhs2.evaluate(state_uTmp_adj_P,state_vTmp_adj_P,state_wTmp_adj_P,
                                  state_u_P,state_v_P,state_w_P,VISC_P,rho,sfsmodel,metric)
                # Stage 3
                state_uTmp_adj_P.ZAXPY(0.5*simDt,adj_rhs2.rhs_u,state_u_adj_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1])
                state_vTmp_adj_P.ZAXPY(0.5*simDt,adj_rhs2.rhs_v,state_v_adj_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1])
                state_wTmp_adj_P.ZAXPY(0.5*simDt,adj_rhs2.rhs_w,state_w_adj_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1])
                adj_rhs3.evaluate(state_uTmp_adj_P,state_vTmp_adj_P,state_wTmp_adj_P,
                                  state_u_P,state_v_P,state_w_P,VISC_P,rho,sfsmodel,metric)
                # Stage 4
                state_uTmp_adj_P.ZAXPY(0.5*simDt,adj_rhs3.rhs_u,state_u_adj_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1])
                state_vTmp_adj_P.ZAXPY(0.5*simDt,adj_rhs3.rhs_v,state_v_adj_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1])
                state_wTmp_adj_P.ZAXPY(0.5*simDt,adj_rhs3.rhs_w,state_w_adj_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1])
                adj_rhs4.evaluate(state_uTmp_adj_P,state_vTmp_adj_P,state_wTmp_adj_P,
                                  state_u_P,state_v_P,state_w_P,VISC_P,rho,sfsmodel,metric)

                # Update the adjoint state
                state_u_adj_P.update( state_u_adj_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1]
                                      + simDt/6.0*(adj_rhs1.rhs_u + 2.0*adj_rhs2.rhs_u + 2.0*adj_rhs3.rhs_u + adj_rhs4.rhs_u) )
                state_v_adj_P.update( state_v_adj_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1]
                                      + simDt/6.0*(adj_rhs1.rhs_v + 2.0*adj_rhs2.rhs_v + 2.0*adj_rhs3.rhs_v + adj_rhs4.rhs_v) )
                state_w_adj_P.update( state_w_adj_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1]
                                      + simDt/6.0*(adj_rhs1.rhs_w + 2.0*adj_rhs2.rhs_w + 2.0*adj_rhs3.rhs_w + adj_rhs4.rhs_w) )
                
            
            # ----------------------------------------------------
            # Post-step tasks
            # ----------------------------------------------------
            # Update the counters
            itCountInner -= 1
            itCountInnerUp += 1
            simTime -= simDt
            
            # Compute stats
            maxU = comms.parallel_max(data_adj_CPU.absmax(0))
            maxV = comms.parallel_max(data_adj_CPU.absmax(1))
            maxW = comms.parallel_max(data_adj_CPU.absmax(2))
            maxCFL = max((maxU/geometry.dx,maxV/geometry.dy,maxW/geometry.dz))*simDt

            # Print stats
            if (decomp.rank==0):
                lineStr = "  Adj {:6d}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}   {:9s}   {:9s}    {:8.3E}"
                print(lineStr.format(itCount-itCountInnerUp,simTime,maxCFL,maxU,maxV,maxW,
                                     ' ',' ',max_resP))
            
        ## END OF ADJOINT INNER LOOP

        # Multiply neural network accumlulated gradients by LES time step
        for param in sfsmodel.model.parameters():
            param.grad.data *= simDt

        # Sync the ML model across processes
        for param in sfsmodel.model.parameters():
            tensor0   = param.grad.data.cpu().numpy()
            tensorAvg = comms.parallel_sum(tensor0.ravel())/float(comms.size)
            tensorOut = torch.tensor(tensorAvg.reshape(np.shape(tensor0)))
            param.grad.data = tensorOut

        # Write the ML model to disk
        if (saveModel and decomp.rank==0):
            print('Saving model...')
            torch.save(sfsmodel.model.state_dict(),modelDictSave)
                
        # Resource utilization
        if (decomp.rank==0):
            mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            mem_usage /= 1e9
            print('Done adjoint iteration, peak mem={:7.5f} GB'.format(itCountInner,mem_usage))

        # Reload last checkpointed velocity solution
        state_u_P.var.copy_( check_u_P[:,:,:,numStepsInner] )
        state_v_P.var.copy_( check_v_P[:,:,:,numStepsInner] )
        state_w_P.var.copy_( check_w_P[:,:,:,numStepsInner] )

        # Restore simulation time
        simTime = simTimeCheckpoint
        
    ## END OF ADJOINT TRAINING

## END OF MAIN SIMULATION LOOP
            
        
# ----------------------------------------------------
# Post-simulation tasks
# ----------------------------------------------------
    
# Write the final state to disk
data_all_CPU.time = simTime
data_all_CPU.dt   = simDt
timeStr = "{:12.7E}".format(simTime)
if (decomp.rank==0):
    dr.writeNGArestart(inputConfig.fNameOut+'_'+timeStr,data_all_CPU,True)
dr.writeNGArestart_parallel(inputConfig.fNameOut+'_'+timeStr,data_all_CPU)
            
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
        print(itCount,test,error,time_elapsed)
else:
    if (decomp.rank==0):
        print("it={}, test={:10.5E}, elapsed={}".format(itCount,test,time_elapsed))
        print("Energy initial={:10.5E}, final={:10.5E}, ratio={:10.5E}".format(initEnergy,finalEnergy,
                                                                               finalEnergy/initEnergy))

if (inputConfig.plotState):
    # Print a pretty picture
    timeStr = "{:12.7E}_{}".format(simTime,decomp.rank)
    decomp.plot_fig_root(dr,state_u_P.var,"state_U_"+str(itCount)+"_"+timeStr)


    

## END main
    

#if __name__ == "__main__":
#    main(sys.argv[1:])
