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

import time
import copy
import os

import resource

# Load PyFlow modules
sys.path.append("../data")
import domain
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


####### TODO
#  1. Non-periodic BCs
#  2. Midpoint fractional-step
#  3. RK3
#  4. Non-uniform grid


def run(inputConfig):

    # ----------------------------------------------------
    # Configure simulation runtime
    # ----------------------------------------------------
    # Basic parallel communication
    comms = parallel.comms(inputConfig)

    # Configure grid sizes
    config = geo.config(inputConfig)
    
    # Domain decomposition
    decomp = parallel.decomp(inputConfig,config)
    numPoints = decomp.nx*decomp.ny*decomp.nz

    if ((decomp.npx>1 and decomp.npy>1) or (decomp.npz>1)):
        inputConfig.plotState = False

    # System type
    memDiv = 1e6
    if (sys.platform=="linux"):
        memDiv = 1e6
    elif (sys.platform=="darwin"):
        memDiv = 1e9
    
    # ----------------------------------------------------
    # Generate initial conditions
    # ----------------------------------------------------
    names,startTime,data_IC,xGrid,yGrid,zGrid = \
        initial_conditions.generate(inputConfig,config,decomp)
    
    # ----------------------------------------------------
    # Configure geometry and metrics
    # ----------------------------------------------------
    geometry = geo.uniform(xGrid,yGrid,zGrid,decomp)
    metric   = metric_staggered.metric_uniform(geometry)
    
    # ----------------------------------------------------
    # Configure domain: velocity and adjoint, SFS model
    # ----------------------------------------------------
    D = domain.Domain(inputConfig,comms,decomp,data_IC,geometry,
                      metric,names,startTime)
    
    # ----------------------------------------------------
    # Read restart data
    # ----------------------------------------------------
    if (inputConfig.configName=='restart'):
        if (inputConfig.dataFileType=='restart'):
            dr.readNGArestart_parallel(inputConfig.dataFileStr,D.data_all_CPU)

    # JFM - for SFS model verification
    #dr.readNGArestart_parallel(dataFileStr,target_data_all_CPU)
    #print(target_data_all_CPU.read(0)[0,0,0])
    
    # Clean up
    del data_IC


    # ----------------------------------------------------
    # Adjoint-based model training
    # ----------------------------------------------------
    adjointTraining = False
    try:
        adjointTraining = inputConfig.adjointTraining
        if (adjointTraining):
            if (decomp.rank==0):
                print("Performing adjoint-based model training")
        else:
            if (decomp.rank==0):
                print("Not performing adjoint-based model training")
    except:
        if (decomp.rank==0):
            print("Adjoint-based training settings not specified")


    # ----------------------------------------------------
    # Adjoint verification
    # ----------------------------------------------------
    adjointVerification = False
    try:
        adjointVerification = inputConfig.adjointVerification
        if (adjointVerification):
            if (decomp.rank==0):
                print("Performing adjoint verification, perturbation={}"
                      .format(inputConfig.perturbation))
            # Perturb the initial condition
            nn = geometry.imin_ + geometry.nx_//2
            D.state_u_P.var[nn,nn,nn] += inputConfig.perturbation

            new_obj = 0.0
            new_adj = 0.0
            
        else:
            if (decomp.rank==0):
                print("Not performing adjoint verification")
    except:
        if (decomp.rank==0):
            print("Adjoint verification settings not specified")


    # ----------------------------------------------------
    # Statistics evaluation
    # ----------------------------------------------------
    advanceSimulation = True
    computeStatsOnly   = False
    try:
        computeStatsOnly = inputConfig.computeStatsOnly
        if (computeStatsOnly):
            # Don't advance the simulation; set outer loop to skip
            # data files as necessary for stats
            advanceSimulation = False
            if (decomp.rank==0):
                print("Computing statistics on input data")
        else:
            if (decomp.rank==0):
                print("Not computing statistics")
    except:
        if (decomp.rank==0):
            print("Statistics settings not specified")

        
    # ----------------------------------------------------
    # Pre-simulation monitoring tasks
    # ----------------------------------------------------
    
    # Simulation time
    simDt    = inputConfig.simDt
    simTime  = startTime
    stopTime = startTime + inputConfig.numIt*simDt

    # Synchronize the overlap cells before stepping
    D.state_u_P.update_border()
    D.state_v_P.update_border()
    D.state_w_P.update_border()
    D.state_p_P.update_border()

    # Write the initial data file
    timeStr = "{:12.7E}".format(simTime)
    # Root process writes the header
    if (decomp.rank==0):
        dr.writeNGArestart(inputConfig.fNameOut+'_'+timeStr,D.data_all_CPU,True)
    # All processes write data
    dr.writeNGArestart_parallel(inputConfig.fNameOut+'_'+timeStr,D.data_all_CPU)
    
    # Write the stdout header
    if (inputConfig.equationMode=='NS'):
        if (decomp.rank==0):
            headStr = "\n  {:10s}   {:9s}   {:9s}   {:9s}   {:9s}   {:9s}   {:9s}   {:9s}   {:9s}"
            print(headStr.format("Step","Time","max CFL","max U","max V","max W","TKE",
                                 "divergence","max res_P"))
    else:
        if (decomp.rank==0):
            headStr = "  {:10s}   {:9s}   {:9s}   {:9s}   {:9s}   {:9s}"
            print(headStr.format("Step","Time","max CFL","max U","max V","max W"))

    # Compute resolved kinetic energy and velocity rms
    initEnergy = comms.parallel_sum(np.sum( D.data_all_CPU.read(0)**2 +
                                            D.data_all_CPU.read(1)**2 +
                                            D.data_all_CPU.read(2)**2 ))
    #rmsVel = np.sqrt(initEnergy/decomp.N)
    if (inputConfig.equationMode=='NS'):
        # Compute the initial divergence
        metric.div_vel(D.state_u_P,D.state_v_P,D.state_w_P,D.source_P)
        maxDivg = comms.parallel_max(torch.max(torch.abs(D.source_P)).cpu().numpy())
    
    # Write initial condition stats to screen
    maxU = comms.parallel_max(D.data_all_CPU.absmax(0))
    maxV = comms.parallel_max(D.data_all_CPU.absmax(1))
    maxW = comms.parallel_max(D.data_all_CPU.absmax(2))
    TKE  = comms.parallel_sum(np.sum( D.data_all_CPU.read(0)**2 +
                                      D.data_all_CPU.read(1)**2 +
                                      D.data_all_CPU.read(2)**2 ))*0.5/float(decomp.N)
    if (decomp.rank==0):
        maxCFL = max((maxU/geometry.dx,maxV/geometry.dy,maxW/geometry.dz))*simDt
        lineStr = "  {:10d}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}"
        print(lineStr.format(0,simTime,maxCFL,maxU,maxV,maxW,TKE,maxDivg))

    # Set up the monitor file
    if (inputConfig.equationMode=='NS' and decomp.rank==0):
        if (not os.path.exists('monitor')):
            os.mkdir('monitor')
        try:
            caseName = inputConfig.caseName
        except:
            caseName = "PyFlow"
        monitorFileName = "monitor/velocity_"+caseName+".txt"
        monitorFile     = open(monitorFileName,'w')
        monitorHeadStr = "  {:10s}   {:9s}   {:9s}   {:9s}   {:9s}   {:9s}   {:9s}   {:9s}   {:9s} \n"
        monitorLineStr = "  {:10d}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}\n"
        monitorFile.write(monitorHeadStr.format("Step","Time","max CFL","max U","max V","max W","TKE",
                                                "divergence","max res_P"))
        monitorFile.write(monitorLineStr.format(0,simTime,maxCFL,maxU,maxV,maxW,TKE,maxDivg))
            
    # Plot the initial state
    if (inputConfig.plotState):
        timeStr = "{:12.7E}_{}".format(simTime,decomp.rank)
        # Plot the initial state
        decomp.plot_fig_root(dr,D.state_u_P.var,"state_U_"+str(0)+"_"+timeStr)

    
    # ----------------------------------------------------
    # Main simulation loop
    # ----------------------------------------------------
    
    time1 = time.time()
    
    # Total iteration counter
    itCount = 0

    # Inner loop step size
    stepSizeInner = 1
    
    # Configure the main simulation loop
    if (adjointTraining):
        # Adjoint training: divide outer loop into checkpointed inner loops
        numStepsOuter = inputConfig.numIt//D.numCheckpointIt
        numStepsInner = D.numCheckpointIt
    elif (computeStatsOnly):
        # Statistics only: read data files in outer loop
        numStepsOuter = inputConfig.numIt//D.numCheckpointIt
        numStepsInner = 1
        stepSizeInner = D.numCheckpointIt
    else:
        # Forward solver only
        numStepsOuter = 1
        numStepsInner = inputConfig.numIt    
    
    # Here we go
    for itCountOuter in range(numStepsOuter):
        
        # Reset the inner iteration counter
        itCountInner = 0
        
        # Checkpoint the velocity initial condition
        if (adjointTraining):
            D.check_u_P[:,:,:,itCountInner].copy_(D.state_u_P.var)
            D.check_v_P[:,:,:,itCountInner].copy_(D.state_v_P.var)
            D.check_w_P[:,:,:,itCountInner].copy_(D.state_w_P.var)

        # Statistics evaluation only
        #   --> Load the next restart data file
        if (computeStatsOnly):
            if (inputConfig.dataFileType=='restart'):
                fileIt = itCount + stepSizeInner + inputConfig.startFileIt
                dataFileStr = inputConfig.dataFileBStr + '{:08d}'.format(fileIt)
                dr.readNGArestart_parallel(dataFileStr,D.data_all_CPU)
    
        # ----------------------------------------------------
        # Forward inner loop
        # ----------------------------------------------------
        while (simTime < stopTime and itCountInner < numStepsInner):

            # ----------------------------------------------------
            # Evaluate the forward step
            #
            if (advanceSimulation):
                D.forwardStep(simDt)
        
            # ----------------------------------------------------
            # Checkpoint the velocity solution
            #
            if (adjointTraining):
                D.check_u_P[:,:,:,itCountInner+1].copy_(D.state_u_P.var)
                D.check_v_P[:,:,:,itCountInner+1].copy_(D.state_v_P.var)
                D.check_w_P[:,:,:,itCountInner+1].copy_(D.state_w_P.var)
                
            # ----------------------------------------------------
            # Post-step tasks
            #
            # Update the counters
            itCountInner += stepSizeInner
            itCount += stepSizeInner
            simTime += simDt*stepSizeInner
            simTimeCheckpoint = simTime
            
            # Compute stats
            viscNu = inputConfig.mu/inputConfig.rho
            maxU = comms.parallel_max(D.data_all_CPU.absmax(0))
            maxV = comms.parallel_max(D.data_all_CPU.absmax(1))
            maxW = comms.parallel_max(D.data_all_CPU.absmax(2))
            maxCFL = max((maxU/geometry.dx,maxV/geometry.dy,maxW/geometry.dz))*simDt
            maxVNN = max((viscNu/geometry.dx**2,viscNu/geometry.dy**2,
                          viscNu/geometry.dz**2))*simDt
            # This works but won't tell user whether CFL or VNN is limiting
            maxCFL = max((maxCFL,maxVNN))
            TKE  = comms.parallel_sum(np.sum( D.data_all_CPU.read(0)**2 +
                                              D.data_all_CPU.read(1)**2 +
                                              D.data_all_CPU.read(2)**2 )) * \
                                              0.5/float(decomp.N)
            if (inputConfig.equationMode=='NS'):
                # Compute the final divergence
                metric.div_vel(D.state_u_P,D.state_v_P,D.state_w_P,D.source_P)
                maxDivg = comms.parallel_max(torch.max(torch.abs(D.source_P)).cpu().numpy())
                
            # Write stats
            if (inputConfig.equationMode=='NS'):
                if (decomp.rank==0):
                    lineStr = "  {:10d}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}"
                    print(lineStr.format(itCount,simTime,maxCFL,maxU,maxV,maxW,
                                         TKE,maxDivg,D.max_resP))
            else:
                if (decomp.rank==0):
                    print(lineStr.format(itCount,simTime,maxCFL,maxU,maxV,maxW))

            # Write to monitor file
            if (decomp.rank==0 and inputConfig.equationMode=='NS'):
                monitorLineStr = "  {:10d}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}\n"
                monitorFile.write(monitorLineStr.format(itCount,simTime,maxCFL,maxU,maxV,maxW,
                                                        TKE,maxDivg,D.max_resP))
                    
            # Write output
            if (np.mod(itCount,inputConfig.numItDataOut)==0):
                # Write data to disk
                D.data_all_CPU.time = simTime
                D.data_all_CPU.dt   = simDt
                timeStr = "{:12.7E}".format(simTime)
                if (decomp.rank==0):
                    dr.writeNGArestart(inputConfig.fNameOut+'_'+timeStr,D.data_all_CPU,True)
                dr.writeNGArestart_parallel(inputConfig.fNameOut+'_'+timeStr,D.data_all_CPU)

            # Plot data
            if (inputConfig.plotState and np.mod(itCount,inputConfig.numItPlotOut)==0):
                timeStr = "{:12.7E}_{}".format(simTime,decomp.rank)
                decomp.plot_fig_root(dr,D.state_u_P.var,"state_U_"+str(itCount)+"_"+timeStr)

            # Compare to target DNS data
            if (D.useTargetData and np.mod(itCount,D.numItTargetComp)==0 and \
                not adjointTraining):
                # Only on root processor for now
                targetFileIt  = inputConfig.startFileIt+itCount
                targetFileStr = inputConfig.targetFileBaseStr + str(targetFileIt)

                # Check to make sure we read at the right time
                if (decomp.rank==0):
                    names_t,simTime_t = dr.readNGArestart(targetFileStr,printOut=False)
                    if (False): #(simTime!=simTime_t):
                        raise Exception("\nPyFlow: target file not at same time as simulation\n")
                    else:
                        print(" --> Comparing to target data file {} at time {:10.5E}"
                              .format(targetFileIt,simTime_t))

                # Read the target state data
                #dr.readNGArestart_parallel(targetFileStr,target_data_all_CPU,ivar_read_start=0,nvar_read=3)
                #names_t,simTime_t,data_t = dr.readNGArestart(targetFileStr,headerOnly=False,printOut=False)
                #target_data_all_CPU.append(0,data_t[:,:,:,0])
                
                # L1 error of x-velocity field
                print(D.data_all_CPU.read(0)[0,0,0])
                print(np.max(D.target_data_all_CPU.read(0)))
                print(D.target_data_all_CPU.read(0)[0,0,0])
                maxU_sim = comms.parallel_max(np.max(np.abs( D.data_all_CPU.read(0) )))
                maxU_t   = comms.parallel_max(np.max(np.abs( D.target_data_all_CPU.read(0) )))
                L1_error = (np.mean(np.abs( D.data_all_CPU.read(0) -
                                            D.target_data_all_CPU.read(0) )))
                #/(geometry.Nx*geometry.Ny*geometry.Nz)
                
                if (decomp.rank==0):
                    print("     Max(U) sim: {:10.5E}, Max(U) target: {:10.5E}"
                          .format(maxU_sim,maxU_t))
                    print("     L1 error  : {:10.5E}".format(L1_error))
                    
        ## END OF FORWARD INNER LOOP
        
        
        
        # ----------------------------------------------------
        # Adjoint inner loop
        # ----------------------------------------------------
        if (adjointTraining):
            itCountInnerUp = 0
            
            # Pre-adjoint step tasks
            if (adjointVerification):
                # --> Evaluate and print the objective function
                new_obj = torch.mean( (D.state_u_P.interior() - 0.0)**2 ).numpy()
                print("Objective function: {}".format(new_obj))

                # --> Load the target state
                D.state_u_adj_P.var.copy_( 2.0*(D.state_u_P.var - 0.0) )
                D.state_v_adj_P.var.copy_( 0.0*(D.state_v_P.var - 0.0) )
                D.state_w_adj_P.var.copy_( 0.0*(D.state_w_P.var - 0.0) )

            else:
                targetDataFileStr = inputConfig.dataFileBStr + \
                    '{:08d}'.format(inputConfig.startFileIt+itCount)
                dr.readNGArestart_parallel(targetDataFileStr,D.target_data_all_CPU)
            
                # Set the adjoint initial condition to the mean absolute error
                D.state_u_adj_P.var.copy_( torch.sign(D.state_u_P.var - D.state_u_T.var) )
                D.state_v_adj_P.var.copy_( torch.sign(D.state_v_P.var - D.state_v_T.var) )
                D.state_w_adj_P.var.copy_( torch.sign(D.state_w_P.var - D.state_w_T.var) )

                # Compute the error vs. the target data
                model_error  = comms.parallel_sum(torch.sum(torch.abs( D.state_u_P.interior() -
                                                                       D.state_u_T.interior() ))
                                                  .cpu().numpy()) / float(numPoints)
                model_error += comms.parallel_sum(torch.sum(torch.abs( D.state_v_P.interior() -
                                                                       D.state_v_T.interior() ))
                                                  .cpu().numpy()) / float(numPoints)
                model_error += comms.parallel_sum(torch.sum(torch.abs( D.state_w_P.interior() -
                                                                       D.state_w_T.interior() ))
                                                  .cpu().numpy()) / float(numPoints)
                
            # Normalize
            D.state_u_adj_P.var.div_( numPoints )
            D.state_v_adj_P.var.div_( numPoints )
            D.state_w_adj_P.var.div_( numPoints )
            
            if (decomp.rank==0):
                print('Starting adjoint iteration')
                
            while (itCountInner > 0):

                # ----------------------------------------------------
                # Load the checkpointed velocity solution at time t+1
                #   Overlap cells are already synced in the
                #   checkpointed solutions
                # --> JFM: check correct time is loaded??
                D.state_u_P.var.copy_( D.check_u_P[:,:,:,itCountInner-1] )
                D.state_v_P.var.copy_( D.check_v_P[:,:,:,itCountInner-1] )
                D.state_w_P.var.copy_( D.check_w_P[:,:,:,itCountInner-1] )
                
                # ----------------------------------------------------
                # Evaluate the adjoint step
                #
                D.adjointStep(simDt)
                
                # ----------------------------------------------------
                # Post-step tasks
                #
                # Update the counters
                itCountInner -= 1
                itCountInnerUp += 1
                simTime -= simDt
                
                # Compute stats
                maxU = comms.parallel_max(D.data_adj_CPU.absmax(0))
                maxV = comms.parallel_max(D.data_adj_CPU.absmax(1))
                maxW = comms.parallel_max(D.data_adj_CPU.absmax(2))
                maxCFL = max((maxU/geometry.dx,maxV/geometry.dy,maxW/geometry.dz))*simDt
                
                # Print stats
                if (decomp.rank==0):
                    lineStr = "  Adj {:6d}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}   {:8.3E}   {:9s}   {:9s}    {:8.3E}"
                    print(lineStr.format(itCount-itCountInnerUp,simTime,maxCFL,
                                         maxU,maxV,maxW,
                                         ' ',' ',D.max_resP))
            
            ## END OF ADJOINT INNER LOOP
            
            if (adjointVerification):
                new_adj = D.state_u_adj_P.var[nn,nn,nn].numpy()
                print("Final adjoint state: {}".format(new_adj))

            # Sync the ML across processors and save to disk
            if (not adjointVerification):
                D.sfsmodel.finalize(comms)
                if (decomp.rank==0):
                    D.sfsmodel.save()
            
            # Resource utilization
            if (decomp.rank==0):
                mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                mem_usage /= memDiv
                print('Done adjoint iteration, peak mem={:7.5f} GB, training error={:12.5E}'
                      .format(mem_usage,model_error))
                
            # Restore last checkpointed velocity solution
            D.state_u_P.var.copy_( D.check_u_P[:,:,:,numStepsInner] )
            D.state_v_P.var.copy_( D.check_v_P[:,:,:,numStepsInner] )
            D.state_w_P.var.copy_( D.check_w_P[:,:,:,numStepsInner] )
            
            # Restore simulation time
            simTime = simTimeCheckpoint
            
        ## END OF ADJOINT TRAINING
        
    ## END OF MAIN SIMULATION LOOP
            
        
    # ----------------------------------------------------
    # Post-simulation tasks
    # ----------------------------------------------------
    
    # Write the final state to disk
    D.data_all_CPU.time = simTime
    D.data_all_CPU.dt   = simDt
    timeStr = "{:12.7E}".format(simTime)
    if (decomp.rank==0):
        dr.writeNGArestart(inputConfig.fNameOut+'_'+timeStr,D.data_all_CPU,True)
    dr.writeNGArestart_parallel(inputConfig.fNameOut+'_'+timeStr,D.data_all_CPU)
    
    #if (useTargetData):
    #    Diff = D.state_u_P.var - target_P
    #    Loss_i = torch.mean( torch.abs( Diff ) )
    #    Loss = Loss + Loss_i
    #    error = np.mean(np.abs( D.state_u_P.var.cpu().numpy() -  target_P.cpu().numpy() ) )
        
    #Loss_np = Loss.cpu().numpy()

    time2 = time.time()
    time_elapsed = time2 - time1
    
    test = torch.mean( D.state_u_P.var)
    
    # Compute the final energy
    finalEnergy = comms.parallel_sum(np.sum( D.data_all_CPU.read(0)**2 +
                                             D.data_all_CPU.read(1)**2 +
                                             D.data_all_CPU.read(2)**2 ))*0.5
    
    if (False): #useTargetData):
        if (decomp.rank==0):
            print(itCount,test,error,time_elapsed)
    else:
        if (decomp.rank==0):
            print("\nit={}, test={:10.5E}, elapsed={}".format(itCount,test,time_elapsed))
            print("Energy initial={:10.5E}, final={:10.5E}, ratio={:10.5E}"
                  .format(initEnergy,finalEnergy,finalEnergy/initEnergy))

    # Close the monitor file
    if (decomp.rank==0):
        monitorFile.close()
            
    if (inputConfig.plotState):
        # Print a pretty picture
        timeStr = "{:12.7E}_{}".format(simTime,decomp.rank)
        decomp.plot_fig_root(dr,state_u_P.var,"state_U_"+str(itCount)+"_"+timeStr)
        
    # Return data as required
    if (adjointVerification):
        return new_obj,new_adj
    else:
        return

    
## END run
    
