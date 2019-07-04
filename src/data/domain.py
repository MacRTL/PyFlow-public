# ------------------------------------------------------------------------
#
# PyFlow: A GPU-accelerated CFD platform written in Python
#
# @file domain.py
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

# Load PyFlow modules
#
import state
#
sys.path.append("../solver")
import velocity
import adjoint
import advancer
import pressure
#
sys.path.append("../sfsmodel")
import sfsmodel_ML
import sfsmodel_smagorinsky
import sfsmodel_gradient


# ----------------------------------------------------
# Domain class
# ----------------------------------------------------
class Domain:
    def __init__(self,inputConfig,comms,decomp,data_IC,geometry,
                 metric,names,startTime):
        
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
        # Set up initial condition
        IC_u_np = data_IC[:,:,:,0]
        IC_v_np = data_IC[:,:,:,1]
        IC_w_np = data_IC[:,:,:,2]
        IC_p_np = data_IC[:,:,:,3]
        
        IC_zeros_np = np.zeros( (nx_,ny_,nz_) )
        IC_ones_np  = np.ones ( (nx_,ny_,nz_) )

        # Timestep size
        simDt = inputConfig.simDt

        # Equation mode
        self.equationMode = inputConfig.equationMode

        # Metrics
        self.metric = metric
        
        # Target data settings
        self.useTargetData = inputConfig.useTargetData

        # ----------------------------------------------------
        # Allocate memory for state data
        self.state_u_P    = state.state_P(decomp,IC_u_np)
        self.state_v_P    = state.state_P(decomp,IC_v_np)
        self.state_w_P    = state.state_P(decomp,IC_w_np)
        self.state_p_P    = state.state_P(decomp,IC_p_np)
        self.state_DP_P   = state.state_P(decomp,IC_p_np)
        
        # Set up a Numpy mirror to the PyTorch state
        #  --> Used for file I/O
        self.state_data_all = (self.state_u_P, self.state_v_P,
                               self.state_w_P, self.state_p_P)
        self.data_all_CPU   = state.data_all_CPU(decomp,startTime,simDt,
                                            names[0:4],self.state_data_all)
        
        # Allocate a temporary velocity state for RK solvers
        if (inputConfig.advancerName[:-1]=="RK"):
            self.state_uTmp_P = state.state_P(decomp,IC_u_np)
            self.state_vTmp_P = state.state_P(decomp,IC_v_np)
            self.state_wTmp_P = state.state_P(decomp,IC_w_np)
        else:
            self.state_uTmp_P = None
            self.state_vTmp_P = None
            self.state_wTmp_P = None
            
        # Density
        self.mu  = inputConfig.mu
        self.rho = inputConfig.rho

        # Allocate pressure source term and local viscosity
        self.source_P = torch.zeros(nx_,ny_,nz_,dtype=inputConfig.dtypeTorch).to(decomp.device)
        self.VISC_P   = torch.ones(nxo_,nyo_,nzo_,dtype=inputConfig.dtypeTorch).to(decomp.device)
        self.VISC_P.mul_(self.mu)

        
        # ----------------------------------------------------
        # Allocate memory for adjoint training
        if (inputConfig.adjointTraining):
            
            # Check for a few prerequisites
            if (inputConfig.SFSmodel!="ML"):
                if (decomp.rank==0):
                    raise Exception("\nAdjoint training requires ML subfilter model\n")
            elif (inputConfig.equationMode!="NS"):
                if (decomp.rank==0):
                    raise Exception("\nAdjoint training requires Navier-Stokes solver\n")
            else:
                self.numCheckpointIt = inputConfig.numCheckpointIt
                
                # Allocate the adjoint state
                self.state_u_adj_P = state.state_P(decomp,IC_zeros_np)
                self.state_v_adj_P = state.state_P(decomp,IC_zeros_np)
                self.state_w_adj_P = state.state_P(decomp,IC_zeros_np)
                
                # Set up a Numpy mirror to the PyTorch adjoint state
                self.adjoint_data_all = (self.state_u_adj_P, self.state_v_adj_P,
                                         self.state_w_adj_P)
                self.data_adj_CPU = state.data_all_CPU(decomp,startTime,simDt,
                                                       names[0:3],self.adjoint_data_all)

                # Allocate temporary adjoint states and rhs objects for RK solvers
                if (inputConfig.advancerName[:-1]=="RK"):
                    self.state_uTmp_adj_P = state.state_P(decomp,IC_zeros_np)
                    self.state_vTmp_adj_P = state.state_P(decomp,IC_zeros_np)
                    self.state_wTmp_adj_P = state.state_P(decomp,IC_zeros_np)
                    
                # Allocate space for checkpointed solutions
                #  --> Could be moved to adjoint module
                self.check_u_P = torch.zeros(nxo_,nyo_,nzo_,self.numCheckpointIt+1,
                                             dtype=inputConfig.dtypeTorch).to(decomp.device)
                self.check_v_P = torch.zeros(nxo_,nyo_,nzo_,self.numCheckpointIt+1,
                                             dtype=inputConfig.dtypeTorch).to(decomp.device)
                self.check_w_P = torch.zeros(nxo_,nyo_,nzo_,self.numCheckpointIt+1,
                                             dtype=inputConfig.dtypeTorch).to(decomp.device)
                
                # Set up to use target data
                self.useTargetData = True
                self.numItTargetComp = self.numCheckpointIt

                
        # ----------------------------------------------------
        # Allocate memory for target state data
        if (self.useTargetData):
            self.state_u_T = state.state_P(decomp,IC_zeros_np,need_gradients=False)
            self.state_v_T = state.state_P(decomp,IC_zeros_np,need_gradients=False)
            self.state_w_T = state.state_P(decomp,IC_zeros_np,need_gradients=False)
            
            #  Set up a Numpy mirror to the target state data
            self.target_data_all = (self.state_u_T, self.state_v_T, self.state_w_T)
            self.target_data_all_CPU = state.data_all_CPU(decomp,startTime,simDt,names[0:3],
                                                          self.target_data_all)

            # Read the target data file
            #   Adjoint training reads target files in outer iteration loop
            if (not inputConfig.adjointTraining):
                self.targetDataFileStr = inputConfig.targetFileBaseStr + \
                    '{:08d}'.format(inputConfig.startFileIt)
                dr.readNGArestart_parallel(self.targetDataFileStr,self.target_data_all_CPU)

                
        # ----------------------------------------------------
        # Initialize SFS model
        if (inputConfig.SFSmodel=='Smagorinsky'):
            self.use_SFSmodel = True
            self.sfsmodel = sfsmodel_smagorinsky.stress_constCs(inputConfig,geometry)
        elif (inputConfig.SFSmodel=='gradient'):
            self.use_SFSmodel = True
            self.sfsmodel = sfsmodel_gradient.residual_stress(decomp,geometry,metric)
        elif (inputConfig.SFSmodel=='ML'):
            self.use_SFSmodel = True
            self.sfsmodel = sfsmodel_ML.residual_stress(inputConfig,decomp,geometry)
        else:
            # Construct a blank SFSmodel object
            self.use_SFSmodel = False
            self.sfsmodel = None #sfsmodel_smagorinsky.stress_constCs(geometry,metric)
            
        # Save the molecular viscosity
        if (self.sfsmodel.modelType=='eddyVisc'):
            self.muMolec = self.mu

        
        # ----------------------------------------------------
        # Initialize forward solver RHS
        self.forwardRHS = velocity.ForwardRHS(inputConfig,decomp,metric,
                                              self.rho,self.VISC_P,self.sfsmodel)
        
        # ----------------------------------------------------
        # Initialize adjoint solver RHS
        if (inputConfig.adjointTraining):
            self.adjointRHS = adjoint.AdjointRHS(inputConfig,decomp,metric,
                                                 self.rho,self.VISC_P,self.sfsmodel,
                                                 self.state_u_P,self.state_v_P,self.state_w_P)

        
        # ----------------------------------------------------
        # Initialize advancers and Poisson solver
        if (inputConfig.advancerName=="Euler"):
            self.forwardAdvancer = advancer.Euler(self.state_u_P,self.state_v_P,self.state_w_P,
                                                  self.forwardRHS)
            if (inputConfig.adjointTraining):
                self.adjointAdvancer = advancer.Euler(self.state_u_adj_P,self.state_v_adj_P,self.state_w_adj_P,
                                                      self.adjointRHS)

        elif (inputConfig.advancerName=="RK4"):
            self.forwardAdvancer = advancer.RK4(self.state_u_P,self.state_uTmp_P,
                                                self.state_v_P,self.state_vTmp_P,
                                                self.state_w_P,self.state_wTmp_P,
                                                self.forwardRHS)
            if (inputConfig.adjointTraining):
                self.adjointAdvancer = advancer.RK4(self.state_u_adj_P,self.state_uTmp_adj_P,
                                                    self.state_v_adj_P,self.state_vTmp_adj_P,
                                                    self.state_w_adj_P,self.state_wTmp_adj_P,
                                                    self.adjointRHS)
        # Poisson solver
        if (inputConfig.equationMode=='NS'):

            if (inputConfig.pSolverMode=='Jacobi'):
                self.poisson = pressure.solver_jacobi(comms,decomp,metric,
                                                      geometry,inputConfig.rho,simDt,
                                                      inputConfig.max_pressure_iterations)
            elif (inputConfig.pSolverMode=='bicgstab'):
                self.poisson = pressure.solver_bicgstab(comms,decomp,metric,geometry,inputConfig.rho,simDt,
                                                        inputConfig.min_pressure_residual,
                                                        inputConfig.max_pressure_iterations)
            elif (inputConfig.pSolverMode=='RedBlackGS'):
                #poisson = pressure.solver_GS_redblack(geometry,rho,simDt,max_pressure_iterations)
                raise Exception('\nRed-black GS not yet implemented\n')

        
        # ----------------------------------------------------
        # Clean up
        del IC_u_np
        del IC_v_np
        del IC_w_np
        del IC_p_np
        del IC_zeros_np
        del IC_ones_np


        
    # ----------------------------------------------------
    # Forward domain step
    # ----------------------------------------------------
    def forwardStep(self,simDt):

        # ----------------------------------------------------
        # Velocity predictor step
        #
        self.forwardAdvancer.step(simDt)
        
        # ----------------------------------------------------
        # Velocity corrector step
        #
        if (self.equationMode=='NS'):
        
            # 1. Currently using Chorin's original fractional step method
            #   (essentially Lie splitting); unclear interpretation of
            #   predictor step RHS w/o pressure. Modern fractional-step
            #   (based on midpoint method) would be better.
            
            # 2. Boundary conditions: zero normal gradient. Note: only
            #   satisfies local mass conservation; global mass
            #   conservation needs to be enforced in open systems before
            #   solving Poisson equation, e.g., by rescaling source_P.
            
            # Divergence of the predicted velocity field
            self.metric.div_vel(self.state_u_P,self.state_v_P,
                                self.state_w_P,self.source_P)
            
            # Integral of the Poisson eqn RHS
            #int_RP = comms.parallel_sum(torch.sum(source_P).cpu().numpy())

            # Solve the Poisson equation
            self.max_resP = self.poisson.solve(self.state_DP_P,self.state_p_P,self.source_P)

            # Compute pressure gradients
            self.metric.grad_P(self.state_DP_P)

            # Update the velocity correction
            self.state_u_P.vel_corr(self.state_DP_P.grad_x,simDt/self.rho)
            self.state_v_P.vel_corr(self.state_DP_P.grad_y,simDt/self.rho)
            self.state_w_P.vel_corr(self.state_DP_P.grad_z,simDt/self.rho)

            

    # ----------------------------------------------------
    # Adjoint domain step
    # ----------------------------------------------------
    def adjointStep(self,simDt):

        # ----------------------------------------------------
        # Adjoint 'pressure' iteration
        #
        # Divergence of the adjoint velocity field
        self.metric.div_vel(self.state_u_adj_P,self.state_v_adj_P,self.state_w_adj_P,self.source_P)

        # Solve the Poisson equation
        self.max_resP = self.poisson.solve(self.state_DP_P,self.state_p_P,self.source_P)
        
        
        # ----------------------------------------------------
        # Adjoint corrector step: \hat{u}^*
        #
        # Compute 'pressure' gradients
        self.metric.grad_P(self.state_DP_P)

        # Update the adjoint solution
        #   Note negative sign
        self.state_u_adj_P.vel_corr(self.state_DP_P.grad_x, -simDt/self.rho)
        self.state_v_adj_P.vel_corr(self.state_DP_P.grad_y, -simDt/self.rho)
        self.state_w_adj_P.vel_corr(self.state_DP_P.grad_z, -simDt/self.rho)

        
        # ----------------------------------------------------
        # Adjoint predictor step: \hat{u}^t
        #
        self.adjointAdvancer.step(simDt)