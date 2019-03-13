# ------------------------------------------------------------------------
#
# PyFlow: A GPU-accelerated CFD platform written in Python
#
# @file pressure.py
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
from scipy.sparse import diags
import scipy.sparse.linalg as sp
import torch
import inspect

def report(xk):
    frame = inspect.currentframe().f_back
    print(frame.f_locals['iter_'], frame.f_locals['resid'])

class solver_jacobi:
    def __init__(self,geo,rho,simDt,Num_pressure_iterations):
        
        # Save a few parameters
        self.Num_pressure_iterations = Num_pressure_iterations
        self.dx    = geo.dx
        self.imin_ = geo.imin_; self.imax_ = geo.imax_
        self.jmin_ = geo.jmin_; self.jmax_ = geo.jmax_
        self.kmin_ = geo.kmin_; self.kmax_ = geo.kmax_
        self.rho   = rho
        self.simDt = simDt
        
    def solve(self,state_pOld_P,state_p_P,source_P):
        imin_ = self.imin_; imax_ = self.imax_
        jmin_ = self.jmin_; jmax_ = self.jmax_
        kmin_ = self.kmin_; kmax_ = self.kmax_
            
        # Poisson equation source term
        source_P *= self.rho/self.simDt*self.dx**2
        
        #source_P = rho/simDt * (state_u_P.grad_x + state_v_P.grad_y + state_w_P.grad_z)
        #source_P = rho*geometry.dx*geometry.dx*(state_u_P.grad_x + state_v_P.grad_y
        #+ state_w_P.grad_z)
        
        # Matrix equation (3D)
        DInv  = -1.0/6.0
        #DInvF = -0.2
        #DInvC = -1.0/3.0
        #DInv  = -geometry.dx**2/6.0 # Centers - uniform grid
        #DInvF = -geometry.dx**2*0.2 # Faces - uniform grid
        #DInvC = -geometry.dx**2/3.0 # Corners - uniform grid
        #EFacX = -1.0/geometry.dx**2
        #EFacY = -1.0/geometry.dy**2
        #EFacZ = -1.0/geometry.dz**2
        #FFacX = -1.0/geometry.dx**2
        #FFacY = -1.0/geometry.dy**2
        #FFacZ = -1.0/geometry.dz**2
        
        for j in range( self.Num_pressure_iterations ):
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
            



class solver_bicgstab_serial:
    def __init__(self,geo,metric,rho,simDt,Num_pressure_iterations):
        
        # Save a few parameters
        self.Num_pressure_iterations = Num_pressure_iterations
        self.dx    = geo.dx
        self.nx_ = geo.nx_
        self.ny_ = geo.ny_
        self.nz_ = geo.nz_
        self.imin_ = geo.imin_; self.imax_ = geo.imax_
        self.jmin_ = geo.jmin_; self.jmax_ = geo.jmax_
        self.kmin_ = geo.kmin_; self.kmax_ = geo.kmax_
        self.rho   = rho
        self.simDt = simDt

        # Laplace operator
        self.Laplace = metric.Laplace

        # Diagonal preconditioner
        N = self.nx_*self.ny_*self.nz_
        diagonals = [ -0.5*(geo.dx**2 + geo.dy**2 + geo.dz**2) ]
        offsets   = [0]
        self.diag = diags(diagonals, offsets, shape=(N,N), dtype=np.float32)

        # Default tolerance
        self.tol = 1e-5
        
    def solve(self,state_pOld_P,state_p_P,source_P):
        imin_ = self.imin_; imax_ = self.imax_+1
        jmin_ = self.jmin_; jmax_ = self.jmax_+1
        kmin_ = self.kmin_; kmax_ = self.kmax_+1
            
        # Poisson equation source term
        source_P *= self.rho/self.simDt
        
        # Initial guess is p from previous time step
        state_pOld_P.update(state_p_P.var[imin_:imax_,jmin_:jmax_,kmin_:kmax_])

        # Solve using the Scipy BiCGStab solver
        xOut,info = sp.bicgstab(self.Laplace,
                                source_P.to(torch.device('cpu')).numpy().ravel(),
                                state_pOld_P.var[imin_:imax_,jmin_:jmax_,kmin_:kmax_].numpy().ravel(),
                                tol=self.tol, maxiter=self.Num_pressure_iterations,
                                M=self.diag )
                                #callback=report)
        
        # Update the state
        if (info>=0):
            state_p_P.update(torch.from_numpy(xOut.reshape(self.nx_,self.ny_,self.nz_)))
        else:
            print("ERROR: bicgstab info="+str(info))
