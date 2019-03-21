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
        
        # Matrix equation (3D)
        DInv  = -1.0/6.0
        
        for j in range( self.Num_pressure_iterations ):
            # Initial guess is p from previous time step
            state_pOld_P.update(state_p_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1])
            
            # Jacobi iteration
            # Don't need to update the border except after the last step
            state_p_P.update(( -state_pOld_P.var[imin_+1:imax_+2,jmin_:jmax_+1,kmin_:kmax_+1]
                               -state_pOld_P.var[imin_-1:imax_  ,jmin_:jmax_+1,kmin_:kmax_+1]
                               -state_pOld_P.var[imin_:imax_+1,jmin_+1:jmax_+2,kmin_:kmax_+1]
                               -state_pOld_P.var[imin_:imax_+1,jmin_-1:jmax_  ,kmin_:kmax_+1]
                               -state_pOld_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_+1:kmax_+2]
                               -state_pOld_P.var[imin_:imax_+1,jmin_:jmax_+1,kmin_-1:kmax_  ]
                               +source_P )*DInv)

            # Red/black GS/SOR?
            

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

        # Local workspace
        self.source_P = torch.zeros(geo.nx_,geo.ny_,geo.nz_,
                                    dtype=geo.prec).to(torch.device('cpu'))
        self.state_P  = torch.zeros(geo.nx_,geo.ny_,geo.nz_,
                                    dtype=geo.prec).to(torch.device('cpu'))
        
    def solve(self,state_pOld_P,state_p_P,source_P):
        imin_ = self.imin_; imax_ = self.imax_+1
        jmin_ = self.jmin_; jmax_ = self.jmax_+1
        kmin_ = self.kmin_; kmax_ = self.kmax_+1
            
        # Poisson equation source term
        source_P *= self.rho/self.simDt
        
        # Initial guess is p from previous time step
        state_pOld_P.update(state_p_P.var[imin_:imax_,jmin_:jmax_,kmin_:kmax_])

        # Set up workspace
        self.source_P.copy_(source_P)
        self.state_P.copy_ (state_p_P.var[imin_:imax_,jmin_:jmax_,kmin_:kmax_])

        # Solve using the Scipy BiCGStab solver
        xOut,info = sp.bicgstab(self.Laplace,
                                self.source_P.numpy().ravel(),
                                self.state_P.numpy().ravel(),
                                tol=self.tol, maxiter=self.Num_pressure_iterations,
                                M=self.diag )
                                #callback=report)
        
        # Update the state
        if (info>=0):
            state_p_P.update(torch.from_numpy(xOut.reshape(self.nx_,self.ny_,self.nz_)))
        else:
            raise Exception("ERROR: bicgstab info="+str(info))
