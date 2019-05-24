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
import scipy.sparse as sp
import torch
import inspect

# Solves Ax=b

# RP = b
# DP = x
# res = b-Ax

class Pressure_Base:
    def __init__(self,decomp,metric):
        # Save pointers to the decomp and Laplacian
        self.decomp = decomp
        self.lap    = metric.Laplace
        
    # Apply the pressure operator: B = Laplace(A)
    #  --> Communicates A in the overlap
    #  --> B is interior-only
    #  --> This method only works for uniform grids
    def applyOperator(self,A,B):
        imin_ = self.decomp.imin_; imax_ = self.decomp.imax_+1
        jmin_ = self.decomp.jmin_; jmax_ = self.decomp.jmax_+1
        kmin_ = self.decomp.kmin_; kmax_ = self.decomp.kmax_+1
        
        # Communicate A in the overlap
        self.decomp.communicate_border(A)
        
        # Apply the Laplacian operator
        B.zero_()
        # x
        B.add_( A[imin_-1:imax_-1,jmin_:jmax_,kmin_:kmax_]
                * self.lap[imin_:imax_,jmin_:jmax_,kmin_:kmax_,0,0] )
        B.add_( A[imin_  :imax_  ,jmin_:jmax_,kmin_:kmax_]
                * self.lap[imin_:imax_,jmin_:jmax_,kmin_:kmax_,0,1] )
        B.add_( A[imin_+1:imax_+1,jmin_:jmax_,kmin_:kmax_]
                * self.lap[imin_:imax_,jmin_:jmax_,kmin_:kmax_,0,2] )
        # y
        B.add_( A[imin_:imax_,jmin_-1:jmax_-1,kmin_:kmax_]
                * self.lap[imin_:imax_,jmin_:jmax_,kmin_:kmax_,1,0] )
        B.add_( A[imin_:imax_,jmin_  :jmax_  ,kmin_:kmax_]
                * self.lap[imin_:imax_,jmin_:jmax_,kmin_:kmax_,1,1] )
        B.add_( A[imin_:imax_,jmin_+1:jmax_+1,kmin_:kmax_]
                * self.lap[imin_:imax_,jmin_:jmax_,kmin_:kmax_,1,2] )
        # z
        B.add_( A[imin_:imax_,jmin_:jmax_,kmin_-1:kmax_-1]
                * self.lap[imin_:imax_,jmin_:jmax_,kmin_:kmax_,2,0] )
        B.add_( A[imin_:imax_,jmin_:jmax_,kmin_  :kmax_  ]
                * self.lap[imin_:imax_,jmin_:jmax_,kmin_:kmax_,2,1] )
        B.add_( A[imin_:imax_,jmin_:jmax_,kmin_+1:kmax_+1]
                * self.lap[imin_:imax_,jmin_:jmax_,kmin_:kmax_,2,2] )
    

def report(xk):
    frame = inspect.currentframe().f_back
    print(frame.f_locals['iter_'], frame.f_locals['resid'])
    

class solver_jacobi:
    def __init__(self,comms,decomp,metric,geo,rho,simDt,Num_pressure_iterations):

        # Save a pointer to the comms object
        self.comms = comms

        # Construct a pressure base class
        self.pressure = Pressure_Base(decomp,metric)
        
        # Save a few parameters
        self.Num_pressure_iterations = Num_pressure_iterations
        self.dx    = geo.dx
        self.imin_ = geo.imin_; self.imax_ = geo.imax_
        self.jmin_ = geo.jmin_; self.jmax_ = geo.jmax_
        self.kmin_ = geo.kmin_; self.kmax_ = geo.kmax_
        self.rho   = rho
        self.simDt = simDt

        # Allocate the residual
        self.res_P = torch.zeros(geo.nx_,geo.ny_,geo.nz_,
                                    dtype=geo.prec).to(geo.device)
        
        
    def solve(self,state_pOld_P,state_p_P,source_P):
        imin_ = self.imin_; imax_ = self.imax_+1
        jmin_ = self.jmin_; jmax_ = self.jmax_+1
        kmin_ = self.kmin_; kmax_ = self.kmax_+1

        # Zero the iC
        state_p_P.var.zero_()
            
        # Poisson equation source term
        #source_P *= self.rho/self.simDt*self.dx**2
        source_P *= self.rho/self.simDt

        # Compute the initial residual
        self.res_P.copy_( source_P )
        max_res0 = self.comms.parallel_max(torch.max(torch.abs(self.res_P)).cpu().numpy())
        
        # Matrix equation (3D)
        DInv      = -1.0/6.0
        source_P *= self.dx**2
        
        for j in range( self.Num_pressure_iterations ):
            # Initial guess is p from previous time step
            # Update the borders from the previous iteration
            state_pOld_P.update(state_p_P.var[imin_:imax_,jmin_:jmax_,kmin_:kmax_])
            
            # Jacobi iteration
            state_p_P.copy(( -state_pOld_P.var[imin_+1:imax_+1,jmin_:jmax_,kmin_:kmax_]
                             -state_pOld_P.var[imin_-1:imax_-1,jmin_:jmax_,kmin_:kmax_]
                             -state_pOld_P.var[imin_:imax_,jmin_+1:jmax_+1,kmin_:kmax_]
                             -state_pOld_P.var[imin_:imax_,jmin_-1:jmax_-1,kmin_:kmax_]
                             -state_pOld_P.var[imin_:imax_,jmin_:jmax_,kmin_+1:kmax_+1]
                             -state_pOld_P.var[imin_:imax_,jmin_:jmax_,kmin_-1:kmax_-1]
                             + source_P )*DInv)

        # Done iterating; update the border
        state_p_P.update_border()

        # Compute the final residual
        source_P /= self.dx**2
        self.pressure.applyOperator(state_p_P.var,self.res_P)
        self.res_P.mul_( -1.0 )
        self.res_P.add_( source_P )
        max_resP = self.comms.parallel_max(torch.max(torch.abs(self.res_P)).cpu().numpy())
        max_resP /= max_res0

        # Return the max residual
        return max_resP
    

class solver_bicgstab:
    def __init__(self,comms,decomp,metric,geo,rho,simDt,min_residual,max_iterations):

        # Save a pointer to the comms object
        self.comms = comms

        # Construct a pressure base class
        self.pressure = Pressure_Base(decomp,metric)
        
        # Save a few parameters
        self.min_residual   = min_residual
        self.max_iterations = max_iterations
        self.dx    = geo.dx
        self.imin_ = geo.imin_; self.imax_ = geo.imax_
        self.jmin_ = geo.jmin_; self.jmax_ = geo.jmax_
        self.kmin_ = geo.kmin_; self.kmax_ = geo.kmax_
        self.rho   = rho
        self.simDt = simDt
        self.vol   = metric.vol

        # Allocate workspace arrays
        self.pp   = torch.zeros(geo.nxo_,geo.nyo_,geo.nzo_,dtype=geo.prec).to(geo.device)
        self.s    = torch.zeros(geo.nxo_,geo.nyo_,geo.nzo_,dtype=geo.prec).to(geo.device)
        self.vv   = torch.zeros(geo.nx_,geo.ny_,geo.nz_,dtype=geo.prec).to(geo.device)
        self.t    = torch.zeros(geo.nx_,geo.ny_,geo.nz_,dtype=geo.prec).to(geo.device)
        self.res  = torch.zeros(geo.nx_,geo.ny_,geo.nz_,dtype=geo.prec).to(geo.device)
        self.res0 = torch.zeros(geo.nx_,geo.ny_,geo.nz_,dtype=geo.prec).to(geo.device)
        # Needed for preconditioned bicgstab
        #self.p_hat= torch.zeros(geo.nxo_,geo.nyo_,geo.nzo_,dtype=geo.prec).to(geo.device)
        #self.s_hat= torch.zeros(geo.nxo_,geo.nyo_,geo.nzo_,dtype=geo.prec).to(geo.device)
        
        
    def solve(self,state_DP_P,state_p_P,source_P):
        imin_ = self.imin_; imax_ = self.imax_+1
        jmin_ = self.jmin_; jmax_ = self.jmax_+1
        kmin_ = self.kmin_; kmax_ = self.kmax_+1

        # Zero the IC
        state_DP_P.var.zero_()
            
        # Poisson equation source term
        source_P *= self.rho/self.simDt

        # Rescale the rhs
        source_P *= -1.0*self.vol

        # Compute the initial residual
        self.res.copy_ ( source_P )
        self.res0.copy_( self.res )
        max_res0 = self.comms.parallel_max(torch.max(torch.abs(self.res)).cpu().numpy())

        # Set initial parameters
        # Vectors
        self.vv.zero_()
        self.pp.zero_()
        # Scalars
        rho1  = 1.0
        rho2  = 1.0
        alpha = 0.0
        omega = 1.0
        iter  = 0
        done  = False

        # Iterate
        while not done:

            # Step 1
            rho1 = self.comms.parallel_sum(torch.sum(self.res * self.res0).cpu().numpy())
            beta = alpha * rho1 / (rho2 * omega)
            # pp = res + beta * ( pp - omega * vv )
            self.pp[imin_:imax_,jmin_:jmax_,kmin_:kmax_].sub_( omega, self.vv )
            self.pp[imin_:imax_,jmin_:jmax_,kmin_:kmax_].mul_( beta )
            self.pp[imin_:imax_,jmin_:jmax_,kmin_:kmax_].add_( self.res )

            # Precondition p here: p_hat = [P]pp

            # vv = [A] p_hat
            self.pressure.applyOperator(self.pp,self.vv)

            # Step 2
            gamma = self.comms.parallel_sum(torch.sum(self.vv * self.res0).cpu().numpy())
            # if gamma=0.0, exit
            alpha = rho1/gamma
            # DP = DP + alpha * p_hat
            state_DP_P.var.add_( alpha, self.pp )
            # s = res - alpha * vv
            self.s[imin_:imax_,jmin_:jmax_,kmin_:kmax_].copy_( self.res )
            self.s[imin_:imax_,jmin_:jmax_,kmin_:kmax_].sub_( alpha, self.vv )

            # Precondition s here: s_hat = [P]s

            # t = [A] s_hat
            self.pressure.applyOperator(self.s,self.t)

            # Step 3
            buf1 = self.comms.parallel_sum(torch.sum(self.s[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
                                                     * self.t).cpu().numpy())
            buf2 = self.comms.parallel_sum(torch.sum(self.t * self.t).cpu().numpy())
            # if buf2==0, exit
            omega = buf1/buf2
            # DP = DP + omega * s_hat
            state_DP_P.var.add_( omega, self.s )

            # Update the residual
            # res = s - omega * t
            self.res.copy_( self.s[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )
            self.res.sub_( omega, self.t )
            rho2 = rho1

            # Check convergence
            max_res = self.comms.parallel_max(torch.max(torch.abs(self.res)).cpu().numpy())
            max_res /= max_res0

            # Check if done
            iter += 1
            if (iter >= self.max_iterations):
                done = True
            if (max_res <= self.min_residual):
                done = True

            #print("{} max_res={:15.7E}".format(iter,max_res))
            # done while loop

        # Update the pressure state
        state_DP_P.update_border()
        state_p_P.var.add_( state_DP_P.var )
        
        # Return the max residual
        return max_res
        

class solver_GS_redblack:
    def __init__(self,geo,rho,simDt,Num_pressure_iterations):
        
        # Save a few parameters
        self.Num_pressure_iterations = Num_pressure_iterations
        self.dx    = geo.dx
        self.imin_ = geo.imin_; self.imax_ = geo.imax_
        self.jmin_ = geo.jmin_; self.jmax_ = geo.jmax_
        self.kmin_ = geo.kmin_; self.kmax_ = geo.kmax_
        self.rho   = rho
        self.simDt = simDt

        # Data sizes
        nx_ = geo.nx_; nxo_ = geo.nxo_
        ny_ = geo.ny_; nyo_ = geo.nyo_
        nz_ = geo.nz_; nzo_ = geo.nzo_

        # Checkerboard patterns
        imin_ = self.imin_; imax_ = self.imax_+1
        jmin_ = self.jmin_; jmax_ = self.jmax_+1
        kmin_ = self.kmin_; kmax_ = self.kmax_+1

        # Red (even permuatations)
        self.cbr    = np.zeros((nxo_,nyo_,nzo_),dtype=bool)
        self.cbrxm1 = np.zeros((nxo_,nyo_,nzo_),dtype=bool)
        self.cbrxp1 = np.zeros((nxo_,nyo_,nzo_),dtype=bool)
        self.cbrym1 = np.zeros((nxo_,nyo_,nzo_),dtype=bool)
        self.cbryp1 = np.zeros((nxo_,nyo_,nzo_),dtype=bool)
        self.cbrzm1 = np.zeros((nxo_,nyo_,nzo_),dtype=bool)
        self.cbrzp1 = np.zeros((nxo_,nyo_,nzo_),dtype=bool)
        
        # Red -- cell centers
        self.cbr[imin_  :imax_  :2,jmin_  :jmax_  :2,kmin_  :kmax_  :2] = 1
        self.cbr[imin_+1:imax_+1:2,jmin_+1:jmax_+1:2,kmin_  :kmax_  :2] = 1
        self.cbr[imin_+1:imax_+1:2,jmin_  :jmax_  :2,kmin_+1:kmax_+1:2] = 1
        self.cbr[imin_  :imax_  :2,jmin_+1:jmax_+1:2,kmin_+1:kmax_+1:2] = 1
        # Red --  x-1
        xo=-1; yo=0; zo=0;
        self.cbrxm1[imin_  +xo:imax_  +xo:2,jmin_  +yo:jmax_  +yo:2,kmin_  +zo:kmax_  +zo:2] = 1
        self.cbrxm1[imin_+1+xo:imax_+1+xo:2,jmin_+1+yo:jmax_+1+yo:2,kmin_  +zo:kmax_  +zo:2] = 1
        self.cbrxm1[imin_+1+xo:imax_+1+xo:2,jmin_  +yo:jmax_  +yo:2,kmin_+1+zo:kmax_+1+zo:2] = 1
        self.cbrxm1[imin_  +xo:imax_  +xo:2,jmin_+1+yo:jmax_+1+yo:2,kmin_+1+zo:kmax_+1+zo:2] = 1
        # Red --  x+1
        xo=+1; yo=0; zo=0;
        self.cbrxp1[imin_  +xo:imax_  +xo:2,jmin_  +yo:jmax_  +yo:2,kmin_  +zo:kmax_  +zo:2] = 1
        self.cbrxp1[imin_+1+xo:imax_+1+xo:2,jmin_+1+yo:jmax_+1+yo:2,kmin_  +zo:kmax_  +zo:2] = 1
        self.cbrxp1[imin_+1+xo:imax_+1+xo:2,jmin_  +yo:jmax_  +yo:2,kmin_+1+zo:kmax_+1+zo:2] = 1
        self.cbrxp1[imin_  +xo:imax_  +xo:2,jmin_+1+yo:jmax_+1+yo:2,kmin_+1+zo:kmax_+1+zo:2] = 1
        # Red --  y-1
        xo=0; yo=-1; zo=0;
        self.cbrym1[imin_  +xo:imax_  +xo:2,jmin_  +yo:jmax_  +yo:2,kmin_  +zo:kmax_  +zo:2] = 1
        self.cbrym1[imin_+1+xo:imax_+1+xo:2,jmin_+1+yo:jmax_+1+yo:2,kmin_  +zo:kmax_  +zo:2] = 1
        self.cbrym1[imin_+1+xo:imax_+1+xo:2,jmin_  +yo:jmax_  +yo:2,kmin_+1+zo:kmax_+1+zo:2] = 1
        self.cbrym1[imin_  +xo:imax_  +xo:2,jmin_+1+yo:jmax_+1+yo:2,kmin_+1+zo:kmax_+1+zo:2] = 1
        # Red --  y+1
        xo=0; yo=+1; zo=0;
        self.cbryp1[imin_  +xo:imax_  +xo:2,jmin_  +yo:jmax_  +yo:2,kmin_  +zo:kmax_  +zo:2] = 1
        self.cbryp1[imin_+1+xo:imax_+1+xo:2,jmin_+1+yo:jmax_+1+yo:2,kmin_  +zo:kmax_  +zo:2] = 1
        self.cbryp1[imin_+1+xo:imax_+1+xo:2,jmin_  +yo:jmax_  +yo:2,kmin_+1+zo:kmax_+1+zo:2] = 1
        self.cbryp1[imin_  +xo:imax_  +xo:2,jmin_+1+yo:jmax_+1+yo:2,kmin_+1+zo:kmax_+1+zo:2] = 1
        # Red --  z-1
        xo=0; yo=0; zo=-1;
        self.cbrzm1[imin_  +xo:imax_  +xo:2,jmin_  +yo:jmax_  +yo:2,kmin_  +zo:kmax_  +zo:2] = 1
        self.cbrzm1[imin_+1+xo:imax_+1+xo:2,jmin_+1+yo:jmax_+1+yo:2,kmin_  +zo:kmax_  +zo:2] = 1
        self.cbrzm1[imin_+1+xo:imax_+1+xo:2,jmin_  +yo:jmax_  +yo:2,kmin_+1+zo:kmax_+1+zo:2] = 1
        self.cbrzm1[imin_  +xo:imax_  +xo:2,jmin_+1+yo:jmax_+1+yo:2,kmin_+1+zo:kmax_+1+zo:2] = 1
        # Red --  z+1
        xo=0; yo=0; zo=+1;
        self.cbrzp1[imin_  +xo:imax_  +xo:2,jmin_  +yo:jmax_  +yo:2,kmin_  +zo:kmax_  +zo:2] = 1
        self.cbrzp1[imin_+1+xo:imax_+1+xo:2,jmin_+1+yo:jmax_+1+yo:2,kmin_  +zo:kmax_  +zo:2] = 1
        self.cbrzp1[imin_+1+xo:imax_+1+xo:2,jmin_  +yo:jmax_  +yo:2,kmin_+1+zo:kmax_+1+zo:2] = 1
        self.cbrzp1[imin_  +xo:imax_  +xo:2,jmin_+1+yo:jmax_+1+yo:2,kmin_+1+zo:kmax_+1+zo:2] = 1

        # Need patterns for x+/-1, etc...

        # Black (odd permutations)
        self.cbb    = np.zeros((nxo_,nyo_,nzo_),dtype=bool)
        self.cbbxm1 = np.zeros((nxo_,nyo_,nzo_),dtype=bool)
        self.cbbxp1 = np.zeros((nxo_,nyo_,nzo_),dtype=bool)
        self.cbbym1 = np.zeros((nxo_,nyo_,nzo_),dtype=bool)
        self.cbbyp1 = np.zeros((nxo_,nyo_,nzo_),dtype=bool)
        self.cbbzm1 = np.zeros((nxo_,nyo_,nzo_),dtype=bool)
        self.cbbzp1 = np.zeros((nxo_,nyo_,nzo_),dtype=bool)

        # Black - cell centers
        self.cbb[imin_+1:imax_+1:2,jmin_  :jmax_  :2,kmin_  :kmax_  :2] = 1
        self.cbb[imin_  :imax_  :2,jmin_+1:jmax_+1:2,kmin_  :kmax_  :2] = 1
        self.cbb[imin_  :imax_  :2,jmin_  :jmax_  :2,kmin_+1:kmax_+1:2] = 1
        self.cbb[imin_+1:imax_+1:2,jmin_+1:jmax_+1:2,kmin_+1:kmax_+1:2] = 1
        
        #self.cbb = np.zeros((nx_,ny_,nz_),dtype=bool)
        #self.cbb[imin_+1::2,jmin_+1::2,kmin_+1::2] = 1
        #self.cbb[imin_  ::2,jmin_  ::2,kmin_+1::2] = 1
        #self.cbb[imin_  ::2,jmin_+1::2,kmin_  ::2] = 1
        #self.cbb[imin_+1::2,jmin_  ::2,kmin_  ::2] = 1

        # Black -- x-1
        xo=-1; yo=0; zo=0;
        self.cbbxm1[imin_+1+xo:imax_+1+xo:2,jmin_  +yo:jmax_  +yo:2,kmin_  +zo:kmax_  +zo:2] = 1
        self.cbbxm1[imin_  +xo:imax_  +xo:2,jmin_+1+yo:jmax_+1+yo:2,kmin_  +zo:kmax_  +zo:2] = 1
        self.cbbxm1[imin_  +xo:imax_  +xo:2,jmin_  +yo:jmax_  +yo:2,kmin_+1+zo:kmax_+1+zo:2] = 1
        self.cbbxm1[imin_+1+xo:imax_+1+xo:2,jmin_+1+yo:jmax_+1+yo:2,kmin_+1+zo:kmax_+1+zo:2] = 1
        # Black -- x+1
        xo=+1; yo=0; zo=0;
        self.cbbxp1[imin_+1+xo:imax_+1+xo:2,jmin_  +yo:jmax_  +yo:2,kmin_  +zo:kmax_  +zo:2] = 1
        self.cbbxp1[imin_  +xo:imax_  +xo:2,jmin_+1+yo:jmax_+1+yo:2,kmin_  +zo:kmax_  +zo:2] = 1
        self.cbbxp1[imin_  +xo:imax_  +xo:2,jmin_  +yo:jmax_  +yo:2,kmin_+1+zo:kmax_+1+zo:2] = 1
        self.cbbxp1[imin_+1+xo:imax_+1+xo:2,jmin_+1+yo:jmax_+1+yo:2,kmin_+1+zo:kmax_+1+zo:2] = 1
        # Black -- y-1
        xo=0; yo=-1; zo=0;
        self.cbbym1[imin_+1+xo:imax_+1+xo:2,jmin_  +yo:jmax_  +yo:2,kmin_  +zo:kmax_  +zo:2] = 1
        self.cbbym1[imin_  +xo:imax_  +xo:2,jmin_+1+yo:jmax_+1+yo:2,kmin_  +zo:kmax_  +zo:2] = 1
        self.cbbym1[imin_  +xo:imax_  +xo:2,jmin_  +yo:jmax_  +yo:2,kmin_+1+zo:kmax_+1+zo:2] = 1
        self.cbbym1[imin_+1+xo:imax_+1+xo:2,jmin_+1+yo:jmax_+1+yo:2,kmin_+1+zo:kmax_+1+zo:2] = 1
        # Black -- y+1
        xo=0; yo=+1; zo=0;
        self.cbbyp1[imin_+1+xo:imax_+1+xo:2,jmin_  +yo:jmax_  +yo:2,kmin_  +zo:kmax_  +zo:2] = 1
        self.cbbyp1[imin_  +xo:imax_  +xo:2,jmin_+1+yo:jmax_+1+yo:2,kmin_  +zo:kmax_  +zo:2] = 1
        self.cbbyp1[imin_  +xo:imax_  +xo:2,jmin_  +yo:jmax_  +yo:2,kmin_+1+zo:kmax_+1+zo:2] = 1
        self.cbbyp1[imin_+1+xo:imax_+1+xo:2,jmin_+1+yo:jmax_+1+yo:2,kmin_+1+zo:kmax_+1+zo:2] = 1
        # Black -- z-1
        xo=0; yo=0; zo=-1;
        self.cbbzm1[imin_+1+xo:imax_+1+xo:2,jmin_  +yo:jmax_  +yo:2,kmin_  +zo:kmax_  +zo:2] = 1
        self.cbbzm1[imin_  +xo:imax_  +xo:2,jmin_+1+yo:jmax_+1+yo:2,kmin_  +zo:kmax_  +zo:2] = 1
        self.cbbzm1[imin_  +xo:imax_  +xo:2,jmin_  +yo:jmax_  +yo:2,kmin_+1+zo:kmax_+1+zo:2] = 1
        self.cbbzm1[imin_+1+xo:imax_+1+xo:2,jmin_+1+yo:jmax_+1+yo:2,kmin_+1+zo:kmax_+1+zo:2] = 1
        # Black -- z+1
        xo=0; yo=0; zo=+1;
        self.cbbzp1[imin_+1+xo:imax_+1+xo:2,jmin_  +yo:jmax_  +yo:2,kmin_  +zo:kmax_  +zo:2] = 1
        self.cbbzp1[imin_  +xo:imax_  +xo:2,jmin_+1+yo:jmax_+1+yo:2,kmin_  +zo:kmax_  +zo:2] = 1
        self.cbbzp1[imin_  +xo:imax_  +xo:2,jmin_  +yo:jmax_  +yo:2,kmin_+1+zo:kmax_+1+zo:2] = 1
        self.cbbzp1[imin_+1+xo:imax_+1+xo:2,jmin_+1+yo:jmax_+1+yo:2,kmin_+1+zo:kmax_+1+zo:2] = 1

        
    def solve(self,state_pOld_P,state_p_P,source_P):
        imin_ = self.imin_; imax_ = self.imax_+1
        jmin_ = self.jmin_; jmax_ = self.jmax_+1
        kmin_ = self.kmin_; kmax_ = self.kmax_+1
            
        # Poisson equation source term
        source_P *= self.rho/self.simDt*self.dx**2
        
        # Matrix equation (3D)
        DInv  = -1.0/6.0
        
        # Initial guess is p from previous time step
        #state_pOld_P.update(state_p_P.var[imin_:imax_,jmin_:jmax_,kmin_:kmax_])
        
        # Set IC to zero
        state_p_P.var.zero_()

        print(np.shape(state_p_P.var.numpy()))
        print(np.shape(state_p_P.var.numpy()[self.cbr]))

        # Red-black Gauss-Seidel iteration
        for j in range( self.Num_pressure_iterations ):
            # Initial guess is p from previous time step
            # Update the borders from the previous iteration
            state_pOld_P.update(state_p_P.var[imin_:imax_,jmin_:jmax_,kmin_:kmax_])
        
            # Red
            state_p_P.var[self.cbr].copy_(( -state_pOld_P.var[self.cbrxm1]
                                            -state_pOld_P.var[self.cbrxp1]
                                            -state_pOld_P.var[self.cbrym1]
                                            -state_pOld_P.var[self.cbryp1]
                                            -state_pOld_P.var[self.cbrzm1]
                                            -state_pOld_P.var[self.cbrzp1]
                                            + source_P[self.cbr] )*DInv)
            
            # Copy the updated red cells to the old state
            #state_pOld_P.update(state_p_P.var[self.cbr])

            # Sync the updated borders
            state_p_P.update_border()
            
            # Black
            state_p_P.var[self.cbb].copy_(( -state_p_P.var[self.cbbxm1]
                                            -state_p_P.var[self.cbbxp1]
                                            -state_p_P.var[self.cbbym1]
                                            -state_p_P.var[self.cbbyp1]
                                            -state_p_P.var[self.cbbzm1]
                                            -state_p_P.var[self.cbbzp1]
                                            + source_P[self.cbb] )*DInv)
            
            ## Copy the updated black cells for the next iteration
            #state_pOld_P.update(state_p_P.var[imin_:imax_,jmin_:jmax_,kmin_:kmax_])
            
            # Sync the updated borders
            state_p_P.update_border()

        # Done iterating; update the border
        state_p_P.update_border()
            

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
        self.diag = diags(diagonals, offsets, shape=(N,N), dtype=geo.dtypeNumpy)

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
        source_P.mul_(self.rho/self.simDt)
        
        # Initial guess is p from previous time step
        #state_pOld_P.update(state_p_P.var[imin_:imax_,jmin_:jmax_,kmin_:kmax_])

        # Set IC to zero
        state_p_P.var.zero_()

        # Set up workspace
        self.source_P.copy_(source_P)
        self.state_P.copy_ (state_p_P.var[imin_:imax_,jmin_:jmax_,kmin_:kmax_])

        # Solve using the Scipy BiCGStab solver
        xOut,info = sp.linalg.bicgstab(self.Laplace,
                                       self.source_P.numpy().ravel(),
                                       self.state_P.numpy().ravel(),
                                       tol=self.tol, maxiter=self.Num_pressure_iterations,
                                       M=None)#,
        #callback=report)
        
        # Update the state
        if (info>=0):
            state_p_P.update(torch.from_numpy(xOut.reshape(self.nx_,self.ny_,self.nz_)))
        else:
            raise Exception("pressure_bicgstab_solve: info="+str(info))
