# ------------------------------------------------------------------------
#
# PyFlow: A GPU-accelerated CFD platform written in Python
#
# @file state.py
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

# ----------------------------------------------------
# CPU container for all state data
# ----------------------------------------------------
class data_all_CPU:
    def __init__(self,decomp,time,dt,names,state_data_all):
        self.device = decomp.device
        
        # Global sizes
        self.nx = decomp.nx
        self.ny = decomp.ny
        self.nz = decomp.nz

        # Local sizes
        self.nx_ = decomp.nx_
        self.ny_ = decomp.ny_
        self.nz_ = decomp.nz_

        # Position in global grid
        self.imin_loc = decomp.imin_loc; self.imax_loc = decomp.imax_loc
        self.jmin_loc = decomp.jmin_loc; self.jmax_loc = decomp.jmax_loc
        self.kmin_loc = decomp.kmin_loc; self.kmax_loc = decomp.kmax_loc

        # Local indices
        imin_ = decomp.imin_; imax_ = decomp.imax_+1
        jmin_ = decomp.jmin_; jmax_ = decomp.jmax_+1
        kmin_ = decomp.kmin_; kmax_ = decomp.kmax_+1

        # Time info
        self.time = time
        self.dt   = dt

        # Variables
        self.names = names
        self.nvar  = len(names)

        # Sanity check
        if (len(state_data_all) != self.nvar):
            print("ERROR: inconsistent state data")

        # State data
        self.data = []
        for state in state_data_all:
            self.data.append(state.var[imin_:imax_,jmin_:jmax_,kmin_:kmax_])

    def append(self,ivar,inData):
        if (ivar<self.nvar):
            self.data[ivar].copy_(torch.from_numpy(inData).to(self.device))

    def read(self,ivar):
        if (ivar<self.nvar):
            return self.data[ivar].cpu().numpy()


# ----------------------------------------------------
# Base class for state data
# ----------------------------------------------------
class state_P:
    def __init__(self,decomp,IC=None):
        # Defaults: zero initial conditions

        # Save a pointer to the decomposition object
        self.decomp = decomp
        prec = decomp.prec

        # Data sizes
        nx_  = decomp.nx_
        ny_  = decomp.ny_
        nz_  = decomp.nz_
        nxo_ = decomp.nxo_
        nyo_ = decomp.nyo_
        nzo_ = decomp.nzo_
        self.nover  = decomp.nover
        self.imin_  = decomp.imin_;  self.imax_  = decomp.imax_
        self.jmin_  = decomp.jmin_;  self.jmax_  = decomp.jmax_
        self.kmin_  = decomp.kmin_;  self.kmax_  = decomp.kmax_
        self.imino_ = decomp.imino_; self.imaxo_ = decomp.imaxo_
        self.jmino_ = decomp.jmino_; self.jmaxo_ = decomp.jmaxo_
        self.kmino_ = decomp.kmino_; self.kmaxo_ = decomp.kmaxo_
        
        # Allocate data arrays
        # State data
        self.var      = torch.zeros(nxo_,nyo_,nzo_,dtype=prec).to(decomp.device)
        # Interpolated state data
        self.var_i    = torch.zeros(nxo_,nyo_,nzo_,dtype=prec).to(decomp.device)
        # First derivatives
        self.grad_x   = torch.zeros(nxo_,nyo_,nzo_,dtype=prec).to(decomp.device)
        self.grad_y   = torch.zeros(nxo_,nyo_,nzo_,dtype=prec).to(decomp.device)
        self.grad_z   = torch.zeros(nxo_,nyo_,nzo_,dtype=prec).to(decomp.device)

        # Copy initial conditions into the data tensor
        if (IC.any()):
            tmp_IC = torch.from_numpy(IC)
            self.var[self.imin_:self.imax_+1,self.jmin_:self.jmax_+1,
                     self.kmin_:self.kmax_+1].copy_(tmp_IC)
            del tmp_IC

        # Communicate the initial condition in the overlap cells
        self.update_border()
    
        
    def update_border(self):
        # Update the overlap cells
        self.decomp.communicate_border(self.var)
        
        
    def update_border_i(self):
        # Update the overlap cells for interpolated data
        self.decomp.communicate_border(self.var_i)
        
            
    def ZAXPY(self,A,X,Y):
        # Z = A*X + Y
        # Scalar A; vector X and Y
        self.var[self.imin_:self.imax_+1,
                 self.jmin_:self.jmax_+1,
                 self.kmin_:self.kmax_+1].copy_(A*X + Y)
        
        # Update the overlap cells
        self.update_border()


    def copy(self,input):
        # Copy new state data
        self.var[self.imin_:self.imax_+1,
                 self.jmin_:self.jmax_+1,
                 self.kmin_:self.kmax_+1].copy_(input)


    def copy_red(self,input):
        # Copy new state data to 'red' cells and update the borders
        self.var[self.imin_:self.imax_:2,
                 self.jmin_:self.jmax_:2,
                 self.kmin_:self.kmax_:2].copy_(input)
        #self.update_border()
        
    def copy_black(self,input):
        # Copy new state data to 'black' cells and update the borders
        self.var[self.imin_+1:self.imax_+1:2,
                 self.jmin_+1:self.jmax_+1:2,
                 self.kmin_+1:self.kmax_+1:2].copy_(input)
        #self.update_border()


    def update(self,input):
        # Copy new state data
        self.copy(input)
        
        # Update the overlap cells
        self.update_border()


    def vel_corr(self,inGrad,inScalar):
        # Only used when state is a velocity component
        imin_ = self.imin_; imax_ = self.imax_+1
        jmin_ = self.jmin_; jmax_ = self.jmax_+1
        kmin_ = self.kmin_; kmax_ = self.kmax_+1
        
        # Subtract inGrad*inScalar
        self.var[imin_:imax_,
                 jmin_:jmax_,
                 kmin_:kmax_].sub_(inScalar,
                                   inGrad[imin_:imax_,jmin_:jmax_,kmin_:kmax_])
        
        # Update the overlap cells
        self.update_border()

        
