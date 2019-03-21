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
    def __init__(self,geo,time,dt,names,state_data_all):

        # Global sizes
        self.nx = geo.Nx
        self.ny = geo.Ny
        self.nz = geo.Nz

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
        imin_ = geo.imin_; imax_ = geo.imax_+1
        jmin_ = geo.jmin_; jmax_ = geo.jmax_+1
        kmin_ = geo.kmin_; kmax_ = geo.kmax_+1
        for state in state_data_all:
            self.data.append(state.var[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
                             .to(torch.device('cpu')).numpy())


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

        # Update the initial condition in the overlap cells
        self.update_border()
    
        
    def update_border(self):
        
        # Update the overlap cells
        self.decomp.communicate_border(self.var)

        # Periodic boundaries
        #self.var[self.imino_:self.imin_,:,:] = self.var[self.imax_-self.nover+1:self.imax_+1,:,:]
        #self.var[:,self.jmino_:self.jmin_,:] = self.var[:,self.jmax_-self.nover+1:self.jmax_+1,:]
        #self.var[:,:,self.kmino_:self.kmin_] = self.var[:,:,self.kmax_-self.nover+1:self.kmax_+1]
        
        #self.var[self.imax_+1:self.imaxo_+1,:,:] = self.var[self.imin_:self.imin_+self.nover,:,:]
        #self.var[:,self.jmax_+1:self.jmaxo_+1,:] = self.var[:,self.jmin_:self.jmin_+self.nover,:]
        #self.var[:,:,self.kmax_+1:self.kmaxo_+1] = self.var[:,:,self.kmin_:self.kmin_+self.nover]
        
            
    def ZAXPY(self,A,X,Y):
        # Z = A*X + Y
        # Scalar A; vector X and Y
        self.var[self.imin_:self.imax_+1,
                 self.jmin_:self.jmax_+1,
                 self.kmin_:self.kmax_+1].copy_(A*X + Y)
        
        # Update the overlap cells
        self.update_border()


    def update(self,input):
        # Copy in updated data
        self.var[self.imin_:self.imax_+1,
                 self.jmin_:self.jmax_+1,
                 self.kmin_:self.kmax_+1].copy_(input)
        
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

        
