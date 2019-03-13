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
# Base class for state data
# ----------------------------------------------------
class data_P:
    def __init__(self,geo,IC=None):
        # Defaults: zero initial conditions

        # Save a pointer to the geometry object
        self.geo = geo
        prec = geo.prec

        # Data sizes
        nx_  = geo.nx_
        ny_  = geo.ny_
        nz_  = geo.nz_
        nxo_ = geo.nxo_
        nyo_ = geo.nyo_
        nzo_ = geo.nzo_
        self.nover  = geo.nover
        self.imin_  = geo.imin_;  self.imax_  = geo.imax_
        self.jmin_  = geo.jmin_;  self.jmax_  = geo.jmax_
        self.kmin_  = geo.kmin_;  self.kmax_  = geo.kmax_
        self.imino_ = geo.imino_; self.imaxo_ = geo.imaxo_
        self.jmino_ = geo.jmino_; self.jmaxo_ = geo.jmaxo_
        self.kmino_ = geo.kmino_; self.kmaxo_ = geo.kmaxo_
        
        # Allocate data arrays
        # State data
        self.var     = torch.zeros(nxo_,nyo_,nzo_,dtype=prec).to(geo.device)
        # Interpolated state data
        self.var_i   = torch.zeros(nx_+1,ny_+1,nz_+1,dtype=prec).to(geo.device)
        # First derivatives
        self.grad_x  = torch.zeros(nx_+1,ny_+1,nz_+1,dtype=prec).to(geo.device)
        self.grad_y  = torch.zeros(nx_+1,ny_+1,nz_+1,dtype=prec).to(geo.device)
        self.grad_z  = torch.zeros(nx_+1,ny_+1,nz_+1,dtype=prec).to(geo.device)

        # Copy initial conditions into the data tensor
        if (IC.any()):
            tmp_IC = torch.from_numpy(IC)
            self.var[self.imin_:self.imax_+1,self.jmin_:self.jmax_+1,
                     self.kmin_:self.kmax_+1].copy_(tmp_IC)
            del tmp_IC

        # Update the overlap cells for the initial condition
        self.update_border()
    
        
    def update_border(self):
        # Update the class data's overlap cells
        # MPI decomp goes here

        # Periodic boundaries
        self.var[self.imino_:self.imin_,:,:] = self.var[self.imax_-self.nover+1:self.imax_+1,:,:]
        self.var[:,self.jmino_:self.jmin_,:] = self.var[:,self.jmax_-self.nover+1:self.jmax_+1,:]
        self.var[:,:,self.kmino_:self.kmin_] = self.var[:,:,self.kmax_-self.nover+1:self.kmax_+1]
        
        self.var[self.imax_+1:self.imaxo_+1,:,:] = self.var[self.imin_:self.imin_+self.nover,:,:]
        self.var[:,self.jmax_+1:self.jmaxo_+1,:] = self.var[:,self.jmin_:self.jmin_+self.nover,:]
        self.var[:,:,self.kmax_+1:self.kmaxo_+1] = self.var[:,:,self.kmin_:self.kmin_+self.nover]
        
            
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
        # Subtract inGrad*inScalar
        self.var[self.imin_:self.imax_+1,
                 self.jmin_:self.jmax_+1,
                 self.kmin_:self.kmax_+1].sub_(inScalar, inGrad[:-1,:-1,:-1])
        
        # Update the overlap cells
        self.update_border()

        
