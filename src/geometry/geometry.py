# ------------------------------------------------------------------------
#
# PyFlow: A GPU-accelerated CFD platform written in Python
#
# @file geometry.py
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
# General uniform geometry
# ----------------------------------------------------
class uniform:
    def __init__(self,xGrid,yGrid,zGrid,device='cpu',prec=torch.float32):
        # Type identifier
        self.type = 'uniform'

        # Offloading settings
        self.device = device

        # Precision - default: 32-bit floating-point precision
        self.prec = prec
        
        # Global grid sizes
        self.Nx = len(xGrid)-1
        self.Ny = len(yGrid)-1
        self.Nz = len(zGrid)-1

        self.Lx = xGrid[-1]-xGrid[0]
        self.Ly = yGrid[-1]-yGrid[0]
        self.Lz = zGrid[-1]-zGrid[0]

        self.dx = self.Lx/float(self.Nx); self.dxi = 1.0/self.dx
        self.dy = self.Ly/float(self.Ny); self.dyi = 1.0/self.dy
        self.dz = self.Lz/float(self.Nz); self.dzi = 1.0/self.dz

        # Local grid sizes - MPI decomp goes here
        self.nx_ = self.Nx
        self.ny_ = self.Ny
        self.nz_ = self.Nz

        # Local overlap cells for 2CD staggered schemes
        self.nover = 2
        self.nxo_  = self.nx_+2*self.nover
        self.nyo_  = self.ny_+2*self.nover
        self.nzo_  = self.nz_+2*self.nover
        self.imino_ = 0
        self.jmino_ = 0
        self.kmino_ = 0
        self.imaxo_ = self.imino_+self.nxo_-1
        self.jmaxo_ = self.jmino_+self.nyo_-1
        self.kmaxo_ = self.kmino_+self.nzo_-1

        # Local grid indices - MPI decomp goes here
        self.imin_ = self.imino_+self.nover
        self.jmin_ = self.jmino_+self.nover
        self.kmin_ = self.kmino_+self.nover
        self.imax_ = self.imin_+self.nx_-1
        self.jmax_ = self.jmin_+self.ny_-1
        self.kmax_ = self.kmin_+self.nz_-1
