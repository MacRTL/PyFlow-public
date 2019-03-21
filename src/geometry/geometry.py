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
    def __init__(self,xGrid,yGrid,zGrid,decomp,device,prec=torch.float32):
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

        # Local overlap cells for 2CD staggered schemes
        self.nover  = decomp.nover
        self.nxo_   = decomp.nxo_; self.nx_ = decomp.nx_
        self.nyo_   = decomp.nyo_; self.ny_ = decomp.ny_
        self.nzo_   = decomp.nzo_; self.nz_ = decomp.nz_
        self.imino_ = decomp.imino_
        self.jmino_ = decomp.jmino_
        self.kmino_ = decomp.kmino_
        self.imaxo_ = decomp.imaxo_
        self.jmaxo_ = decomp.jmaxo_
        self.kmaxo_ = decomp.kmaxo_

        # Local grid indices
        self.imin_ = decomp.imin_; self.imax_ = decomp.imax_
        self.jmin_ = decomp.jmin_; self.jmax_ = decomp.jmax_
        self.kmin_ = decomp.kmin_; self.kmax_ = decomp.kmax_
