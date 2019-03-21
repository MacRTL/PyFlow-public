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
from mpi4py import MPI

# ----------------------------------------------------
# MPI decomposition
# ----------------------------------------------------
class decomp:
    def __init__(self,Nx,Ny,Nz,nproc_decomp,isper):
        # Grid size
        self.nx = Nx
        self.ny = Ny
        self.nz = Nz

        # Overlap size
        self.nover = 2

        # Global sizes
        self.nxo  = self.nx+2*self.nover
        self.nyo  = self.ny+2*self.nover
        self.nzo  = self.nz+2*self.nover
        self.imino = 0
        self.jmino = 0
        self.kmino = 0
        self.imaxo = self.imino+self.nxo-1
        self.jmaxo = self.jmino+self.nyo-1
        self.kmaxo = self.kmino+self.nzo-1
        self.imin  = self.imino+self.nover
        self.jmin  = self.jmino+self.nover
        self.kmin  = self.kmino+self.nover
        self.imax  = self.imin+self.nx-1
        self.jmax  = self.jmin+self.ny-1
        self.kmax  = self.kmin+self.nz-1
        
        # Get MPI communicator info
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        #npy = int(np.floor(np.sqrt(self.size)))
        #npx = size//npy
        #if (npx*npy > size):
        #    npy -= 1
        #if (npx*npy > size):
        #    npx -= 1
        #dims  = [npx,npy]
        #isper = [0,0]

        self.npx = nproc_decomp[0]
        self.npy = nproc_decomp[1]
        self.npz = nproc_decomp[2]

        # Create a Cartesian communicator to determine coordinates
        comm = MPI.COMM_WORLD
        cartComm = comm.Create_cart(nproc_decomp, periods=isper, reorder=True)
        
        # Get the cpu's position in the communicator
        self.iproc, self.jproc, self.kproc = cartComm.Get_coords(self.rank)

        # x-decomposition
        imin = 0
        q = int(self.nx/self.npx)
        r = int(np.mod(self.nx,self.npx))
        if ((self.iproc+1)<=r):
            self.nx_   = q+1
            self.imin_loc = imin + self.iproc*(q+1)
        else:
            self.nx_   = q
            self.imin_loc = imin + r*(q+1) + (self.iproc-r)*q
        self.imax_loc = self.imin_loc + self.nx_ - 1
        
        # y-deomposition
        jmin = 0
        q = int(self.ny/self.npy)
        r = int(np.mod(self.ny,self.npy))
        if ((self.jproc+1)<=r):
            self.ny_   = q+1
            self.jmin_loc = jmin + self.jproc*(q+1)
        else:
            self.ny_   = q
            self.jmin_loc = jmin + r*(q+1) + (self.jproc-r)*q
        self.jmax_loc = self.jmin_loc + self.ny_ - 1
        
        # z-decomposition
        kmin = 0
        q = int(self.nz/self.npz)
        r = int(np.mod(self.nz,self.npz))
        if ((self.kproc+1)<=r):
            self.nz_   = q+1
            self.kmin_loc = kmin + self.kproc*(q+1)
        else:
            self.nz_   = q
            self.kmin_loc = kmin + r*(q+1) + (self.kproc-r)*q
        self.kmax_loc = self.kmin_loc + self.nz_ - 1

        print("rank={}\t imin_={}\timax_={}\tnx_={}\t jmin_={}\tjmax_={}\tny_={}\t kmin_={}\tkmax_={}\tnz_={}"
              .format(self.rank,
                      self.imin_loc,self.imax_loc,self.nx_,
                      self.jmin_loc,self.jmax_loc,self.ny_,
                      self.kmin_loc,self.kmax_loc,self.nz_))

        # Local grid sizes
        #self.nx_ = self.Nx
        #self.ny_ = self.Ny
        #self.nz_ = self.Nz

        # Local overlap cells for 2CD staggered schemes
        self.nxo_  = self.nx_+2*self.nover
        self.nyo_  = self.ny_+2*self.nover
        self.nzo_  = self.nz_+2*self.nover
        self.imino_ = 0
        self.jmino_ = 0
        self.kmino_ = 0
        self.imaxo_ = self.imino_+self.nxo_-1
        self.jmaxo_ = self.jmino_+self.nyo_-1
        self.kmaxo_ = self.kmino_+self.nzo_-1

        # Local grid indices
        self.imin_ = self.imino_+self.nover
        self.jmin_ = self.jmino_+self.nover
        self.kmin_ = self.kmino_+self.nover
        self.imax_ = self.imin_+self.nx_-1
        self.jmax_ = self.jmin_+self.ny_-1
        self.kmax_ = self.kmin_+self.nz_-1

        

# ----------------------------------------------------
# General uniform geometry
# ----------------------------------------------------
class uniform:
    def __init__(self,xGrid,yGrid,zGrid,decomp,device='cpu',prec=torch.float32):
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
