# ------------------------------------------------------------------------
#
# @file dataClasses.py
# Definition file for basic data classes, parallel operations,
#  and grid metrics
# @author Jonathan F. MacArt
#
# The MIT License (MIT)
# Copyright (c) 2019 University of Illinois Board of Trustees

# Permission is hereby granted, free of charge, to any person 
# obtaining a copy of this software and associated documentation 
# files (the "Software"), to deal in the Software without 
# restriction, including without limitation the rights to use, 
# copy, modify, merge, publish, distribute, sublicense, and/or 
# sell copies of the Software, and to permit persons to whom the 
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be 
# included in all copies or substantial portions of the Software.

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
import time
import os

from mpi4py import MPI

# =================================================================
# Class for parallel communication functions
# =================================================================
class comms:
    def __init__(self,outSize):
        # Get MPI decomposition info
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.outSize = outSize

    def parallel_sum(self,sendBuf,type='float64'):
        if (self.size>1):
            # More than one process, need to communicate
            recvBuf = np.zeros(self.outSize, dtype=type)
            self.comm.Allreduce(sendBuf,recvBuf,op=MPI.SUM)
            out = recvBuf
        else:
            # Serial computation; nothing to do
            out = sendBuf

        return out
            

# =================================================================
# Class for feature indices from NGA filtered DNS data
# =================================================================
class featureIndices:
    def __init__(self,idStr,names,combust=True):
        self.name = idStr
        self.combust = combust
        self.P = int(names.index('fP'))
        # Velocity components
        self.velLoc = "center"
        self.V = np.empty(3,dtype=np.int8)
        self.V[0] = int(names.index('fU'))
        self.V[1] = int(names.index('fV'))
        self.V[2] = int(names.index('fW'))
        # Subfilter stress components
        self.SFS = np.empty([3,3],dtype=np.int8)
        self.SFS[0,0] = int(names.index('fUU'))
        self.SFS[0,1] = int(names.index('fUV'))
        self.SFS[0,2] = int(names.index('fUW'))
        self.SFS[1,0] = int(names.index('fUV'))
        self.SFS[1,1] = int(names.index('fVV'))
        self.SFS[1,2] = int(names.index('fVW'))
        self.SFS[2,0] = int(names.index('fUW'))
        self.SFS[2,1] = int(names.index('fVW'))
        self.SFS[2,2] = int(names.index('fWW'))
        # SFS from filtered DNS data is NOT anisotropic
        self.haveAnisoSFS = False
        # Velocity gradients
        self.DVDX = np.empty([3,3],dtype=np.int8)
        self.DVDX[0,0] = int(names.index('dU_1'))
        self.DVDX[0,1] = int(names.index('dU_2'))
        self.DVDX[0,2] = int(names.index('dU_3'))
        self.DVDX[1,0] = int(names.index('dV_1'))
        self.DVDX[1,1] = int(names.index('dV_2'))
        self.DVDX[1,2] = int(names.index('dV_3'))
        self.DVDX[2,0] = int(names.index('dW_1'))
        self.DVDX[2,1] = int(names.index('dW_2'))
        self.DVDX[2,2] = int(names.index('dW_3'))
        if ("fRHO" in names):
            # Scalars
            self.RHO = int(names.index('fRHO'))
            self.H2O = int(names.index('fS_H2O'))
            # Subfilter scalar flux components
            self.SFF = np.empty(3,dtype=np.int8)
            self.SFF[0] = int(names.index('fU_H2O'))
            self.SFF[1] = int(names.index('fV_H2O'))
            self.SFF[2] = int(names.index('fW_H2O'))
            # Scalar gradients
            self.DSCDX = np.empty(3,dtype=np.int8)
            self.DSCDX[0] = int(names.index('dx_H2O'))
            self.DSCDX[1] = int(names.index('dy_H2O'))
            self.DSCDX[2] = int(names.index('dz_H2O'))
        
            

# =================================================================
# Class for feature indices from NGA LES volume data
# =================================================================
class featureIndices_LES:
    def __init__(self,idStr,names,combust=True):
        self.name = idStr
        self.combust = combust
        self.P = int(names.index('P'))
        self.V = np.empty(3,dtype=np.int8)
        if ("U" in names):
            self.velLoc = "face"
            self.V[0] = int(names.index('U'))
            self.V[1] = int(names.index('V'))
            self.V[2] = int(names.index('W'))
        if ("Ui" in names):
            self.velLoc = "center"
            self.V[0] = int(names.index('Ui'))
            self.V[1] = int(names.index('Vi'))
            self.V[2] = int(names.index('Wi'))
        # Do we have velocity gradients?
        if ("dUdx" in names):
            self.haveVelGrad = True
            self.DVDX = np.empty([3,3],dtype=np.int8)
            self.DVDX[0,0] = int(names.index('dUdx'))
            self.DVDX[0,1] = int(names.index('dUdy'))
            self.DVDX[0,2] = int(names.index('dUdz'))
            self.DVDX[1,0] = int(names.index('dVdx'))
            self.DVDX[1,1] = int(names.index('dVdy'))
            self.DVDX[1,2] = int(names.index('dVdz'))
            self.DVDX[2,0] = int(names.index('dWdx'))
            self.DVDX[2,1] = int(names.index('dWdy'))
            self.DVDX[2,2] = int(names.index('dWdz'))
        else:
            self.haveVelGrad = False
        # Subfilter stress components
        self.SFS = np.empty([3,3],dtype=np.int8)
        self.SFS[0,0] = int(names.index('SFSxx'))
        self.SFS[0,1] = int(names.index('SFSxy'))
        self.SFS[0,2] = int(names.index('SFSxz'))
        self.SFS[1,1] = int(names.index('SFSyy'))
        self.SFS[1,2] = int(names.index('SFSyz'))
        self.SFS[2,2] = int(names.index('SFSzz'))
        self.SFS[1,0] = self.SFS[0,1]
        self.SFS[2,0] = self.SFS[0,2]
        self.SFS[2,1] = self.SFS[1,2]
        # SFS from LES models IS anisotropic
        self.haveAnisoSFS = True
        if ('RHO' in names):
            # Scalars
            self.RHO = int(names.index('RHO'))
            self.H2O = int(names.index('PROG'))
            # Subfilter scalar flux components
            self.SFF = np.empty(3,dtype=np.int8)
            self.SFF[0] = int(names.index('Fx-PROG'))
            self.SFF[1] = int(names.index('Fy-PROG'))
            self.SFF[2] = int(names.index('Fz-PROG'))



# =================================================================
# Class for data containers and metrics
# =================================================================
class filteredData:
    def __init__(self,names,xGrid,yGrid,zGrid,DeltaN,size,rank,
                 write_file_number,nxCube=None,symmetry='sphere',
                 Lx=None,inTime=0.0,clip=True,nvar_alloc=None):
        self.nSnapshots = 0
        self.nvar = len(names)
        self.names = names

        # Do we have DNS or LES?
        if (('Ui' in names) or ('U' in names)):
            self.type = "LES"
        else:
            self.type = "DNS"
            
        # Constructor initializes grid, decomposition, and filter info
        self.DeltaN = DeltaN
        self.nx = len(xGrid)
        self.ny = len(yGrid)
        self.nz = len(zGrid)
        self.nFilt = DeltaN+1
        self.xGridIn = xGrid
        self.yGridIn = yGrid
        self.zGridIn = zGrid
        self.rank = rank
        self.init = False
        self.symmetry = symmetry
        self.Lx = Lx
        self.time = inTime

        # Do we clip the edges of the domain?
        self.clip_X = clip
        self.clip_Y = clip
        self.clip_Z = clip

        # Switch for different decomposition types
        #  Decomposition maps also initialize filter metrics
        if (nxCube==None):
            # Set up uniform domain decomposition
            #  - Slices the domain in x
            #  - Reads the whole domain
            self.decomp_uniform(size,rank)
        else:
            # Set up random sub-cube domain decomposition
            #  - Reads nxCube^3 sub-cubes on each rank
            #  - Does NOT read the whole domain
            #  - Can be re-initialized by calling self.decomp_random(size,rank)
            self.nxCube = nxCube
            pid = os.getpid()
            sec = int(time.time())
            self.random = np.random
            self.random.seed(sec+100*(rank+1)+100*write_file_number)
            # Set up the decomposition
            self.decomp_random()
        
        # Initialize empty data array
        if (nvar_alloc==None):
            nvar_alloc = self.nvar
        self.data = np.empty((self.nxFilt,self.nyFilt,self.nzFilt,nvar_alloc),
                             dtype='float64', order='F')
        self.ivar = 0


    def decomp_uniform(self,size,rank):
        # Whole-domain MPI decomposition
        #  For now, decompose only in x
        #  imin_, imax_ use C indexing convention
        #npx   = size
        #iproc = rank

        # Pencil decomposition (x and y)
        #npx = size//2 # integer division
        #npy = size-npx
        npy = int(np.floor(np.sqrt(size)))
        npx = size//npy
        if (npx*npy > size):
            npy -= 1
        if (npx*npy > size):
            npx -= 1
        dims  = [npx,npy]
        isper = [0,0]

        # Create a Cartesian communicator to determine coordinates
        comm = MPI.COMM_WORLD
        cartComm = comm.Create_cart(dims, periods=isper, reorder=True)
        
        # Get the cpu's position in the communicator
        iproc, jproc = cartComm.Get_coords(rank)

        # Save the decomposition for future use
        self.npx = npx
        self.nyp = npy
        self.iproc = iproc
        self.jproc = jproc
        # x-decomposition
        imin  = 0
        q = int(self.nx/npx)
        r = int(np.mod(self.nx,npx))
        if ((iproc+1)<=r):
            self.nx_   = q+1
            self.imin_ = imin + iproc*(q+1)
        else:
            self.nx_   = q
            self.imin_ = imin + r*(q+1) + (iproc-r)*q
        self.imax_ = self.imin_ + self.nx_
        # y-deomposition
        jmin  = 0
        q = int(self.ny/npy)
        r = int(np.mod(self.ny,npy))
        if ((jproc+1)<=r):
            self.ny_   = q+1
            self.jmin_ = jmin + jproc*(q+1)
        else:
            self.ny_   = q
            self.jmin_ = jmin + r*(q+1) + (jproc-r)*q
        self.jmax_ = self.jmin_ + self.ny_
        # z -- no decomposition
        self.kmin_ = 0; self.kmax_ = self.nz; self.nz_ = self.nz

        print("rank={}\t imin_={}\t imax_={}\t nx_={}  \t jmin_={}\t jmax_={}\t ny_={}"
              .format(rank,self.imin_,self.imax_,self.nx_,self.jmin_,self.jmax_,self.ny_))
        
        # Optional: Clip domain boundaries of the filtered data
        #   For non-periodic domains, the volumeStats a-priori filter does not
        #     maintain the nominal filter width at domain boundaries
        #   Only clip the physical boundaries of decomposed domains
        #   imin_filt, imax_filt use Python indexing convention
        if ((npx>1 and iproc>0 and iproc<(npx-1)) or self.clip_X==False):
            self.nxFilt = self.nx_
            self.imin_filt = self.imin_
            self.imax_filt = self.imax_
        elif (npx>1 and iproc==0):
            self.nxFilt = self.nx_-self.nFilt
            self.imin_filt = self.nFilt
            self.imax_filt = self.imax_
        elif (npx>1 and iproc==(npx-1)):
            self.nxFilt = self.nx_-self.nFilt
            self.imin_filt = self.imin_
            self.imax_filt = self.nx-self.nFilt
        else:
            self.nxFilt = self.nx_-2*self.nFilt
            self.imin_filt = self.nFilt
            self.imax_filt = self.nx-self.nFilt

        if ((npy>1 and jproc>0 and jproc<(npy-1)) or self.clip_Y==False):
            self.nyFilt = self.ny_
            self.jmin_filt = self.jmin_
            self.jmax_filt = self.jmax_
        elif (npy>1 and jproc==0):
            self.nyFilt = self.ny_-self.nFilt
            self.jmin_filt = self.nFilt
            self.jmax_filt = self.jmax_
        elif (npy>1 and jproc==(npy-1)):
            self.nyFilt = self.ny_-self.nFilt
            self.jmin_filt = self.jmin_
            self.jmax_filt = self.ny-self.nFilt
        else:
            self.nyFilt = self.ny_-2*self.nFilt
            self.jmin_filt = self.nFilt
            self.jmax_filt = self.ny-self.nFilt
            
        if (self.clip_Z==False):
            self.nzFilt = self.nz
            self.kmin_filt = self.kmin_
            self.kmax_filt = self.kmax_
        else:
            self.nzFilt = self.nz-2*self.nFilt
            self.kmin_filt = self.nFilt
            self.kmax_filt = self.nz-self.nFilt

        # Total number of points in the decomposed, (optionally clipped) dataset
        self.npFilt = self.nxFilt*self.nyFilt*self.nzFilt

        #print("rank={}\t imin_filt={}\t imax_filt={}\t nxFilt={}  \t jmin_filt={}\t jmax_filt={}\t nyFilt={}"
        #      .format(rank,self.imin_filt,self.imax_filt,self.nxFilt,self.jmin_filt,self.jmax_filt,self.nyFilt))
        
        # Set up the filter using the new decomposition
        self.filter_init()


    def decomp_random(self):
        # Set the dimensions
        self.nx_ = self.nxCube; self.ny_ = self.nxCube; self.nz_ = self.nxCube
        self.nxFilt = self.nxCube; self.nyFilt = self.nxCube; self.nzFilt = self.nxCube
        self.npFilt = self.nxFilt*self.nyFilt*self.nzFilt
        # Random sub-cube decomposition
        self.imin_ = self.random.randint(self.nFilt,self.nx-self.nFilt-self.nxFilt-1)
        self.jmin_ = self.random.randint(self.nFilt,self.ny-self.nFilt-self.nyFilt-1)
        self.kmin_ = self.random.randint(self.nFilt,self.nz-self.nFilt-self.nzFilt-1)
        # Set the decomposition
        self.decomp_set_cube(self.imin_,self.jmin_,self.kmin_)


    def decomp_set_cube(self,imin_,jmin_,kmin_):
        # Re-initialize the variable count
        self.ivar = 0
        # Set the coordinates
        self.imin_ = imin_
        self.jmin_ = jmin_
        self.kmin_ = kmin_
        self.imax_ = self.imin_ + self.nx_
        self.jmax_ = self.jmin_ + self.ny_
        self.kmax_ = self.kmin_ + self.nz_
        self.imin_filt = self.imin_; self.imax_filt = self.imax_
        self.jmin_filt = self.jmin_; self.jmax_filt = self.jmax_
        self.kmin_filt = self.kmin_; self.kmax_filt = self.kmax_
        
        # Set the filter using the new decomposition
        self.filter_init()
        

    def filter_init(self):
        #
        # Initialize the cell-centered grid
        #
        # Filtered DNS (volumeStats) already has xm,ym,zm
        # LES (NGA dump_volume) has x,y,z and needs to be interpolated
        #
        if (self.type=="LES"):
            # Interpolate GridIn to cell centers
            self.xGridIn[:self.nx-1] = 0.5*(self.xGridIn[1:]+self.xGridIn[:self.nx-1])
            self.yGridIn[:self.ny-1] = 0.5*(self.yGridIn[1:]+self.yGridIn[:self.ny-1])
            self.zGridIn[:self.nz-1] = 0.5*(self.zGridIn[1:]+self.zGridIn[:self.nz-1])
            self.xGridIn[self.nx-1]  = 2.0*self.xGridIn[self.nx-2]-self.xGridIn[self.nx-3]
            self.yGridIn[self.ny-1]  = 2.0*self.yGridIn[self.ny-2]-self.yGridIn[self.ny-3]
            self.zGridIn[self.nz-1]  = 2.0*self.zGridIn[self.nz-2]-self.zGridIn[self.nz-3]

        # Local grid spacing
        self.dx = np.empty(self.nxFilt)
        self.dy = np.empty(self.nyFilt)
        self.dz = np.empty(self.nzFilt)
        self.dx[:self.nxFilt-1] = (self.xGridIn[self.imin_filt+1:self.imax_filt] -
                                   self.xGridIn[self.imin_filt:self.imax_filt-1] )
        self.dx[self.nxFilt-1] = self.dx[self.nxFilt-2]
        
        self.dy[:self.nyFilt-1] = (self.yGridIn[self.jmin_filt+1:self.jmax_filt] -
                                   self.yGridIn[self.jmin_filt:self.jmax_filt-1] )
        self.dy[self.nyFilt-1] = self.dy[self.nyFilt-2]
        
        self.dz[:self.nzFilt-1] = (self.zGridIn[self.kmin_filt+1:self.kmax_filt] -
                                   self.zGridIn[self.kmin_filt:self.kmax_filt-1] )
        self.dz[self.nzFilt-1] = self.dz[self.nzFilt-2]

        # Grid for the data we actually process
        self.xGridF = self.xGridIn[self.imin_filt:self.imax_filt]
        self.yGridF = self.yGridIn[self.jmin_filt:self.jmax_filt]
        self.zGridF = self.zGridIn[self.kmin_filt:self.kmax_filt]

        #
        # Compute directional and characteristic filter widths
        #
        self.Delta  = (np.outer(self.DeltaN*self.dx,
                                np.outer(self.DeltaN*self.dy,
                                         self.DeltaN*self.dz))
                       .reshape(self.nxFilt,self.nyFilt,self.nzFilt))**(1.0/3.0)
        
        if (self.init==False):
            self.DeltaX = np.empty([self.nxFilt,self.nyFilt,self.nzFilt,3])
            self.init = True
        self.DeltaX[:,:,:,0] = (np.outer(self.DeltaN*self.dx,
                                         np.outer(np.ones(self.nyFilt),
                                                  np.ones(self.nzFilt)))
                                .reshape(self.nxFilt,self.nyFilt,self.nzFilt))
        self.DeltaX[:,:,:,1] = (np.outer(np.ones(self.nxFilt),
                                         np.outer(self.DeltaN*self.dy,
                                                  np.ones(self.nzFilt)))
                                .reshape(self.nxFilt,self.nyFilt,self.nzFilt))
        self.DeltaX[:,:,:,2] = (np.outer(np.ones(self.nxFilt),
                                         np.outer(np.ones(self.nyFilt),
                                                  self.DeltaN*self.dz))
                                .reshape(self.nxFilt,self.nyFilt,self.nzFilt))

        #
        # Data arrays for spherical coordinate transformation
        #
        if (self.symmetry=='sphere'):
            # Cartesian coordinates
            self.radius_min = 0.0
            if (self.Lx==None):
                self.radius_max = 0.5*self.xGridIn[self.nx-1]
            else:
                self.radius_max = 0.5*self.Lx
            
            ri  = self.xGridF-self.radius_max
            rj  = self.yGridF-self.radius_max
            rk  = self.zGridF-self.radius_max
            
            ri_mat = (np.outer(ri, np.outer(np.ones(self.nyFilt), np.ones(self.nzFilt)))
                      .reshape(self.nxFilt,self.nyFilt,self.nzFilt))
            rj_mat = (np.outer(np.ones(self.nxFilt), np.outer(rj, np.ones(self.nzFilt)))
                      .reshape(self.nxFilt,self.nyFilt,self.nzFilt))
            rk_mat = (np.outer(np.ones(self.nxFilt), np.outer(np.ones(self.nyFilt), rk))
                      .reshape(self.nxFilt,self.nyFilt,self.nzFilt))
            
            # Spherical coordinates
            self.radius = np.sqrt(ri_mat**2 + rj_mat**2 + rk_mat**2)
            phi    = np.arctan2(rj_mat, ri_mat)
            theta  = np.arccos(rk_mat/self.radius)
            
            # Rotation matrix
            self.transform = np.empty((self.nxFilt,self.nyFilt,self.nzFilt,3,3), dtype='float64')
            self.transform[:,:,:,0,0] = np.sin(theta)*np.cos(phi)
            self.transform[:,:,:,0,1] = np.sin(theta)*np.sin(phi)
            self.transform[:,:,:,0,2] = np.cos(theta)
            self.transform[:,:,:,1,0] = np.cos(theta)*np.cos(phi)
            self.transform[:,:,:,1,1] = np.cos(theta)*np.sin(phi)
            self.transform[:,:,:,1,2] =-np.sin(theta)
            self.transform[:,:,:,2,0] =-np.sin(phi)
            self.transform[:,:,:,2,1] = np.cos(phi)
            self.transform[:,:,:,2,2] = 0.0

        else:
            if (self.rank==0):
                print("Transform "+self.symmetry+" not implemented!")
        
    def add(self,rawData):
        # Clip data, reshape, and append
        if (len(rawData.shape)==4):
            # Append all variables (from dr.readNGA serial)
            for ivarIn in range(rawData.shape[3]):
                self.data[:,:,:,self.ivar] = rawData[self.imin_filt-self.imin_:self.imax_filt-self.imin_,
                                                     self.jmin_filt-self.jmin_:self.jmax_filt-self.jmin_,
                                                     self.kmin_filt-self.kmin_:self.kmax_filt-self.kmin_,
                                                     ivarIn]
                self.ivar += 1
        else:
            # Append one variable at a time (from dr.readNGA_parallel)
            np.copyto(self.data[:,:,:,self.ivar],
                      rawData[self.imin_filt-self.imin_:self.imax_filt-self.imin_,
                              self.jmin_filt-self.jmin_:self.jmax_filt-self.jmin_,
                              self.kmin_filt-self.kmin_:self.kmax_filt-self.kmin_])
            
            self.ivar += 1


    # Compute the divergence of a 3x3 tensor using filteredData class metrics
    def divergence(self,tensor_in):
        nx = self.nxFilt; ny = self.nyFilt; nz = self.nzFilt
        divg_out = np.zeros((self.nxFilt,self.nyFilt,self.nzFilt,3), dtype='float64')

        for ii in range(3):
            # d(tau_i1)/dx1
            divg_out[0   ,:,:,ii] += ( ( tensor_in[1 ,:,:,ii,0] - tensor_in[0  ,:,:,ii,0] )
                                       / self.dx[0   ,np.newaxis,np.newaxis] )
            divg_out[1:-1,:,:,ii] += ( ( tensor_in[2:,:,:,ii,0] - tensor_in[:-2,:,:,ii,0] )*0.5
                                       / self.dx[1:-1,np.newaxis,np.newaxis] )
            divg_out[nx-1,:,:,ii] += ( ( tensor_in[nx-1,:,:,ii,0] - tensor_in[nx-2,:,:,ii,0] )
                                       / self.dx[nx-1,np.newaxis,np.newaxis] )
            # d(tau_i2)/dx2
            divg_out[:,0   ,:,ii] += ( ( tensor_in[:,1 ,:,ii,1] - tensor_in[:,0  ,:,ii,1] )
                                       / self.dy[np.newaxis,0   ,np.newaxis] )
            divg_out[:,1:-1,:,ii] += ( ( tensor_in[:,2:,:,ii,1] - tensor_in[:,:-2,:,ii,1] )*0.5
                                       / self.dy[np.newaxis,1:-1,np.newaxis] )
            divg_out[:,ny-1,:,ii] += ( ( tensor_in[:,ny-1,:,ii,1] - tensor_in[:,ny-2,:,ii,1] )
                                       / self.dy[np.newaxis,ny-1,np.newaxis] )
            # d(tau_i3)/dx3
            divg_out[:,:,0   ,ii] += ( ( tensor_in[:,:,1 ,ii,2] - tensor_in[:,:,0  ,ii,2] )
                                       / self.dz[np.newaxis,np.newaxis,0   ] )
            divg_out[:,:,1:-1,ii] += ( ( tensor_in[:,:,2:,ii,2] - tensor_in[:,:,:-2,ii,2] )*0.5
                                       / self.dz[np.newaxis,np.newaxis,1:-1] )
            divg_out[:,:,nz-1,ii] += ( ( tensor_in[:,:,nz-1,ii,2] - tensor_in[:,:,nz-2,ii,2] )
                                       / self.dz[np.newaxis,np.newaxis,nz-1] )
        return divg_out

    
    # Compute the velocity gradient tensor
    def velGrad(self,ind):
        nx = self.nxFilt; ny = self.nyFilt; nz = self.nzFilt
        grad_out = np.zeros((self.nxFilt,self.nyFilt,self.nzFilt,3,3), dtype='float64')
        
        # du_i/dx
        for ii in range(3):
            grad_out[0   ,:,:,ii,0] = ( ( self.data[1 ,:,:,ind.V[ii]] - self.data[0  ,:,:,ind.V[ii]] )
                                        / self.dx[0   ,np.newaxis,np.newaxis] )
            grad_out[1:-1,:,:,ii,0] = ( ( self.data[2:,:,:,ind.V[ii]] - self.data[:-2,:,:,ind.V[ii]] )*0.5
                                        / self.dx[1:-1,np.newaxis,np.newaxis] )
            grad_out[nx-1,:,:,ii,0] = ( ( self.data[nx-1,:,:,ind.V[ii]] - self.data[nx-2,:,:,ind.V[ii]] )
                                        / self.dx[nx-1,np.newaxis,np.newaxis] )
        # du_i/dy
        for ii in range(3):
            grad_out[:,0   ,:,ii,1] = ( ( self.data[:,1 ,:,ind.V[ii]] - self.data[:,0  ,:,ind.V[ii]] )
                                        / self.dy[np.newaxis,0   ,np.newaxis] )
            grad_out[:,1:-1,:,ii,1] = ( ( self.data[:,2:,:,ind.V[ii]] - self.data[:,:-2,:,ind.V[ii]] )*0.5
                                        / self.dy[np.newaxis,1:-1,np.newaxis] )
            grad_out[:,ny-1,:,ii,1] = ( ( self.data[:,ny-1,:,ind.V[ii]] - self.data[:,ny-2,:,ind.V[ii]] )
                                        / self.dy[np.newaxis,ny-1,np.newaxis] )
        # du_i/dz
        for ii in range(3):
            grad_out[:,:,0   ,ii,2] = ( ( self.data[:,:,1 ,ind.V[ii]] - self.data[:,:,0  ,ind.V[ii]] )
                                        / self.dz[np.newaxis,np.newaxis,0   ] )
            grad_out[:,:,1:-1,ii,2] = ( ( self.data[:,:,2:,ind.V[ii]] - self.data[:,:,:-2,ind.V[ii]] )*0.5
                                        / self.dz[np.newaxis,np.newaxis,1:-1] )
            grad_out[:,:,nz-1,ii,2] = ( ( self.data[:,:,nz-1,ind.V[ii]] - self.data[:,:,nz-2,ind.V[ii]] )
                                        / self.dz[np.newaxis,np.newaxis,nz-1] )
            
        return grad_out

    
    # Compute the velocity gradient tensor using the LES grid resolution
    def velGradLESGrid(self,ind,nsIn=None):
        if (nsIn==None):
            ns = int(self.DeltaN/2)
        else:
            ns = int(nsIn/2)
        nx = self.nxFilt; ny = self.nyFilt; nz = self.nzFilt
        grad_out = np.zeros((self.nxFilt,self.nyFilt,self.nzFilt,3,3), dtype='float64')

        # du_i/dx
        for ii in range(3):
            # Forward difference
            for i in range(0,ns):
                grad_out[i,:,:,ii,0] = ( ( self.data[i+ns,:,:,ind.V[ii]] - self.data[i,:,:,ind.V[ii]] )
                                         / (self.xGridF[i+ns,np.newaxis,np.newaxis] - self.xGridF[i,np.newaxis,np.newaxis]) )
            # Central difference
            grad_out[ns:-ns,:,:,ii,0] = ( ( self.data[2*ns:,:,:,ind.V[ii]] - self.data[:-2*ns,:,:,ind.V[ii]] )
                                        / (self.xGridF[2*ns:,np.newaxis,np.newaxis] - self.xGridF[:-2*ns,np.newaxis,np.newaxis]) )
            # Backward difference
            for i in range(nx-ns,nx):
                grad_out[i,:,:,ii,0] = ( ( self.data[i,:,:,ind.V[ii]] - self.data[i-ns,:,:,ind.V[ii]] )
                                         / (self.xGridF[i,np.newaxis,np.newaxis] - self.xGridF[i-ns,np.newaxis,np.newaxis]) )
        # du_i/dy
        for ii in range(3):
            for j in range(0,ns):
                grad_out[:,j,:,ii,1] = ( ( self.data[:,j+ns,:,ind.V[ii]] - self.data[:,j,:,ind.V[ii]] )
                                         / (self.yGridF[np.newaxis,j+ns,np.newaxis] - self.yGridF[np.newaxis,j,np.newaxis]) )
            grad_out[:,ns:-ns,:,ii,1] = ( ( self.data[:,2*ns:,:,ind.V[ii]] - self.data[:,:-2*ns,:,ind.V[ii]] )
                                        / (self.yGridF[np.newaxis,2*ns:,np.newaxis] - self.yGridF[np.newaxis,:-2*ns,np.newaxis]) )
            for j in range(ny-ns,ny):
                grad_out[:,j,:,ii,1] = ( ( self.data[:,j,:,ind.V[ii]] - self.data[:,j-ns,:,ind.V[ii]] )
                                         / (self.yGridF[np.newaxis,j,np.newaxis] - self.yGridF[np.newaxis,j-ns,np.newaxis]) )
        # du_i/dz
        for ii in range(3):
            for k in range(0,ns):
                grad_out[:,:,k,ii,2] = ( ( self.data[:,:,k+ns,ind.V[ii]] - self.data[:,:,k,ind.V[ii]] )
                                         / (self.zGridF[np.newaxis,np.newaxis,k+ns] - self.zGridF[np.newaxis,np.newaxis,k]) )
            grad_out[:,:,ns:-ns,ii,2] = ( ( self.data[:,:,2*ns:,ind.V[ii]] - self.data[:,:,:-2*ns,ind.V[ii]] )
                                        / (self.zGridF[np.newaxis,np.newaxis,2*ns:] - self.zGridF[np.newaxis,np.newaxis,:-2*ns]) )
            for k in range(nz-ns,nz):
                grad_out[:,:,k,ii,2] = ( ( self.data[:,:,k,ind.V[ii]] - self.data[:,:,k-ns,ind.V[ii]] )
                                         / (self.zGridF[np.newaxis,np.newaxis,k] - self.zGridF[np.newaxis,np.newaxis,k-ns]) )
        return grad_out


    # Compute a scalar gradient
    def scalarGrad(self,index):
        nx = self.nxFilt; ny = self.nyFilt; nz = self.nzFilt
        grad_out = np.zeros((self.nxFilt,self.nyFilt,self.nzFilt,3), dtype='float64')

        # du_i/dx
        grad_out[0   ,:,:,0] = ( ( self.data[1 ,:,:,index] - self.data[0  ,:,:,index] )
                                 / self.dx[0   ,np.newaxis,np.newaxis] )
        grad_out[1:-1,:,:,0] = ( ( self.data[2:,:,:,index] - self.data[:-2,:,:,index] )*0.5
                                 / self.dx[1:-1,np.newaxis,np.newaxis] )
        grad_out[nx-1,:,:,0] = ( ( self.data[nx-1,:,:,index] - self.data[nx-2,:,:,index] )
                                 / self.dx[nx-1,np.newaxis,np.newaxis] )
        # du_i/dy
        grad_out[:,0   ,:,1] = ( ( self.data[:,1 ,:,index] - self.data[:,0  ,:,index] )
                                 / self.dy[np.newaxis,0   ,np.newaxis] )
        grad_out[:,1:-1,:,1] = ( ( self.data[:,2:,:,index] - self.data[:,:-2,:,index] )*0.5
                                 / self.dy[np.newaxis,1:-1,np.newaxis] )
        grad_out[:,ny-1,:,1] = ( ( self.data[:,ny-1,:,index] - self.data[:,ny-2,:,index] )
                                 / self.dy[np.newaxis,ny-1,np.newaxis] )
        # du_i/dz
        grad_out[:,:,0   ,2] = ( ( self.data[:,:,1 ,index] - self.data[:,:,0  ,index] )
                                 / self.dz[np.newaxis,np.newaxis,0   ] )
        grad_out[:,:,1:-1,2] = ( ( self.data[:,:,2:,index] - self.data[:,:,:-2,index] )*0.5
                                 / self.dz[np.newaxis,np.newaxis,1:-1] )
        grad_out[:,:,nz-1,2] = ( ( self.data[:,:,nz-1,index] - self.data[:,:,nz-2,index] )
                                 / self.dz[np.newaxis,np.newaxis,nz-1] )
        return grad_out
                    
          
# =================================================================
# Class to compute the mean along symmetry coordinates
# =================================================================
class symmetric_mean:
    def __init__(self,nbins,data_in,Test,ind,direction='radial'):
        self.nbins   = nbins
        self.data = data_in
        self.dims    = len(np.shape(data_in))
        self.Test = Test
        
        # Allocate count and sum arrays
        self.count = np.zeros(self.nbins, dtype='int')
        if (self.dims==3):
            # Return scalar-valued bins
            self.sum  = np.zeros(self.nbins, dtype='float64')
        elif (self.dims==4):
            # Return vector-valued bins
            self.sum  = np.zeros((self.nbins,3), dtype='float64')
        elif (self.dims==5):
            # Return tensor-valued bins
            self.sum  = np.zeros((self.nbins,3,3), dtype='float64')

        # Switch for independent coordinate of mean
        if (direction=='radial'):
            self.C_min   = Test.radius_min
            self.C_max   = Test.radius_max
            self.C_local = Test.radius
        elif (direction=='conditional'):
            H2O_min = 0.0
            H2O_max = 0.184539
            self.C_min = 0.0
            self.C_max = 1.0
            self.C_local = Test.data[:,:,:,ind.H2O]/(H2O_max-H2O_min)
            
        self.bin_limits = np.linspace(self.C_min, self.C_max, nbins+1)

        # Get the transformation matrix from the data object
        self.transform = Test.transform

        # Set up a parallel communicator
        self.comms = comms(nbins)
        
        # Change data_in in main routine, then call self.update()
        
    def update(self,useBias=False,bias=None,power=None,nskip=None):
        # Set the bias - only works for scalar data
        if (useBias):
            if (len(bias)==self.nbins):
                myBias = bias
            else:
                myBias = np.zeros(self.nbins)
        else:
            myBias = np.zeros(self.nbins)
        # Set the power - only works for scalar data
        if (power==None):
            myPower = 1
        else:
            myPower = power
            
        # Set the number of grid points to skip
        #   Useful for reducing DNS grids to LES grids
        if (nskip==None):
            iskip = 1; jskip = 1; kskip = 1
            imin = 0; imax = self.Test.nxFilt
            jmin = 0; jmax = self.Test.nyFilt
            kmin = 0; kmax = self.Test.nzFilt
        else:
            iskip = nskip; jskip = nskip; kskip = nskip
            imin = int(nskip/2); imax = self.Test.nxFilt-int(nskip/2)+1
            jmin = int(nskip/2); jmax = self.Test.nyFilt-int(nskip/2)+1
            kmin = int(nskip/2); kmax = self.Test.nzFilt-int(nskip/2)+1

        #print(imin,imax,iskip)
        xPtsLES = np.arange(self.Test.nx)
        print(xPtsLES[imin+self.Test.imin_:imax+self.Test.imin_:iskip])
            
        if (self.dims==3):
            self.data = self.data[imin:imax:iskip,jmin:jmax:jskip,kmin:kmax:kskip]
            
            for ibin in range(0,self.nbins):
                c = np.ma.masked_where((self.C_local[imin:imax:iskip,jmin:jmax:jskip,kmin:kmax:kskip] >= self.bin_limits[ibin]) &
                                       (self.C_local[imin:imax:iskip,jmin:jmax:jskip,kmin:kmax:kskip] <  self.bin_limits[ibin+1]),
                                       self.C_local[imin:imax:iskip,jmin:jmax:jskip,kmin:kmax:kskip] )
                c_arr = np.ma.getmaskarray(c)
                self.count[ibin] += np.sum(c_arr)
                #self.sum[ibin]   += np.sum(self.data_in[c_arr])
                self.sum[ibin]   += np.sum((self.data[c_arr] + myBias[ibin])**myPower)
                
        elif (self.dims==4):
            # Transform vector quantities into symmetry coordinates
            self.data = np.einsum('...ij,...j->...i',
                                  self.transform[imin:imax:iskip,jmin:jmax:jskip,kmin:kmax:kskip],
                                  self.data[imin:imax:iskip,jmin:jmax:jskip,kmin:kmax:kskip])
            
            for ibin in range(0,self.nbins):
                c = np.ma.masked_where((self.C_local[imin:imax:iskip,jmin:jmax:jskip,kmin:kmax:kskip] >= self.bin_limits[ibin]) &
                                       (self.C_local[imin:imax:iskip,jmin:jmax:jskip,kmin:kmax:kskip] <  self.bin_limits[ibin+1]),
                                       self.C_local[imin:imax:iskip,jmin:jmax:jskip,kmin:kmax:kskip] )
                c_arr = np.ma.getmaskarray(c)
                self.count[ibin] += np.sum(c_arr)
                self.sum[ibin,:] += np.sum(self.data[c_arr,:], axis=0)
                
        elif (self.dims==5):
            # Transform tensor quantities into symmetry coordinates
            self.data = np.einsum( '...ik,...kj->...ij',
                                   self.transform[imin:imax:iskip,jmin:jmax:jskip,kmin:kmax:kskip],
                                   self.data[imin:imax:iskip,jmin:jmax:jskip,kmin:kmax:kskip] )
            self.data = np.einsum( '...ik,...jk->...ij',
                                   self.data,
                                   self.transform[imin:imax:iskip,jmin:jmax:jskip,kmin:kmax:kskip] )
            
            for ibin in range(0,self.nbins):
                c = np.ma.masked_where((self.C_local[imin:imax:iskip,jmin:jmax:jskip,kmin:kmax:kskip] >= self.bin_limits[ibin]) &
                                       (self.C_local[imin:imax:iskip,jmin:jmax:jskip,kmin:kmax:kskip] <  self.bin_limits[ibin+1]),+
                                       self.C_local[imin:imax:iskip,jmin:jmax:jskip,kmin:kmax:kskip] )
                c_arr = np.ma.getmaskarray(c)
                self.count[ibin]   += np.sum(c_arr)
                self.sum[ibin,:,:] += np.sum(self.data[c_arr,:,:], axis=0)

    def report(self):
        # not working for nbinsx3x3, deprecated
        return np.divide(self.sum, 1.0*self.count, 
                         out=np.zeros_like(self.sum), where=self.count!=0)

    def getBinLoc(self):
        return 0.5*(self.bin_limits[1:]+self.bin_limits[:-1])

    def finalize(self):
        # Compute the parallel sums and return the result
        count_all = self.comms.parallel_sum(self.count,'int')
        
        if (self.dims==3):
            sum_all   = self.comms.parallel_sum(self.sum,'float64')
            count_arr = count_all
            
        elif (self.dims==4):
            sum_all   = np.zeros((self.nbins,3))
            count_arr = np.zeros((self.nbins,3))
            for ii in range(3):
                sum_ind = np.array(self.sum[:,ii], copy=True)
                sum_all[:,ii]   = self.comms.parallel_sum(sum_ind,'float64')
                count_arr[:,ii] = count_all
            
        elif (self.dims==5):
            sum_all   = np.zeros((self.nbins,3,3))
            count_arr = np.zeros((self.nbins,3,3))
            for ii in range(3):
                for jj in range(3):
                    sum_ind = np.array(self.sum[:,ii,jj], copy=True)
                    sum_all[:,ii,jj]   = self.comms.parallel_sum(sum_ind,'float64')
                    count_arr[:,ii,jj] = count_all
                    
        return count_all, np.divide(sum_all, count_arr, 
                                    out=np.zeros_like(sum_all), where=count_arr!=0)


# =================================================================
# Print time-series data
# =================================================================
def printTimeResults(outName,time,outHead,outData):
    ntime = len(time)
    with open(outName,"w") as outfile:
        # Write the header
        outfile.write("{:20s}".format("time"))
        for head in outHead:
            outfile.write("    {:20s}".format(head))
        outfile.write("\n")
        
        # Write the data
        for itime in range(0,ntime):
            outfile.write("{:20.12e}".format(time[itime]))
            for data in outData:
                outfile.write("    {:20.12e}".format(data[itime]))
            outfile.write("\n")


# =================================================================
# Print binned data
# =================================================================
def printBinResults(outName,nbins,count,loc,outHead,outData):
    with open(outName,"w") as outfile:
        # Write the header
        outfile.write("{:4s}    {:20s}    {:20s}".format("bin", "count", "loc"))
        for head in outHead:
            outfile.write("    {:20s}".format(head))
        outfile.write("\n")
        
        # Write the data
        for ibin in range(0,nbins):
            outfile.write("{:4d}    {:20d}    {:20.12e}".format(ibin,count[ibin],loc[ibin]))
            for data in outData:
                outfile.write("    {:20.12e}".format(data[ibin]))
            outfile.write("\n")
                
