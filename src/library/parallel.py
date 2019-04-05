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
# Parallel communication functions
# ----------------------------------------------------
class comms:
    def __init__(self):
        # Get MPI decomposition info
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def parallel_sum(self,sendBuf,type=np.float32):
        if (self.size>1):
            if (hasattr(sendBuf,"__size__")):
                sendLen = len(sendBuf)
            else:
                sendLen = 1
                sendBuf = np.array((sendBuf,))
            
            recvBuf = np.zeros_like(sendBuf)
            self.comm.Allreduce(sendBuf,recvBuf,op=MPI.SUM)
            if (sendLen==1):
                out = recvBuf[0]
            else:
                out = recvBuf
        else:
            # Serial computation; nothing to do
            out = sendBuf
        return out

    def parallel_max(self,sendBuf,type=np.float32):
        if (self.size>1):
            if (hasattr(sendBuf,"__size__")):
                sendLen = len(sendBuf)
            else:
                sendLen = 1
            recvBuf = np.zeros(sendLen, dtype=type)
            self.comm.Allreduce(sendBuf,recvBuf,op=MPI.MAX)
            if (sendLen==1):
                out = recvBuf[0]
            else:
                out = recvBuf
        else:
            # Serial computation; nothing to do
            out = sendBuf
        return out

    def parallel_min(self,sendBuf,type=np.float32):
        if (self.size>1):
            if (hasattr(sendBuf,"__size__")):
                sendLen = len(sendBuf)
            else:
                sendLen = 1
            recvBuf = np.zeros(sendLen, dtype=type)
            self.comm.Allreduce(sendBuf,recvBuf,op=MPI.MIN)
            if (sendLen==1):
                out = recvBuf[0]
            else:
                out = recvBuf
        else:
            # Serial computation; nothing to do
            out = sendBuf
        return out

    
# ----------------------------------------------------
# MPI decomposition
# ----------------------------------------------------
class decomp:
    def __init__(self,Nx,Ny,Nz,nproc_decomp,isper,device,prec=torch.float32):

        self.prec = prec
        self.dtypeNumpy = np.float32

        # Offloading settings
        self.device = device
        
        # ---------------------------------------
        # MPI communicators
        
        # Get the global communicator
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # User-specified decomposition
        self.npx = nproc_decomp[0]
        self.npy = nproc_decomp[1]
        self.npz = nproc_decomp[2]
        
        # Check the decomp
        if (self.size!=self.npx*self.npy*self.npz):
            raise Exception('\nNumber of MPI tasks does not match the specified domain decomposition\n')
        
        #npy = int(np.floor(np.sqrt(self.size)))
        #npx = size//npy
        #if (npx*npy > size):
        #    npy -= 1
        #if (npx*npy > size):
        #    npx -= 1
        #dims  = [npx,npy]
        #isper = [0,0]

        # Cartesian communicator to determine coordinates
        self.cartComm = self.comm.Create_cart(nproc_decomp, periods=isper, reorder=True)
        
        # Proc's location in the cartesian communicator
        self.iproc, self.jproc, self.kproc = self.cartComm.Get_coords(self.rank)

        # Create line communicators
        # Along x
        dir = [True,False,False]
        self.cartCommX = self.cartComm.Sub(dir)
        # Along y
        dir = [False,True,False]
        self.cartCommY = self.cartComm.Sub(dir)
        # Along z
        dir = [False,False,True]
        self.cartCommZ = self.cartComm.Sub(dir)

        # Create plane communicators
        # Along xy
        dir = [True,True,False]
        self.cartCommXY = self.cartComm.Sub(dir)
        # Along xz
        dir = [True,False,True]
        self.cartCommXZ = self.cartComm.Sub(dir)
        # Along yz
        dir = [False,True,True]
        self.cartCommYZ = self.cartComm.Sub(dir)

        
        # ---------------------------------------
        # Domain decomposition
        
        # Grid size
        self.N  = Nx*Ny*Nz
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
        
        # x-decomposition
        #   imin_loc, imax_loc, etc. are position in global grid but
        #   do NOT include overlap cells

        # need imin_loco, etc.
        
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

        #print("rank={}\t imin_={}\timax_={}\tnx_={}\t jmin_={}\tjmax_={}\tny_={}\t kmin_={}\tkmax_={}\tnz_={}"
        #      .format(self.rank,
        #              self.imin_loc,self.imax_loc,self.nx_,
        #              self.jmin_loc,self.jmax_loc,self.ny_,
        #              self.kmin_loc,self.kmax_loc,self.nz_))

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

        
    # ------------------------------------------------
    # Communicate overlap cells for generic state data
    def communicate_border(self,A):
        n1 = self.nxo_
        n2 = self.nyo_
        n3 = self.nzo_
        no = self.nover
        
        self.communicate_border_x(A,n1,n2,n3,no)
        self.communicate_border_y(A,n1,n2,n3,no)
        self.communicate_border_z(A,n1,n2,n3,no)
        
    # --------------------------------------------
    # Communicate overlap cells in the x-direction
    def communicate_border_x(self,A,n1,n2,n3,no):

        # Initialize receive buffer
        recvbuf = np.empty([no,n2,n3],dtype=self.dtypeNumpy)
        icount = no*n2*n3
        
        # Left buffer
        sendbuf = A[no:2*no,:,:].to(torch.device('cpu')).numpy()
        
        # Send left buffer to left neighbor
        isource,idest = self.cartComm.Shift(0,-1)
        self.cartComm.Sendrecv(sendbuf,idest,0,recvbuf,isource,0)

        # Copy the received left buffer to the right overlap cells
        if (isource!=MPI.PROC_NULL):
            A[n1-no:n1,:,:].copy_(torch.from_numpy(recvbuf).to(self.device))

        # Right buffer
        sendbuf = A[n1-2*no:n1-no,:,:].to(torch.device('cpu')).numpy()
        
        # Send right buffer to right neighbor
        isource,idest = self.cartComm.Shift(0,+1)
        self.cartComm.Sendrecv(sendbuf,idest,0,recvbuf,isource,0)

        # Copy the received right buffer to the left overlap cells
        if (isource!=MPI.PROC_NULL):
            A[0:no,:,:].copy_(torch.from_numpy(recvbuf).to(self.device))

        # Clean up
        del recvbuf
        
    # --------------------------------------------
    # Communicate overlap cells in the y-direction
    def communicate_border_y(self,A,n1,n2,n3,no):

        # Initialize the buffers
        # We need to allocate and copy since y is not contiguous in memory
        sendbuf = np.empty([n1,no,n3],dtype=self.dtypeNumpy)
        recvbuf = np.empty([n1,no,n3],dtype=self.dtypeNumpy)
        icount = no*n1*n3
        
        # Lower buffer
        sendbuf = np.copy(A[:,no:2*no,:].to(torch.device('cpu')).numpy())
        
        # Send lower buffer to lower neighbor
        isource,idest = self.cartComm.Shift(1,-1)
        self.cartComm.Sendrecv(sendbuf,idest,0,recvbuf,isource,0)

        # Copy the received lower buffer to the upper overlap cells
        if (isource!=MPI.PROC_NULL):
            A[:,n2-no:n2,:] = torch.from_numpy(recvbuf).to(self.device)

        # Upper buffer
        sendbuf = np.copy(A[:,n2-2*no:n2-no,:].to(torch.device('cpu')).numpy())
        
        # Send upper buffer to upper neighbor
        isource,idest = self.cartComm.Shift(1,+1)
        self.cartComm.Sendrecv(sendbuf,idest,0,recvbuf,isource,0)

        # Copy the received upper buffer to the lower overlap cells
        if (isource!=MPI.PROC_NULL):
            A[:,0:no,:] = torch.from_numpy(recvbuf).to(self.device)

        # Clean up
        del sendbuf
        del recvbuf
        
    # --------------------------------------------
    # Communicate overlap cells in the z-direction
    def communicate_border_z(self,A,n1,n2,n3,no):

        # Initialize the buffers
        # We need to allocate and copy since z is not contiguous in memory
        sendbuf = np.empty([n1,n2,no],dtype=self.dtypeNumpy)
        recvbuf = np.empty([n1,n2,no],dtype=self.dtypeNumpy)
        icount = no*n2*n3
        
        # Front buffer
        sendbuf = np.copy(A[:,:,no:2*no].to(torch.device('cpu')).numpy())
        
        # Send front buffer to front neighbor
        isource,idest = self.cartComm.Shift(2,-1)
        self.cartComm.Sendrecv(sendbuf,idest,0,recvbuf,isource,0)

        # Copy the received front buffer to the back overlap cells
        if (isource!=MPI.PROC_NULL):
            A[:,:,n3-no:n3] = torch.from_numpy(recvbuf).to(self.device)

        # Back buffer
        sendbuf = np.copy(A[:,:,n3-2*no:n3-no].to(torch.device('cpu')).numpy())
        
        # Send back buffer to back neighbor
        isource,idest = self.cartComm.Shift(2,+1)
        self.cartComm.Sendrecv(sendbuf,idest,0,recvbuf,isource,0)

        # Copy the received back buffer to the front overlap cells
        if (isource!=MPI.PROC_NULL):
            A[:,:,0:no] = torch.from_numpy(recvbuf).to(self.device)

        # Clean up
        del sendbuf
        del recvbuf

        
    # --------------------------------------------
    # Plot a 2D figure from the root processor
    #   Currently only set up for npx=2 or npy=2
    def plot_fig_root(self,dr,A,outname):
        no = self.nover
        n1 = self.nx_
        n2 = self.ny_
        k  = int(self.nz/2)
        
        # Initialize buffers
        sendbuf = np.empty([n1,n2],dtype=self.dtypeNumpy)
        recvbuf = np.empty([n1,n2],dtype=self.dtypeNumpy)
        icount = n1*n2
        
        # Root processor output buffer
        if (self.rank==0):
            outbuf = np.empty([self.nx,self.ny],dtype=self.dtypeNumpy)
            outbuf[:n1,:n2] = np.copy(A[no:-no,no:-no,k].to(torch.device('cpu')).numpy())

        # Grab from x-proc
        if (self.npx>1):
            sendbuf = np.copy(A[no:-no,no:-no,k].to(torch.device('cpu')).numpy())
            isource,idest = self.cartComm.Shift(0,+1)
            self.cartComm.Sendrecv(sendbuf,idest,0,recvbuf,isource,0)

            # Copy to output buffer
            if (self.rank==0):
                outbuf[n1:,:n2] = recvbuf

        # Grab from y-proc
        if (self.npy>1):
            sendbuf = np.copy(A[no:-no,no:-no,k].to(torch.device('cpu')).numpy())
            isource,idest = self.cartComm.Shift(1,+1)
            self.cartComm.Sendrecv(sendbuf,idest,0,recvbuf,isource,0)

            # Copy to output buffer
            if (self.rank==0):
                outbuf[:n1,n2:] = recvbuf

        # Plot it
        if (self.rank==0):
            dr.plotData(outbuf,outname)
