# ------------------------------------------------------------------------
#
# PyFlow: A GPU-accelerated CFD platform written in Python
#
# @file initial_conditions.py
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
import dataReader as dr


# ----------------------------------------------------
# Configure initial conditions
# ----------------------------------------------------
def generate(IC,config,decomp):
    
    # Default state variable names
    names = ['U','V','W','P']
    startTime = 0.0

    # Local grid sizes
    nx_ = decomp.nx_
    ny_ = decomp.ny_
    nz_ = decomp.nz_
    imin_ = decomp.imin_; imax_ = decomp.imax_
    jmin_ = decomp.jmin_; jmax_ = decomp.jmax_
    kmin_ = decomp.kmin_; kmax_ = decomp.kmax_
    imin_loc = decomp.imin_loc; imax_loc = decomp.imax_loc
    jmin_loc = decomp.jmin_loc; jmax_loc = decomp.jmax_loc
    kmin_loc = decomp.kmin_loc; kmax_loc = decomp.kmax_loc

    # Process initial condition type
    if (IC.configName=='restart'):
        # Already have x,y,z grids from geometry.config
        xGrid = config.xGrid
        yGrid = config.yGrid
        zGrid = config.zGrid
            
        if (IC.dataFileType=='restart'):
            # Read state information from an NGA restart file
            names,startTime = dr.readNGArestart(IC.dataFileStr)
            
        elif (IC.dataFileType=='volume'):
            # Read grid and state data from a volume-format file
            xGrid,yGrid,zGrid,names,dataTime,data = dr.readNGA(IC.dataFileStr)
            
            # Need periodicity information
            
            # Interpolate grid and state data to cell faces
            raise Exception('\nPyFlow: Volume file interpolation not implemented\n')

        # Read restart data later
        data_IC = np.zeros((nx_,ny_,nz_,4),dtype=decomp.dtypeNumpy)
    
    elif (IC.configName=='periodicGaussian'):
        # Initialize the uniform grid
        xGrid = np.linspace(-0.5*config.Lx,0.5*config.Lx,config.Nx+1)
        yGrid = np.linspace(-0.5*config.Ly,0.5*config.Ly,config.Ny+1)
        zGrid = np.linspace(-0.5*config.Lz,0.5*config.Lz,config.Nz+1)
        
        # Initial condition
        uMax = 2.0
        vMax = 2.0
        wMax = 2.0
        stdDev = 0.1
        gaussianBump = ( np.exp(-0.5*(xGrid[imin_loc:imax_loc+1,np.newaxis,np.newaxis]/stdDev)**2) *
                         np.exp(-0.5*(yGrid[np.newaxis,jmin_loc:jmax_loc+1,np.newaxis]/stdDev)**2) *
                         np.exp(-0.5*(zGrid[np.newaxis,np.newaxis,kmin_loc:kmax_loc+1]/stdDev)**2) )
        data_IC = np.zeros((nx_,ny_,nz_,4),dtype=decomp.dtypeNumpy)
        data_IC[:,:,:,0] = uMax * gaussianBump
        data_IC[:,:,:,1] = vMax * gaussianBump
        data_IC[:,:,:,2] = wMax * gaussianBump
        data_IC[:,:,:,3] = 0.0
        del gaussianBump

    elif (IC.configName=='uniform'):
        # Uniform flow
        # Initialize the uniform grid
        xGrid = np.linspace(-0.5*config.Lx,0.5*config.Lx,config.Nx+1)
        yGrid = np.linspace(-0.5*config.Ly,0.5*config.Ly,config.Ny+1)
        zGrid = np.linspace(-0.5*config.Lz,0.5*config.Lz,config.Nz+1)
        
        uMax = 2.0
        vMax = 2.0
        wMax = 2.0
        data_IC = np.zeros((nx_,ny_,nz_,4),dtype=decomp.dtypeNumpy)
        data_IC[:,:,:,0] = uMax
        data_IC[:,:,:,1] = vMax
        data_IC[:,:,:,2] = wMax
        data_IC[:,:,:,3] = 0.0
        
    elif (IC.configName=='notAchannel'):
        # Not a channel; periodic BCs at +/- y; no walls
        # Initialize the uniform grid
        xGrid = np.linspace(-0.5*config.Lx,0.5*config.Lx,config.Nx+1)
        yGrid = np.linspace(-0.5*config.Ly,0.5*config.Ly,config.Ny+1)
        zGrid = np.linspace(-0.5*config.Lz,0.5*config.Lz,config.Nz+1)
        
        uMax = 2.0
        vMax = 0.0
        wMax = 0.0
        amp  = 0.4
        if (decomp.rank==0):
            print("Bulk Re={:7f}".format(rho*uMax*config.Ly/mu))
        parabolaX = ( 6.0*(yGrid[np.newaxis,jmin_loc:jmax_loc+1,np.newaxis] + 0.5*config.Ly)
                      *(0.5*config.Ly - yGrid[np.newaxis,jmin_loc:jmax_loc+1,np.newaxis])
                      /config.Ly**2 )
        Unorm = np.sqrt(uMax**2 + wMax**2)
        data_IC = np.zeros((nx_,ny_,nz_,4),dtype=decomp.dtypeNumpy)
        data_IC[:,:,:,0] = (uMax * parabolaX
                            + amp*Unorm*np.cos(16.0*3.1415926*xGrid[imin_loc:imax_loc+1,np.newaxis,np.newaxis]/config.Lx))
        data_IC[:,:,:,1] = 0.0
        data_IC[:,:,:,2] = (wMax * parabolaX
                            + amp*Unorm*np.cos(16.0*3.1415926*zGrid[np.newaxis,np.newaxis,kmin_loc:kmax_loc+1]/config.Lz))
        data_IC[:,:,:,3] = 0.0
        del parabolaX

    # Return
    return names,startTime,data_IC,xGrid,yGrid,zGrid
