# ------------------------------------------------------------------------
#
# PyFlow: A GPU-accelerated CFD platform written in Python
#
# @file run_PyFlow_example_dnsbox128.py
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

import sys
import os
import numpy as np
import torch

import PyFlow


# ----------------------------------------------------
# User-specified parameters
# ----------------------------------------------------
class inputConfigClass:
    def __init__(self):
        # Configuration type
        self.configName = "restart"
        
        # Example case in git repository
        inFileDir     = '../../restart/'
        self.configFileStr = inFileDir+'config_dnsbox_128_Lx0.0056'
        self.dataFileStr   = inFileDir+'data_dnsbox_128_Lx0.0056.1_2.50000E-04'
        self.dataFileType  = 'restart'
        
        # Data file to write
        self.fNameOut     = 'data_dnsbox_128_Lx0.0056.2'
        self.numItDataOut = 20

        # Parallel decomposition
        self.nproc_x = 1
        self.nproc_y = 1
        self.nproc_z = 1

        # Physical constants
        #   mu : dynamic viscosity [Pa*s]
        #  rho : mass density [kg/m^3]
        self.mu  = 1.8678e-5
        self.rho = 1.2

        # Time step info
        self.simDt        = 1.0e-6
        self.numIt        = 10
        self.startTime    = 0.0
        
        # SFS model settings
        #   SFSmodel : none, Smagorinsky, gradient, ML
        self.SFSmodel = 'none'

        # Advancer settings
        #   advancerName : Euler, RK4
        #   equationMode : scalar, NS (Navier-Stokes)
        self.advancerName = "RK4"
        self.equationMode = "NS"
        #
        # Pressure solver settings
        #   pSolverMode  : Jacobi, bicgstab
        self.pSolverMode             = "bicgstab"
        self.min_pressure_residual   = 1e-12
        self.max_pressure_iterations = 300
        #
        # Accuracy and precision settings
        self.genericOrder = 2
        self.dtypeTorch   = torch.float64
        self.dtypeNumpy   = np.float64
        
        # Output options
        self.plotState    = False
        self.numItPlotOut = 20


# ----------------------------------------------------
# PyFlow driver
# ----------------------------------------------------
def main(argv):
    # Generate the input configuration
    inputConfig = inputConfigClass()

    # Run PyFlow
    PyFlow.run(inputConfig)
    
    
    
if __name__ == "__main__":
    main(sys.argv[1:])
