# ------------------------------------------------------------------------
#
# PyFlow: A GPU-accelerated CFD platform written in Python
#
# @file run_PyFlow_adjointTraining.py
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
        # Simulation geometry
        #   Options: restart, periodicGaussian, uniform, notAchannel
        self.configName = "restart"
        
        # Downsampled 1024^3 DNS data - needs SGS model
        self.inFileDir     = '../../restart/'
        self.configFileStr = self.inFileDir+'config_dnsbox_1024_Lx0.045_NR_Delta16_Down16_0000'
        self.dataFileType  = 'restart'
        # Data file base string
        self.dataFileBStr  = self.inFileDir+'dnsbox_1024_Lx0.045_NR_Delta16_Down16_'
        # Restart file index
        self.startFileIt   = 20
        # Restart file name
        self.dataFileStr   = self.getTargetFileName(self.startFileIt)
        
        # Data file to write
        self.fNameOut     = 'dnsbox_1024_Lx0.045_NR_Delta16_Down16_00000020'
        self.numItDataOut = 20

        # Parallel decomposition
        self.nproc_x = 2
        self.nproc_y = 1
        self.nproc_z = 1

        # Physical constants
        self.mu  = 1.8678e-5
        self.rho = 1.2

        # Time step info
        self.simDt        = 1.0e-5
        self.numIt        = 10
        self.startTime    = 0.0
        
        # SFS model settings
        #   SFSmodel options: none, Smagorinsky, gradient, ML
        # ML model input
        self.SFSmodel      = 'ML';
        self.modelDictName = 'test_model.dict'
        self.modelDictSave = 'save_model.dict'
        self.loadModel     = False
        self.saveModel     = True

        # Adjoint training settings
        self.adjointTraining = True
        # Number of forward iterations to checkpoint
        #   PyFlow will look for a target data file every numCheckpointIt
        self.numCheckpointIt = 5
        # Learning rate decay schedule (iterations)
        self.LR_decay        = 2500
        # Parameter averaging schedule (iterations)
        self.numSumDistr     = 1
        # Target data velocity correction (pressure projection)
        self.targetDataVelCorr = False

        # Solver settings
        #   advancerName options: Euler, RK4
        #   equationMode options: scalar, NS
        #   pSolverMode options:  Jacobi, bicgstab
        #
        self.advancerName = "Euler"
        self.equationMode = "NS"
        #
        # Pressure solver settings
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

        # Adjoint verification settings
        self.adjointVerification = False
        self.perturbation = 0.0

        
    # ------------------------------------------------
    # PyFlow looks for this function to return target 
    # data file names
    def getTargetFileName(self,iter):
        return self.dataFileBStr + '{:08d}'.format(iter)

          
# ----------------------------------------------------
# PyFlow driver
# ----------------------------------------------------  
def main(argv):
    # Generate the input configuration
    inputConfig = inputConfigClass()

    # Run PyFlow
    PyFlow.run(inputConfig)

# END MAIN
    
if __name__ == "__main__":
    main(sys.argv[1:])
