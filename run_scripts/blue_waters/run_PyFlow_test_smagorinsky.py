# ------------------------------------------------------------------------
#
# PyFlow: A GPU-accelerated CFD platform written in Python
#
# @file run_PyFlow_test_smagorinsky.py
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
        
        # Data file to write
        self.fNameOut     = 'dnsbox_1024_Lx0.045_NR_Delta16_Down16_00000020'
        self.numItDataOut = 20

        # Parallel decomposition
        self.nproc_x = 1
        self.nproc_y = 1
        self.nproc_z = 1

        # Physical constants
        self.mu  = 1.8678e-5
        self.rho = 1.2

        # Time step info
        self.simDt        = 1.0e-5
        self.numIt        = 120
        self.startTime    = 0.0
        
        # SFS model settings
        #   SFSmodel options: none, Smagorinsky, gradient, ML
        self.SFSmodel     = 'Smagorinsky'; 
        self.Cs           = 0.18; 
        self.expFilterFac = 1.0;
        # ML model input
        #self.SFSmodel      = 'ML';
        #self.modelDictName = 'LES_model_PyFlow_Adjoint_1024_Lx0_045_NR_Delta16_Down16.dict'
        #self.modelDictSave = 'LES_model_PyFlow_Adjoint_1024_Lx0_045_NR_Delta16_Down16.dict'
        #self.loadModel     = True
        #self.saveModel     = False

        # Adjoint training settings
        #   PyFlow will look for a target data file every numCheckpointIt
        self.adjointTraining = False
        self.numCheckpointIt = 5

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


    # Set the input file base name
    def SetInFileBase(self,name,startIter=20):
        self.inFileBase    = name
        self.inFileDir     = self.inFileBase+'/restart_1024_Lx0.045_NR_Delta16_Down16/'
        self.configFileStr = self.inFileDir+'config_dnsbox_1024_Lx0.045_NR_Delta16_Down16_0000'
        self.dataFileBStr  = self.inFileDir+'dnsbox_1024_Lx0.045_NR_Delta16_Down16_'
        self.startFileIt   = startIter
        self.dataFileStr   = self.dataFileBStr + '{:08d}'.format(self.startFileIt)
        self.dataFileType  = 'restart'


            
def main(argv):
    # Generate the input configuration
    inputConfig = inputConfigClass()

    #  ----------- TRAINING ----------- 
    # Run names, start iteration, max iterations, LES time step sizes, viscosity multiplier
    #runs = (('dnsbox_1024_Lx0.045_NR_run_2',        20,125,1e-5,1.0),
    #        ('dnsbox_1024_Lx0.045_NR_visc0_5_run_1',20,270,2e-5,0.5),
    #        ('dnsbox_1024_Lx0.045_NR_visc2_0_run_1',20,275,5e-6,2.0))

    #  ----------- TESTING ------------ 
    # Run names, start iteration, max iterations, LES time step sizes, viscosity multiplier
    runs = (('dnsbox_1024_Lx0.045_NR_visc0_75_run_1',10, 95,1e-5,0.75),
            ('dnsbox_1024_Lx0.045_NR_visc1_25_run_1',10,115,5e-6,1.25),
            ('dnsbox_1024_Lx0.045_NR_visc1_5_run_1', 10,245,5e-6,1.50))

    # Data file base path
    ifBase = '/projects/sciteam/baxh/downsampled_LES_restart/'

    # Base viscosity
    baseVisc = 1.8678e-5

    # Loop over the runs
    for runName,startIter,maxIter,simDt,viscMult in runs:
        # Set run-specific input options
        inputConfig.SetInFileBase(ifBase+runName, startIter)
        inputConfig.numIt = maxIter
        inputConfig.simDt = simDt
        inputConfig.mu    = baseVisc * viscMult

        # Run PyFlow
        PyFlow.run(inputConfig)

        # Load the model the next time through
        #inputConfig.loadModel = True
    
    
                                                                  

# END MAIN
    
if __name__ == "__main__":
    main(sys.argv[1:])
