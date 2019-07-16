# ------------------------------------------------------------------------
#
# PyFlow: A GPU-accelerated CFD platform written in Python
#
# @file run_PyFlow_adjointVerification.py
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ----------------------------------------------------
# User-specified parameters
# ----------------------------------------------------
class inputConfigClass:
    def __init__(self):
        # Simulation geometry
        #   Options: restart, periodicGaussian, uniform, notAchannel
        self.configName = "restart"
        #configName = "periodicGaussian"
        #configName = "uniform"
        #configName = "notAchannel"
        #self.Nx   = 2
        #self.Ny   = 2
        #self.Nz   = 2
        #self.Nx   = 64
        #self.Ny   = 64
        #self.Nz   = 64
        #self.Lx   = 1.0
        #self.Ly   = 1.0
        #self.Lz   = 1.0
        #isper = [1,1,1]
        
        # Example case in git repository
        #inFileDir     = '../../examples/'
        #configFileStr = inFileDir+'config_dnsbox_128_Lx0.0056'
        #dataFileBStr  = inFileDir+'dnsbox_128_128_Lx0.0056.1_2.50000E-04'
        #dataFileStr   = inFileDir+'data_dnsbox_128_Lx0.0056.1_2.50000E-04'
        #dataFileType  = 'restart'
        
        # Isotropic 128^3 DNS - verification vs. NGA
        #inFileDir     = '../../verification/dnsbox_128_Lx0.0056_NR/test_input_files/'
        #configFileStr = inFileDir+'config_dnsbox_128_Lx0.0056'
        #dataFileBStr  = inFileDir+'dnsbox_128_128_Lx0.0056.1_5.00000E-04'
        #dataFileStr   = inFileDir+'data_dnsbox_128_Lx0.0056.1_5.00000E-04'
        #dataFileType  = 'restart'
        
        # Downsampled 1024^3 DNS - needs SGS model
        self.inFileBase    = '../../verification/'
        self.inFileDir     = self.inFileBase+'downsampled_LES_restart/dnsbox_1024_Lx0.045_NR_run_2/restart_1024_Lx0.045_NR_Delta16_Down16/test_input_files/'
        self.configFileStr = self.inFileDir+'config_dnsbox_1024_Lx0.045_NR_Delta16_Down16_0000'
        self.dataFileBStr  = self.inFileDir+'dnsbox_1024_Lx0.045_NR_Delta16_Down16_'
        self.startFileIt   = 20
        #self.dataFileStr   = self.dataFileBStr + '{:08d}'.format(self.startFileIt)
        self.dataFileStr   = 'restart_test_2.0200000E-04'
        self.dataFileType  = 'restart'
        
        # Data file to write
        #self.fNameOut     = 'data_dnsbox_128_Lx0.0056'
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
        self.numIt        = 20
        self.startTime    = 0.0
        
        # SFS model settings
        #   SFSmodel options: none, Smagorinsky, gradient, ML
        #self.SFSmodel = 'none'
        #self.SFSmodel = 'Smagorinsky'; self.Cs = 0.18; self.expFilterFac = 1.0;
        #self.SFSmodel = 'gradient'
        # ML model input
        self.SFSmodel      = 'ML';
        self.modelDictName = 'test_model.dict'
        self.modelDictSave = 'save_model.dict'
        self.loadModel     = False
        self.saveModel     = True

        # Adjoint training settings
        #   PyFlow will look for a target data file every numCheckpointIt
        self.adjointTraining = True
        self.numCheckpointIt = 20

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
        #self.min_pressure_residual   = 1e-12
        #self.max_pressure_iterations = 300
        self.min_pressure_residual   = 1e-15
        self.max_pressure_iterations = 1000
        #
        # Accuracy and precision settings
        self.genericOrder = 2
        self.dtypeTorch   = torch.float64
        self.dtypeNumpy   = np.float64
        
        # Output options
        self.plotState    = False
        self.numItPlotOut = 20
        
        # Comparison options (deprecated)
        self.useTargetData = False
        if (self.useTargetData):
            self.targetFileBaseStr = dataFileBStr
            self.numItTargetComp = 50

        # Adjoint verification settings
        self.adjointVerification = True
        self.perturbation = 0.0


            
def main(argv):
    # Generate the input configuration
    inputConfig = inputConfigClass()
    maxIter = inputConfig.numIt

    #testName = 'diff_noAdv_noPressure'
    #testName = 'diff_advOld_noPressure'
    testName = 'diff_adv_noPressure'
    #testName = 'diff_noAdv_pressure'
    #testName = 'diff_advOld_pressure'
    #testName = 'diff_adv_pressure'
    perturbationList = [0.0, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, \
                        1e-5, 5e-6, 1e-6]
    
    objList = np.empty(0,dtype=np.float64)
    adjList = np.empty(0,dtype=np.float64)

    for perturbation in perturbationList:
        # Perturb the configuration
        inputConfig.perturbation = perturbation
        
        # Run PyFlow
        obj,adj = PyFlow.run(inputConfig)

        # Save the results
        objList = np.append(objList,obj)
        adjList = np.append(adjList,adj)

        # Turn off adjoint training after perturbations start
        inputConfig.adjointTraining = False
        

    print(objList,adjList)
    print('\n')

    
    # Compute error
    #
    # --> JFM - should this be compared to old_adj or new_adj? What is
    # --> the correct time level for the adjoint comparison?
    #
    deltaList = np.empty(0,dtype=np.float64)
    errorList = np.empty(0,dtype=np.float64)
    old_obj = objList[0]
    old_adj = adjList[0]
    ii = 0
    for Delta in perturbationList[1:]:
        ii += 1
        new_obj = objList[ii]
        num_adj = (new_obj - old_obj)/Delta
        new_adj = adjList[ii]
        error   = old_adj - num_adj
        rel_err = abs(error/old_adj)
        #rel_err = abs(error/num_adj)
        print("Delta={:5.7E}, adjoint={:5.7E}, num_adj={:5.7E}, error={:5.7E}, rel_err={:5.7E}"
              .format(Delta,new_adj,num_adj,error,rel_err))

        # Save for plots
        deltaList = np.append(deltaList,Delta)
        errorList = np.append(errorList,rel_err)

        
    # Plot the error convergence
    fSize = 20
    fig1,ax1 = plt.subplots(figsize=(4,3.125))

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Delta (u_1 perturbation)')
    plt.ylabel('abs( 1 - adj_FD / adj_PyFlow )')

    cvg1 = ax1.plot(deltaList,errorList,marker='o')
    cvg2 = ax1.plot(deltaList,deltaList*1e-1,color='k',linestyle='--')

    ax1.legend(('Error','1:1'))
    plt.title(testName+', iter='+str(maxIter)+', Dt='+str(inputConfig.simDt))
    plt.tight_layout()

    folder = 'figures/'
    if (not os.path.exists(folder)):
        os.mkdir(folder)
    fig1.savefig(folder+'errorCvg_'+testName+
                 '_iter'+str(maxIter)+
                 '_dt'+str(inputConfig.simDt)+'.pdf')
    plt.show()
    
    
                                                                  

# END MAIN
    
if __name__ == "__main__":
    main(sys.argv[1:])
