
import sys
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
        self.dataFileStr   = self.dataFileBStr + '{:08d}'.format(self.startFileIt)
        self.dataFileType  = 'restart'
        
        # Data file to write
        #self.fNameOut     = 'data_dnsbox_128_Lx0.0056'
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
        self.simDt        = 1.0e-6
        self.numIt        = 50
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
        self.numCheckpointIt = 5

        # Solver settings
        #   advancerName options: Euler, RK4
        #   equationMode options: scalar, NS
        #   pSolverMode options:  Jacobi, bicgstab
        #
        self.advancerName = "RK4"
        self.equationMode = "NS"
        #
        # Pressure solver settings
        self.pSolverMode             = "bicgstab"
        self.min_pressure_residual   = 1e-9
        self.max_pressure_iterations = 150
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


            
def main(argv):
    # Generate the input configuration
    inputConfig = inputConfigClass()

    # Run PyFlow
    PyFlow.run(inputConfig)

if __name__ == "__main__":
    main(sys.argv[1:])
