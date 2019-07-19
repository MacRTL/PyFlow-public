# ------------------------------------------------------------------------
#
# PyFlow: A GPU-accelerated CFD platform written in Python
#
# @file sfsmodel_ML.py
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

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

#Number of hidden units
H = 100
number_inputs = 3+1+9+18

class ClosureModel2(nn.Module):
    def __init__(self):
        super(ClosureModel2, self).__init__()
        # input is 28x28
        # padding=2 for same padding
        
        self.fc1 = nn.Linear(number_inputs, H)

        self.fc5 = nn.Linear(H, H)
        self.fc6 = nn.Linear(H, H)

        self.fc7 = nn.Linear(H, 6, bias = True)
        
        
    def forward(self, x):
        
        h1 = F.tanh(self.fc1( x ))
          
        
        h2 = F.tanh(self.fc5( h1 ))
        
        h3 = F.tanh(self.fc6( h2 ))
        
        f_out = self.fc7(h3)
        
        return f_out    


# Number of hidden units
H = 200

#num_inputs = 21 + 6*12
num_inputs = 21 + 12*12

class NeuralNetworkModel2(nn.Module):
    def __init__(self):
        super(NeuralNetworkModel2, self).__init__()
        
        self.fc1 = nn.Linear(num_inputs, H).type(torch.FloatTensor)
        self.fcG = nn.Linear(num_inputs, H).type(torch.FloatTensor)
        
        self.fcG2 = nn.Linear(num_inputs, H).type(torch.FloatTensor)
        
        self.fc2 = nn.Linear(H, H).type(torch.FloatTensor)
        self.fc3 = nn.Linear(H, H).type(torch.FloatTensor)
        self.fc4 = nn.Linear(H, 18).type(torch.FloatTensor)
        
    def forward(self, x):
        
        L1 =  self.fc1( x ) 
        
        H1 = torch.tanh( L1 )
        
        L2 = self.fc2( H1 ) 
        
        H2 = torch.tanh(L2 )
        
        L3 =  self.fcG( x ) 
        G = torch.tanh(L3)
        
        Gate2 = torch.tanh( self.fcG2(x) )
        
        H2_G = G*H2
        
        H2_G_division = H2_G
        
        L4 =  self.fc3( H2_G_division ) 
        
        H3 = torch.tanh(L4)
        
        H4 = H3*Gate2
        
        f_out = self.fc4( H4 )
        
        return f_out
    

# ----------------------------------------------------
# Machine learning model for the SFS residual stress
# ----------------------------------------------------
class residual_stress:
    def __init__(self,inputConfig,decomp,geo):
        
        # Default precision and offloading settings
        self.prec = decomp.prec
        self.device = decomp.device

        # External model type identifier
        self.modelType = 'source'

        # Time step size
        self.simDt = inputConfig.simDt

        # Data sizes
        nx_ = decomp.nx_
        ny_ = decomp.ny_
        nz_ = decomp.nz_
        nxo_ = decomp.nxo_
        nyo_ = decomp.nyo_
        nzo_ = decomp.nzo_

        # Allocate workspace arrays
        self.GX = torch.zeros(nxo_,nyo_,nzo_,1,
                              dtype=self.prec).to(self.device)
        self.GY = torch.zeros(nxo_,nyo_,nzo_,1,
                              dtype=self.prec).to(self.device)
        self.GZ = torch.zeros(nxo_,nyo_,nzo_,1,
                              dtype=self.prec).to(self.device)

        # Filter width
        if (geo.type=='uniform'):
            self.const = geo.dx*geo.dx/12.0
            self.dx    = geo.dx
        else:
            raise Exception('sfsmodel_ML: geometry type not implemented')

        # Initialize PyTorch model object
        self.model = NeuralNetworkModel2()
        self.model.to(self.device)

        # Learning rate and training epoch
        self.LR    = 0.01
        self.epoch = 0

        # Optimizer
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.LR)

        # Do we load an existing ML model state?
        try:
            loadModel = inputConfig.loadModel
        except AttributeError:
            loadModel = False
            if (decomp.rank==0):
                print("\n --> WARNING: ML SFS model configured, but existing model not loaded")

        # If so, load the state
        if (loadModel):
            # Get the name
            try:
                modelDictName = inputConfig.modelDictName
            except AttributeError:
                if (decomp.rank==0):
                    raise Exception('sfsmodel_ML: must specify modelDictName in inputConfig')

            # Look for the file and load it
            if os.path.isfile(modelDictName):
                checkpoint = torch.load(modelDictName)
                self.epoch = checkpoint['epoch']
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.LR = checkpoint['LR']
                if (decomp.rank==0):
                    print("\n --> Loaded ML model state {}".format(modelDictName))
            else:
                if (decomp.rank==0):
                    raise Exception("\nML model state {} not found!".format(modelDictName))

        # Print starting model configuration info
        if (decomp.rank==0):
            print("\n --> ML model initialized at epoch={}, LR={}".format(self.epoch,self.LR))

        # Model save settings
        try:
            self.modelDictSave  = inputConfig.modelDictSave
            self.saveModel = True
        except AttributeError:
            self.saveModel = False

        # Offload model
        #self.model.to(self.device)

        # Normalization ... needs update
        Constant_norm = 10.0*np.array([1.0233662e+00, 1.0241578e+00, 1.0196122e+00, 
                                       5.0992206e+02, 5.2713013e+02, 5.2689368e+02, 5.2550342e+02,
                                       5.1325964e+02, 5.2564642e+02, 5.3132422e+02, 5.2574353e+02,
                                       5.1027972e+02, 9.5621462e+05, 1.2852805e+06, 1.2698791e+06,
                                       1.2744681e+06, 9.6111162e+05, 1.2722050e+06, 1.2891108e+06,
                                       1.2891108e+06, 9.6587112e+05] )  
        
        self.Constant_norm_P = Variable(torch.FloatTensor( Constant_norm )).to(self.device)
        
        
    # ----------------------------------------------------
    # Save the model
    def save(self):
        if (self.saveModel):
            print('Saving model at epoch={}, LR={}...'.format(self.epoch,self.LR))
            state = {'epoch': self.epoch,
                     'LR': self.LR,
                     'model_state_dict': self.model.state_dict(),
                     'optimizer_state_dict': self.optimizer.state_dict(), }
            torch.save(state,self.modelDictSave)
            #torch.save(self.model.state_dict(),self.modelDictSave)
            #torch.save(self.optimizer.state_dict(),self.modelDictSave+'_optimizer')
        
        
    # ----------------------------------------------------
    # Finalize the model
    def finalize(self,comms):
        # Multiply neural network accumlulated gradients by LES time step
        for param in self.model.parameters():
            param.grad.data *= self.simDt

        # Sync the ML model across processes (distribute gradients)
        for param in self.model.parameters():
            tensor0   = param.grad.data.cpu().numpy()
            tensorAvg = comms.parallel_sum(tensor0.ravel())/float(comms.size)
            tensorOut = torch.tensor(tensorAvg.reshape(np.shape(tensor0)))
            param.grad.data = tensorOut.to(self.device)

        # Take an optimizer step
        self.optimizer.step()

        # Update epoch counter and learning rate
        self.epoch += 1
        if ((self.epoch >= 200) and (self.epoch % 250 == 0)):
            self.LR *= 0.5
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0.5*param_group['lr']
        
        
    # ----------------------------------------------------
    # Evaluate the model
    def update(self,u_P,v_P,w_P,requires_grad=False):

        # Set precision for ML model evaluation
        float_u_P = u_P.type(torch.FloatTensor).to(self.device)
        float_v_P = v_P.type(torch.FloatTensor).to(self.device)
        float_w_P = w_P.type(torch.FloatTensor).to(self.device)

        # Compute derivatives of input variables
        #   --> In adjoint training step, u_P, etc., record the computational graph
        u_x_P, u_y_P, u_z_P, u_xx_P, u_yy_P, u_zz_P = self.Finite_Difference_ConCat(float_u_P)
        v_x_P, v_y_P, v_z_P, v_xx_P, v_yy_P, v_zz_P = self.Finite_Difference_ConCat(float_v_P)
        w_x_P, w_y_P, w_z_P, w_xx_P, w_yy_P, w_zz_P = self.Finite_Difference_ConCat(float_w_P)
        
        # Concatenate data for input to ML model
        ML_input_data = torch.stack( (float_u_P, float_v_P, float_w_P,
                                      u_x_P, u_y_P, u_z_P,
                                      v_x_P, v_y_P, v_z_P, w_x_P, w_y_P, w_z_P,
                                      u_xx_P, u_yy_P, u_zz_P, v_xx_P, v_yy_P,
                                      v_zz_P, w_xx_P, w_yy_P, w_zz_P), 3 )
        #print(self.GX.size())
        
        # Evaluate the ML model
        Closure_u_P,Closure_v_P,Closure_w_P = self.ML_Closure(ML_input_data,self.model,
                                                              self.Constant_norm_P)
        
        # Detach the closure model from the computational graph
        if (not requires_grad):
            Closure_u_P = Closure_u_P.detach()
            Closure_v_P = Closure_v_P.detach()
            Closure_w_P = Closure_w_P.detach()
            
        # Update the residual stress source term
        if (self.prec==torch.float64):
            self.GX = Closure_u_P.type(torch.DoubleTensor).to(self.device) # self.GX.copy_() ?
            self.GY = Closure_v_P.type(torch.DoubleTensor).to(self.device)
            self.GZ = Closure_w_P.type(torch.DoubleTensor).to(self.device)
        else:
            # Untested
            self.GX = Closure_u_P.type(torch.FloatTensor).to(self.device)
            self.GY = Closure_v_P.type(torch.FloatTensor).to(self.device)
            self.GZ = Closure_w_P.type(torch.FloatTensor).to(self.device)
        
        # Clean up
        del ML_input_data

        #print(self.GX.size())
        
        
    # -----------------------------------------------------
    # Compute model-input derivatives on the staggered grid
    #   --> All model inputs are computed at cell centers
    def Finite_Difference_ConCat(self,u_P):
        dx = self.dx
        
        u_xx_P = (u_P[2:,:,:] -2*u_P[1:-1,:,:]+ u_P[0:-2,:,:])/(dx*dx)
        u_xx_P_T = (u_P[0:1,:,:] -2*u_P[-1:,:,:]+ u_P[-2:-1,:,:])/(dx*dx)
        u_xx_P_B = (u_P[1:2,:,:] -2*u_P[0:1,:,:]+ u_P[-1:,:,:])/(dx*dx)
        
        u_yy_P = (u_P[:,2:,:] -2*u_P[:,1:-1,:]+ u_P[:,0:-2,:])/(dx*dx)
        u_yy_P_T = (u_P[:,0:1,:] -2*u_P[:,-1:,:]+ u_P[:,-2:-1,:])/(dx*dx)
        u_yy_P_B = (u_P[:,1:2,:] -2*u_P[:,0:1,:]+ u_P[:,-1:,:])/(dx*dx)
        
        u_zz_P = (u_P[:,:,2:] -2*u_P[:,:,1:-1]+ u_P[:,:,0:-2])/(dx*dx)
        u_zz_P_T = (u_P[:,:,0:1] -2*u_P[:,:,-1:]+ u_P[:,:,-2:-1])/(dx*dx)
        u_zz_P_B = (u_P[:,:,1:2] -2*u_P[:,:,0:1]+ u_P[:,:,-1:])/(dx*dx)        

        u_xx_P = torch.cat((u_xx_P, u_xx_P_T), 0)
        u_xx_P = torch.cat((u_xx_P_B, u_xx_P), 0)       

        u_yy_P = torch.cat((u_yy_P, u_yy_P_T), 1)
        u_yy_P = torch.cat((u_yy_P_B, u_yy_P), 1)  

        u_zz_P = torch.cat((u_zz_P, u_zz_P_T), 2)
        u_zz_P = torch.cat((u_zz_P_B, u_zz_P), 2)    

        u_x_P = (u_P[2:,:,:] - u_P[0:-2,:,:])/(2*dx)

        u_x_P_T = (u_P[0:1,:,:] - u_P[-2:-1,:,:])/(2*dx)
        u_x_P_B = (u_P[1:2,:,:] - u_P[-1:,:,:])/(2*dx)


        u_y_P = (u_P[:,2:,:] - u_P[:,0:-2,:])/(2*dx)     


        u_y_P_T = (u_P[:,0:1,:] - u_P[:,-2:-1,:])/(2*dx)
        u_y_P_B = (u_P[:,1:2,:] - u_P[:,-1:,:])/(2*dx)

        u_z_P = (u_P[:,:,2:] - u_P[:,:,0:-2])/(2*dx)    

        u_z_P_T = (u_P[:,:,0:1] - u_P[:,:,-2:-1])/(2*dx)
        u_z_P_B = (u_P[:,:,1:2] - u_P[:,:,-1:])/(2*dx)

        u_x_P = torch.cat((u_x_P, u_x_P_T), 0)
        u_x_P = torch.cat((u_x_P_B, u_x_P), 0)    

        u_y_P = torch.cat((u_y_P, u_y_P_T ), 1)
        u_y_P = torch.cat((u_y_P_B, u_y_P), 1)

        u_z_P = torch.cat((u_z_P, u_z_P_T ), 2)
        u_z_P = torch.cat((u_z_P_B, u_z_P), 2)  
    
        return u_x_P, u_y_P, u_z_P, u_xx_P, u_yy_P, u_zz_P
        

    
        
    # -----------------------------------------------------
    # Evaluate ML closure model
    def ML_Closure(self,input_data,model_in,Constant_norm_P):
        dx = self.dx
    
        input_data_xPlus   = torch.cat( (input_data[1:,:,:,0:12], input_data[0:1,:,:,0:12] ), 0 )
        input_data_xMinus = torch.cat( (input_data[-1:,:,:,0:12], input_data[0:-1,:,:,0:12] ), 0 )

        input_data_yPlus   = torch.cat( (input_data[:,1:,:,0:12], input_data[:,0:1,:,0:12] ), 1 )
        input_data_yMinus = torch.cat( (input_data[:,-1:,:,0:12], input_data[:,0:-1,:,0:12] ), 1 ) 

        input_data_zPlus   = torch.cat( (input_data[:,:,1:,0:12], input_data[:,:,0:1,0:12] ), 2 )
        input_data_zMinus = torch.cat( (input_data[:,:,-1:,0:12], input_data[:,:,0:-1,0:12] ), 2 )

        input_data_xPlusPlus   = torch.cat( (input_data[2:,:,:,0:12], input_data[0:2,:,:,0:12] ), 0 )
        input_data_xMinusMinus = torch.cat( (input_data[-2:,:,:,0:12], input_data[0:-2,:,:,0:12] ), 0 ) 

        input_data_yPlusPlus   = torch.cat( (input_data[:,2:,:,0:12], input_data[:,0:2,:,0:12] ), 1 )
        input_data_yMinusMinus = torch.cat( (input_data[:,-2:,:,0:12], input_data[:,0:-2,:,0:12] ), 1 )     

        input_data_zPlusPlus   = torch.cat( (input_data[:,:,2:,0:12], input_data[:,:,0:2,0:12] ), 2 )
        input_data_zMinusMinus = torch.cat( (input_data[:,:,-2:,0:12], input_data[:,:,0:-2,0:12] ), 2 )    


        Constant_norm_P22 = Constant_norm_P[0:12]

        input_data_total = torch.cat( (input_data/Constant_norm_P,
                                       input_data_xPlus/Constant_norm_P22,
                                       input_data_xMinus/Constant_norm_P22,
                                       input_data_yPlus/Constant_norm_P22,
                                       input_data_yMinus/Constant_norm_P22,
                                       input_data_zPlus/Constant_norm_P22,
                                       input_data_zMinus/Constant_norm_P22 ,
                                       input_data_xPlusPlus/Constant_norm_P22,
                                       input_data_xMinusMinus/Constant_norm_P22,
                                       input_data_yPlusPlus/Constant_norm_P22,
                                       input_data_yMinusMinus/Constant_norm_P22,
                                       input_data_zPlusPlus/Constant_norm_P22,
                                       input_data_zMinusMinus/Constant_norm_P22), 3 )

        input_data2 = input_data_total

        #input_data2 = input_data/Constant_norm_P


        model_output = model_in( input_data2 )

        p_N = model_output


        ##### u closure terms
        p_x_N_u = (p_N[2:, :, : ,0:1] - p_N[0:-2, :, : ,0:1] )/(2*dx)

        p_x_N_T_u = (p_N[0:1,:,:,0:1] - p_N[-2:-1,:,:,0:1])/(2*dx)
        p_x_N_B_u = (p_N[1:2,:,:,0:1] - p_N[-1:,:,:,0:1])/(2*dx)

        p_x_N_u = torch.cat((p_x_N_u, p_x_N_T_u), 0)
        p_x_N_u = torch.cat((p_x_N_B_u, p_x_N_u), 0)            

        p_y_N_u = (p_N[:, 2:, : , 1:2] - p_N[:, 0:-2, : ,1:2] )/(2*dx)

        p_y_N_T_u = (p_N[:, 0:1,:,1:2] - p_N[:,-2:-1,:,1:2])/(2*dx)
        p_y_N_B_u = (p_N[:,1:2,:,1:2] - p_N[:,-1:,:,1:2])/(2*dx)

        p_y_N_u = torch.cat((p_y_N_u, p_y_N_T_u), 1)
        p_y_N_u = torch.cat((p_y_N_B_u, p_y_N_u), 1)          

        p_z_N_u = (p_N[:, :, 2: ,2:3] - p_N[:, :, 0:-2 ,2:3] )/(2*dx)

        p_z_N_T_u = (p_N[:,:, 0:1,2:3] - p_N[:,:,-2:-1,2:3])/(2*dx)
        p_z_N_B_u = (p_N[:,:,1:2,2:3] - p_N[:,:,-1:,2:3])/(2*dx)

        p_z_N_u = torch.cat((p_z_N_u, p_z_N_T_u), 2)
        p_z_N_u = torch.cat((p_z_N_B_u, p_z_N_u), 2)  

        ########## v closure terms
        p_x_N_v = (p_N[2:, :, : ,3:4] - p_N[0:-2, :, : ,3:4] )/(2*dx)

        p_x_N_T_v = (p_N[0:1,:,:,3:4] - p_N[-2:-1,:,:,3:4])/(2*dx)
        p_x_N_B_v = (p_N[1:2,:,:,3:4] - p_N[-1:,:,:,3:4])/(2*dx)

        p_x_N_v = torch.cat((p_x_N_v, p_x_N_T_v), 0)
        p_x_N_v = torch.cat((p_x_N_B_v, p_x_N_v), 0)            

        p_y_N_v = (p_N[:, 2:, : , 4:5] - p_N[:, 0:-2, : ,4:5] )/(2*dx)

        p_y_N_T_v = (p_N[:, 0:1,:,4:5] - p_N[:,-2:-1,:,4:5])/(2*dx)
        p_y_N_B_v = (p_N[:,1:2,:,4:5] - p_N[:,-1:,:,4:5])/(2*dx)

        p_y_N_v = torch.cat((p_y_N_v, p_y_N_T_v), 1)
        p_y_N_v = torch.cat((p_y_N_B_v, p_y_N_v), 1)          

        p_z_N_v = (p_N[:, :, 2: ,5:6] - p_N[:, :, 0:-2 ,5:6] )/(2*dx)

        p_z_N_T_v = (p_N[:,:, 0:1,5:6] - p_N[:,:,-2:-1,5:6])/(2*dx)
        p_z_N_B_v = (p_N[:,:,1:2,5:6] - p_N[:,:,-1:,5:6])/(2*dx)

        p_z_N_v = torch.cat((p_z_N_v, p_z_N_T_v), 2)
        p_z_N_v = torch.cat((p_z_N_B_v, p_z_N_v), 2)     

        #########  w closure terms

        p_x_N_w = (p_N[2:, :, : ,6:7] - p_N[0:-2, :, : ,6:7] )/(2*dx)

        p_x_N_T_w = (p_N[0:1,:,:,6:7] - p_N[-2:-1,:,:,6:7])/(2*dx)
        p_x_N_B_w = (p_N[1:2,:,:,6:7] - p_N[-1:,:,:,6:7])/(2*dx)

        p_x_N_w = torch.cat((p_x_N_w, p_x_N_T_w), 0)
        p_x_N_w = torch.cat((p_x_N_B_w, p_x_N_w), 0)            

        p_y_N_w = (p_N[:, 2:, : , 7:8] - p_N[:, 0:-2, : ,7:8] )/(2*dx)

        p_y_N_T_w = (p_N[:, 0:1,:,7:8] - p_N[:,-2:-1,:,7:8])/(2*dx)
        p_y_N_B_w = (p_N[:,1:2,:,7:8] - p_N[:,-1:,:,7:8])/(2*dx)

        p_y_N_w = torch.cat((p_y_N_w, p_y_N_T_w), 1)
        p_y_N_w = torch.cat((p_y_N_B_w, p_y_N_w), 1)          

        p_z_N_w = (p_N[:, :, 2: ,8:9] - p_N[:, :, 0:-2 ,8:9] )/(2*dx)

        p_z_N_T_w = (p_N[:,:, 0:1,8:9] - p_N[:,:,-2:-1,8:9])/(2*dx)
        p_z_N_B_w = (p_N[:,:,1:2,8:9] - p_N[:,:,-1:,8:9])/(2*dx)

        p_z_N_w = torch.cat((p_z_N_w, p_z_N_T_w), 2)
        p_z_N_w = torch.cat((p_z_N_B_w, p_z_N_w), 2)  


        Closure_u_P = p_x_N_u + p_y_N_u + p_z_N_u 
        Closure_v_P = p_x_N_v + p_y_N_v + p_z_N_v 
        Closure_w_P = p_x_N_w + p_y_N_w + p_z_N_w 
      
        return Closure_u_P, Closure_v_P, Closure_w_P
