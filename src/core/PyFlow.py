# ------------------------------------------------------------------------
#
# PyFlow: A GPU-accelerated CFD platform written in Python
#
# @author Justin A. Sirignano
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import datasets, transforms
from torch.autograd import Variable

import time
import copy

import dataReaderFeb2019 as dr

x_max16 = np.array( [1.63781185e+01, 1.58859625e+01, 1.42067013e+01, 1.0,
 4.07930586e+04, 6.61947031e+04, 6.84387969e+04, 7.55127656e+04,
 4.52174883e+04, 6.18495938e+04, 6.63429844e+04, 6.53548398e+04,
 4.72169219e+04, 2.35592576e+08, 5.82017792e+08, 5.66800960e+08,
 6.06438195e+09, 6.44448256e+09, 8.68913254e+09, 4.70188640e+08,
 5.84262976e+08, 2.25895120e+08, 3.91230048e+08, 6.27150400e+08,
 7.79649920e+08, 2.32722912e+08, 3.16961472e+08, 2.27726320e+08,
 5.91761536e+08, 6.22432256e+08, 4.68478272e+08, 1.25525498e+00,
 4.80542541e-01, 9.08308625e-01, 4.80542541e-01, 1.10598314e+00,
 7.01258421e-01, 9.08308625e-01, 7.01258421e-01, 9.65657115e-01] )

x_max16 = x_max16[0:-9]

x_max_P = torch.FloatTensor( x_max16 ).cuda()

#Number of grid points
N = 128

#data = dr.readNGA('10secondsOutputGaussianFilter8xNR_Adjoint_005')
#data = data[-1][:,:,:,0:4]
#names,data = dr.readNGArestart('data_LESbox_128_Dx8_Lx0.045.1_2.01000E-04.filter')

data = dr.readNGA('/projects/sciteam/baxh/Downsampled_Data_Folder/data_1024_Lx0.045_NR_Delta32_downsample8x/filtered_vol_dnsbox_1024_Lx0.045_NR_00000020_coarse')

data = data[-1][:,:,:,0:4]



data_filtered_DNS_128 =  data


data_target10 = dr.readNGA('/projects/sciteam/baxh/Downsampled_Data_Folder/data_1024_Lx0.045_NR_Delta32_downsample8x/filtered_vol_dnsbox_1024_Lx0.045_NR_00000030_coarse')

data_target10_np = data_target10[-1][:,:,:,0:4]

target_P = torch.FloatTensor( data_target10_np ).cuda()




####Neural Network solution


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


    
model = ClosureModel2()

model_name = 'LES_model_NR_March2019'


model.load_state_dict(torch.load(model_name))

       
model.cuda()

#Set up some vectors which will be helpful for the PyTorch finite difference implementation..

#initial condition

IC_u_np = data_filtered_DNS_128[:,:,:,0:1]

#IC_v_np = np.zeros( ( 64, 64,1) )
IC_v_np = data_filtered_DNS_128[:,:,:,1:2]

IC_w_np = data_filtered_DNS_128[:,:,:,2:3]

IC_p_np  = data_filtered_DNS_128[:,:,:,3:4]

IC_zeros_np = np.zeros( ( N, N,N,1) )

dx = 0.045/float(N)

Num_pressure_iterations = 100

mu = 1.8678e-5

rho = 1.2

#dt_LES = 0.000000001
dt_LES = 0.0000025

#T_LES = 2*0.0001

T_LES = 1*0.0001



#Smagorinsky Closure Model
def Smagorinsky(u_x_P, u_y_P, u_z_P, v_x_P, v_y_P, v_z_P, w_x_P, w_y_P, w_z_P, Closure_u_P, Closure_v_P, Closure_w_P):
    
    S_11 = 0.5*(u_x_P + u_x_P)
    S_12 = 0.5*(u_y_P + v_x_P)
    S_13 = 0.5*(u_z_P + w_x_P)
    S_22 = 0.5*(v_y_P + v_y_P)
    S_23 = 0.5*(v_z_P + w_y_P)
    S_33 = 0.5*(w_z_P + w_z_P)
    
    Delta_Smag = 32.0*0.045/1024.0
    C_S = 0.18
    
    S_sum = S_11*S_11 + S_22*S_22 + S_33*S_33 + 2.0*( S_12*S_12 + S_13*S_13  + S_23*S_23 )
    
    S_magnitude = torch.sqrt( 2.0*S_sum )
    
    Coeff = -2.0*(C_S*Delta_Smag)*(C_S*Delta_Smag)*S_magnitude
    
    #S_11 = 
    
    Closure_u_P[1:-1,1:-1,1:-1, :] =  Coeff[1:-1,1:-1, 1:-1,:]*( (S_11[2:,1:-1, 1:-1,:] - S_11[0:-2,1:-1, 1:-1,:])    +  (S_12[1:-1,2:, 1:-1,:] - S_12[1:-1,0:-2, 1:-1,:]) +  (S_13[1:-1,1:-1, 2:,:] - S_13[1:-1, 1:-1, 0:-2,:])  )/(2*dx)
    
    Closure_v_P[1:-1,1:-1,1:-1, :] =  Coeff[1:-1,1:-1, 1:-1,:]*( (S_12[2:,1:-1, 1:-1,:] - S_12[0:-2,1:-1, 1:-1,:])    +  (S_22[1:-1,2:, 1:-1,:] - S_22[1:-1,0:-2, 1:-1,:]) +  (S_23[1:-1,1:-1, 2:,:] - S_23[1:-1, 1:-1, 0:-2,:])  )/(2*dx)
    
    Closure_w_P[1:-1,1:-1,1:-1, :] =  Coeff[1:-1,1:-1, 1:-1,:]*( (S_13[2:,1:-1, 1:-1,:] - S_13[0:-2,1:-1, 1:-1,:])    +  (S_23[1:-1,2:, 1:-1,:] - S_23[1:-1,0:-2, 1:-1,:]) +  (S_33[1:-1,1:-1, 2:,:] - S_33[1:-1, 1:-1, 0:-2,:])  )/(2*dx)
    
    
    return Closure_u_P, Closure_v_P, Closure_w_P
    
    

#Upwind FD scheme
def FiniteDifference_u_UPWIND(u_P):
    #u derivatives
    u_x_P[1:-1,:,:,:] = (u_P[1:-1,:,:,:] - u_P[0:-2,:,:,:])/(1*dx)
    u_y_P[:,1:-1,:,:] = (u_P[:,1:-1,:,:] - u_P[:,0:-2,:,:])/(1*dx)
    u_z_P[:,:,1:-1,:] = (u_P[:,:,1:-1,:] - u_P[:,:,0:-2,:])/(1*dx)


    #u periodic boundary conditions
    u_x_P[-1,:,:,:] = (u_P[-1,:,:,:] - u_P[-2,:,:,:])/(1*dx)
    u_x_P[0,:,:,:] = (u_P[0,:,:,:] - u_P[-1,:,:,:])/(1*dx)
    
    u_y_P[:,-1,:,:] = (u_P[:,-1,:,:] - u_P[:,-2,:,:])/(1*dx)
    u_y_P[:,0,:,:] = (u_P[:,0,:,:] - u_P[:,-1,:,:])/(1*dx)
    
    u_z_P[:,:,-1,:] = (u_P[:,:,-1,:] - u_P[:,:,-2,:])/(1*dx)
    u_z_P[:,:,0,:] = (u_P[:,:,0,:] - u_P[:,:,-1,:])/(1*dx)

    return u_x_P, u_y_P, u_z_P





def FiniteDifference_u_DOWNWIND(u_P):
    #u derivatives
    u_x_P[1:-1,:,:,:] = (u_P[2:,:,:,:] - u_P[1:-1,:,:,:])/(1*dx)
    u_y_P[:,1:-1,:,:] = (u_P[:,2:,:,:] - u_P[:,1:-1,:,:])/(1*dx)
    u_z_P[:,:,1:-1,:] = (u_P[:,:,2:,:] - u_P[:,:,1:-1,:])/(1*dx)


    #u periodic boundary conditions
    u_x_P[-1,:,:,:] = (u_P[0,:,:,:] - u_P[-1,:,:,:])/(1*dx)
    u_x_P[0,:,:,:] = (u_P[1,:,:,:] - u_P[0,:,:,:])/(1*dx)
    
    u_y_P[:,-1,:,:] = (u_P[:,0,:,:] - u_P[:,-1,:,:])/(1*dx)
    u_y_P[:,0,:,:] = (u_P[:,1,:,:] - u_P[:,0,:,:])/(1*dx)
    
    u_z_P[:,:,-1,:] = (u_P[:,:,0,:] - u_P[:,:,-1,:])/(1*dx)
    u_z_P[:,:,0,:] = (u_P[:,:,1,:] - u_P[:,:,0,:])/(1*dx)


    return u_x_P, u_y_P, u_z_P


#Central Difference FD scheme
def FiniteDifference_u(u_P, u_x_P, u_y_P, u_z_P, u_xx_P, u_yy_P, u_zz_P, u_xy_P, u_xz_P, u_yz_P):
    #u derivatives
    u_x_P[1:-1,:,:,:] = (u_P[2:,:,:,:] - u_P[0:-2,:,:,:])/(2*dx)
    u_y_P[:,1:-1,:,:] = (u_P[:,2:,:,:] - u_P[:,0:-2,:,:])/(2*dx)
    u_z_P[:,:,1:-1,:] = (u_P[:,:,2:,:] - u_P[:,:,0:-2,:])/(2*dx)
    u_xx_P[1:-1,:,:,:] = (u_P[2:,:,:,:] -2*u_P[1:-1,:,:,:]+ u_P[0:-2,:,:,:])/(dx*dx)   
    u_yy_P[:,1:-1,:,:] = (u_P[:,2:,:,:] -2*u_P[:,1:-1,:,:]+ u_P[:,0:-2,:,:])/(dx*dx)  
    u_zz_P[:,:,1:-1,:] = (u_P[:,:,2:,:] -2*u_P[:,:,1:-1,:]+ u_P[:,:,0:-2,:])/(dx*dx) 

    u_xy_P[1:-1,1:-1,:,:] = (u_P[2:,2:,:,:] - u_P[0:-2,2:,:,:] - u_P[2:,0:-2,:,:] + u_P[0:-2,0:-2,:,:] )/(4*dx*dx)     
    u_xz_P[1:-1,:,1:-1,:] = (u_P[2:,:,2:,:] - u_P[0:-2,:,2:,:] - u_P[2:,:,0:-2,:] + u_P[0:-2,:,0:-2,:] )/(4*dx*dx)     
    u_yz_P[:,1:-1,1:-1,:] = (u_P[:,2:,2:,:] - u_P[:,0:-2,2:,:] - u_P[:,2:,0:-2,:] + u_P[:,0:-2,0:-2,:] )/(4*dx*dx)    
    
    #higher-order finite difference....
    u_x_P[3:-3,1:-1,1:-1, :] = ( -(1.0/60.0)*u_P[0:-6,1:-1,1:-1,:] + (3.0/20.0)*u_P[1:-5,1:-1,1:-1,:] -(3.0/4.0)*u_P[2:-4,1:-1,1:-1,:] +  (3.0/4.0)*u_P[4:-2,1:-1,1:-1,:] - (3.0/20.0)*u_P[5:-1,1:-1,1:-1,:] + (1.0/60.0)*u_P[6:,1:-1,1:-1,:])/(dx)
    
    u_y_P[1:-1,3:-3,1:-1,:] = ( -(1.0/60.0)*u_P[1:-1,0:-6,1:-1,:] + (3.0/20.0)*u_P[1:-1,1:-5,1:-1,:] -(3.0/4.0)*u_P[1:-1,2:-4,1:-1,0:3] +  (3.0/4.0)*u_P[1:-1,4:-2,1:-1,:] - (3.0/20.0)*u_P[1:-1,5:-1,1:-1,:] + (1.0/60.0)*u_P[1:-1,6:,1:-1,:])/(dx)       
    
    u_z_P[1:-1,1:-1,3:-3,:] = ( -(1.0/60.0)*u_P[1:-1,1:-1,0:-6,:] + (3.0/20.0)*u_P[1:-1,1:-1,1:-5,:] -(3.0/4.0)*u_P[1:-1,1:-1,2:-4,:] +  (3.0/4.0)*u_P[1:-1,1:-1,4:-2,:] - (3.0/20.0)*u_P[1:-1,1:-1,5:-1,:] + (1.0/60.0)*u_P[1:-1,1:-1,6:,:])/(dx)      
    
    #data_in[12:-12,12:-12,12:-12,4:7] = ( -(1.0/60.0)*data_in[0:-24,12:-12,12:-12,0:3] + (3.0/20.0)*data_in[4:-20,12:-12,12:-12,0:3] -(3.0/4.0)*data_in[8:-16,12:-12,12:-12,0:3] +  (3.0/4.0)*data_in[16:-8,12:-12,12:-12,0:3] - (3.0/20.0)*data_in[20:-4,12:-12,12:-12,0:3] + (1.0/60.0)*data_in[24:,12:-12,12:-12,0:3])/(Delta_LES_grid)    
    


    #u periodic boundary conditions
    u_x_P[-1,:,:,:] = (u_P[0,:,:,:] - u_P[-2,:,:,:])/(2*dx)
    u_x_P[0,:,:,:] = (u_P[1,:,:,:] - u_P[-1,:,:,:])/(2*dx)
    
    u_y_P[:,-1,:,:] = (u_P[:,0,:,:] - u_P[:,-2,:,:])/(2*dx)
    u_y_P[:,0,:,:] = (u_P[:,1,:,:] - u_P[:,-1,:,:])/(2*dx)
    
    u_z_P[:,:,-1,:] = (u_P[:,:,0,:] - u_P[:,:,-2,:])/(2*dx)
    u_z_P[:,:,0,:] = (u_P[:,:,1,:] - u_P[:,:,-1,:])/(2*dx)

    u_xx_P[-1,:,:,:] = (u_P[0,:,:,:] -2*u_P[-1,:,:,:]+ u_P[-2,:,:,:])/(dx*dx)   
    u_xx_P[0,:,:,:] = (u_P[1,:,:,:] -2*u_P[0,:,:,:]+ u_P[-1,:,:,:])/(dx*dx)   

    u_yy_P[:,-1,:,:] = (u_P[:,0,:,:] -2*u_P[:,-1,:,:]+ u_P[:,-2,:,:])/(dx*dx)  
    u_yy_P[:,0,:,:] = (u_P[:,1,:,:] -2*u_P[:,0,:,:]+ u_P[:,-1,:,:])/(dx*dx)  

    u_zz_P[:,:,-1,:] = (u_P[:,:,0,:] -2*u_P[:,:,-1,:]+ u_P[:,:,-2,:])/(dx*dx)  
    u_zz_P[:,:,0,:] = (u_P[:,:,1,:] -2*u_P[:,:,0,:]+ u_P[:,:,-1,:])/(dx*dx) 

    u_xy_P[-1,1:-1,:,:] = (u_P[0,2:,:,:] - u_P[-2,2:,:,:] - u_P[0,0:-2,:,:] + u_P[-2,0:-2,:,:] )/(4*dx*dx)   
    u_xy_P[0,1:-1,:,:] = (u_P[1,2:,:,:] - u_P[-1,2:,:,:] - u_P[1,0:-2,:,:] + u_P[-1,0:-2,:,:] )/(4*dx*dx)     
    u_xy_P[1:-1,-1,:,:] = (u_P[2:,0,:,:] - u_P[0:-2,0,:,:] - u_P[2:,-2,:,:] + u_P[0:-2,-2,:,:] )/(4*dx*dx)
    u_xy_P[1:-1,0,:,:] = (u_P[2:,1,:,:] - u_P[0:-2,1,:,:] - u_P[2:,-1,:,:] + u_P[0:-2,-1,:,:] )/(4*dx*dx)
    u_xy_P[-1,-1,:,:] = (u_P[0,0,:,:] - u_P[-2,0,:,:] - u_P[0,-2,:,:] + u_P[-2,-2,:,:] )/(4*dx*dx)   
    u_xy_P[-1,0,:,:] = (u_P[0,1,:,:] - u_P[-2,1,:,:] - u_P[0,-1,:,:] + u_P[-2,-1,:,:] )/(4*dx*dx)
    u_xy_P[0,-1,:,:] = (u_P[1,0,:,:] - u_P[-1,0,:,:] - u_P[1,-2,:,:] + u_P[-1,-2,:,:] )/(4*dx*dx) 
    u_xy_P[0,0,:,:] = (u_P[1,1,:,:] - u_P[-1,1,:,:] - u_P[1,-1,:,:] + u_P[-1,-1,:,:] )/(4*dx*dx)

    u_xy_P[-1,1:-1,:,:] = (u_P[0,2:,:,:] - u_P[-2,2:,:,:] - u_P[0,0:-2,:,:] + u_P[-2,0:-2,:,:] )/(4*dx*dx)   
    u_xy_P[0,1:-1,:,:] = (u_P[1,2:,:,:] - u_P[-1,2:,:,:] - u_P[1,0:-2,:,:] + u_P[-1,0:-2,:,:] )/(4*dx*dx)     
    u_xy_P[1:-1,-1,:,:] = (u_P[2:,0,:,:] - u_P[0:-2,0,:,:] - u_P[2:,-2,:,:] + u_P[0:-2,-2,:,:] )/(4*dx*dx)
    u_xy_P[1:-1,0,:,:] = (u_P[2:,1,:,:] - u_P[0:-2,1,:,:] - u_P[2:,-1,:,:] + u_P[0:-2,-1,:,:] )/(4*dx*dx)
    u_xy_P[-1,-1,:,:] = (u_P[0,0,:,:] - u_P[-2,0,:,:] - u_P[0,-2,:,:] + u_P[-2,-2,:,:] )/(4*dx*dx)   
    u_xy_P[-1,0,:,:] = (u_P[0,1,:,:] - u_P[-2,1,:,:] - u_P[0,-1,:,:] + u_P[-2,-1,:,:] )/(4*dx*dx)
    u_xy_P[0,-1,:,:] = (u_P[1,0,:,:] - u_P[-1,0,:,:] - u_P[1,-2,:,:] + u_P[-1,-2,:,:] )/(4*dx*dx) 
    u_xy_P[0,0,:,:] = (u_P[1,1,:,:] - u_P[-1,1,:,:] - u_P[1,-1,:,:] + u_P[-1,-1,:,:] )/(4*dx*dx)

    return u_x_P, u_y_P, u_z_P, u_xx_P, u_yy_P, u_zz_P, u_xy_P, u_xz_P, u_yz_P


#Constant0_P = Variable( torch.FloatTensor( Constant0_np ) )

model.eval()

for iterations in range(1):
    
    time1 = time.time()
    
    #for param_group in optimizer.param_groups:
    #        param_group['lr'] = 0.1*LR
    
    #optimizer.zero_grad()
    
    u_P =  torch.FloatTensor( IC_u_np ).cuda()
    v_P = torch.FloatTensor( IC_v_np ).cuda()
    w_P = torch.FloatTensor( IC_w_np ).cuda()
    
    Closure_u_P =  torch.FloatTensor( IC_zeros_np ).cuda()
    Closure_v_P = torch.FloatTensor( IC_zeros_np ).cuda()
    Closure_w_P = torch.FloatTensor( IC_zeros_np ).cuda()    
    
    p_P =  torch.FloatTensor( IC_zeros_np ).cuda()
    p_OLD_P =  torch.FloatTensor( IC_zeros_np).cuda()
    p_x_P =  torch.FloatTensor( IC_p_np ).cuda()
    p_y_P =  torch.FloatTensor( IC_p_np ).cuda()
    p_z_P =  torch.FloatTensor( IC_p_np ).cuda()
    
    u_x_P =  torch.FloatTensor( IC_u_np ).cuda()
    u_y_P =  torch.FloatTensor( IC_u_np ).cuda()
    u_z_P =  torch.FloatTensor( IC_u_np ).cuda()
    u_xx_P =  torch.FloatTensor( IC_u_np ).cuda()
    u_yy_P =  torch.FloatTensor( IC_u_np ).cuda()
    u_zz_P =  torch.FloatTensor( IC_u_np ).cuda()
    u_xz_P =  torch.FloatTensor( IC_u_np ).cuda()
    u_xy_P =  torch.FloatTensor( IC_u_np ).cuda()
    u_yz_P =  torch.FloatTensor( IC_u_np ).cuda()
    
    v_x_P =  torch.FloatTensor( IC_v_np ).cuda()
    v_y_P =  torch.FloatTensor( IC_v_np ).cuda()
    v_z_P =  torch.FloatTensor( IC_v_np ).cuda()
    v_xx_P =  torch.FloatTensor( IC_v_np ).cuda()
    v_yy_P =  torch.FloatTensor( IC_v_np ).cuda()
    v_zz_P =  torch.FloatTensor( IC_v_np ).cuda()
    v_xz_P =  torch.FloatTensor( IC_v_np ).cuda()
    v_xy_P =  torch.FloatTensor( IC_v_np ).cuda()
    v_yz_P =  torch.FloatTensor( IC_v_np ).cuda()
    
    w_x_P =  torch.FloatTensor( IC_w_np ).cuda()
    w_y_P =  torch.FloatTensor( IC_w_np ).cuda()
    w_z_P =  torch.FloatTensor( IC_w_np ).cuda()
    w_xx_P =  torch.FloatTensor( IC_w_np ).cuda()
    w_yy_P =  torch.FloatTensor( IC_w_np ).cuda()
    w_zz_P =  torch.FloatTensor( IC_w_np ).cuda()
    w_xz_P =  torch.FloatTensor( IC_w_np ).cuda()
    w_xy_P =  torch.FloatTensor( IC_w_np ).cuda()
    w_yz_P =  torch.FloatTensor( IC_w_np ).cuda()    

    #u_P = Variable( torch.FloatTensor( IC_u_np ) )
    #v_P = Variable( torch.FloatTensor( IC_v_np ) )
    #
    Loss = 0.0
    
    #Chorin's projection method for solution of incompressible Navier-Stokes PDE with periodic boundary conditions in a box. 
        
    for i in range(int(T_LES/dt_LES)):
        
        u_x_P, u_y_P, u_z_P, u_xx_P, u_yy_P, u_zz_P, u_xy_P, u_xz_P, u_yz_P = FiniteDifference_u(u_P, u_x_P, u_y_P, u_z_P, u_xx_P, u_yy_P, u_zz_P, u_xy_P, u_xz_P, u_yz_P)
        
        v_x_P, v_y_P, v_z_P, v_xx_P, v_yy_P, v_zz_P, v_xy_P, v_xz_P, v_yz_P = FiniteDifference_u(v_P, v_x_P, v_y_P, v_z_P, v_xx_P, v_yy_P, v_zz_P, v_xy_P, v_xz_P, v_yz_P)
        
        w_x_P, w_y_P, w_z_P, w_xx_P, w_yy_P, w_zz_P, w_xy_P, w_xz_P, w_yz_P = FiniteDifference_u(w_P, w_x_P, w_y_P, w_z_P, w_xx_P, w_yy_P, w_zz_P, w_xy_P, w_xz_P, w_yz_P)
        
        #input_data = torch.cat( (u_P, v_P, w_P, 0.0*p_P, u_x_P, v_x_P, w_x_P,  u_y_P, v_y_P, w_y_P, u_z_P, v_z_P, w_z_P, u_xx_P, v_xx_P, w_xx_P, u_yy_P, v_yy_P, w_yy_P, u_zz_P, v_zz_P, w_zz_P, u_xy_P, v_xy_P, w_xy_P, u_xz_P, v_xz_P, w_xz_P, u_yz_P, v_yz_P, w_yz_P), 3 )
                                 
        #input_data = input_data/x_max_P
        
        #model_output = model(input_data)
        
        #output = (output_X_plus[:,0:1] - output_X_minus[:,0:1] )/(2*Delta_DNS_grid*100000.0) + (output_Y_plus[:,3:4] - output_Y_minus[:,3:4] )/(2*Delta_DNS_grid*100000.0) + (output_Z_plus[:,4:5] - output_Z_minus[:,4:5] )/(2*Delta_DNS_grid*100000.0)        
        
        #Closure_u_P[1:-1, 1:-1, 1:-1, :] = (model_output[2:,1:-1,1:-1,0:1] - model_output[0:-2,1:-1,1:-1,0:1] )/(2*dx) + (model_output[1:-1,2:,1:-1,3:4] - model_output[1:-1, 0:-2, 1:-1 ,3:4] )/(2*dx) + (model_output[1:-1, 1:-1,2:,4:5] - model_output[1:-1, 1:-1, 0:-2 ,4:5] )/(2*dx)        
        
        #output2 = (output_X_plus[:,3:4] - output_X_minus[:,3:4] )/(2*Delta_DNS_grid*100000.0) + (output_Y_plus[:,1:2] - output_Y_minus[:,1:2] )/(2*Delta_DNS_grid*100000.0) + (output_Z_plus[:,5:6] - output_Z_minus[:,5:6] )/(2*Delta_DNS_grid*100000.0)        
        #Closure_v_P[1:-1, 1:-1, 1:-1, :] = (model_output[2:,1:-1, 1:-1, 3:4] - model_output[0:-2, 1:-1, 1:-1 ,3:4] )/(2*dx) + (model_output[1:-1, 2:, 1:-1,1:2] - model_output[1:-1, 0:-2, 1:-1 ,1:2] )/(2*dx) + (model_output[1:-1, 1:-1, 2:,5:6] - model_output[1:-1, 1:-1, 0:-2 ,5:6] )/(2*dx)        

        #output3 = (output_X_plus[:,4:5] - output_X_minus[:,4:5] )/(2*Delta_DNS_grid*100000.0) + (output_Y_plus[:,5:6] - output_Y_minus[:,5:6] )/(2*Delta_DNS_grid*100000.0) + (output_Z_plus[:,2:3] - output_Z_minus[:,2:3] )/(2*Delta_DNS_grid*100000.0)   
                
        #Closure_w_P[1:-1, 1:-1, 1:-1, :] = (model_output[2:, 1:-1, 1:-1 ,4:5] - model_output[0:-2, 1:-1, 1:-1 ,4:5] )/(2*dx) + (model_output[1:-1, 2:, 1:-1,5:6] - model_output[1:-1, 0:-2, 1:-1 ,5:6] )/(2*dx) + (model_output[1:-1, 1:-1, 2:,2:3] - model_output[1:-1, 1:-1, 0:-2 ,2:3] )/(2*dx)   
        
        
        
        #u_x_P_UP, u_y_P_UP, u_z_P_UP = FiniteDifference_u_UPWIND(u_P)
        #v_x_P_UP, v_y_P_UP, v_z_P_UP = FiniteDifference_u_UPWIND(v_P)
        #w_x_P_UP, w_y_P_UP, w_z_P_UP = FiniteDifference_u_UPWIND(w_P)   
        
        #u_x_P_DOWN, u_y_P_DOWN, u_z_P_DOWN = FiniteDifference_u_DOWNWIND(u_P)
        #v_x_P_DOWN, v_y_P_DOWN, v_z_P_DOWN = FiniteDifference_u_DOWNWIND(v_P)
        #w_x_P_DOWN, w_y_P_DOWN, w_z_P_DOWN = FiniteDifference_u_DOWNWIND(w_P)        
        
        #Booleans..
        #I_u_UP = (u_P >= 0).type(torch.FloatTensor)
        #I_u_DOWN = (u_P < 0).type(torch.FloatTensor)
        #I_v_UP = (v_P >= 0).type(torch.FloatTensor)
        #I_v_DOWN = (v_P < 0).type(torch.FloatTensor)
        #I_w_UP = (w_P >= 0).type(torch.FloatTensor)
        #I_w_DOWN = (w_P < 0).type(torch.FloatTensor)

        #u_x_P = u_x_P_UP*I_u_UP + u_x_P_DOWN*I_u_DOWN
        #v_x_P = v_x_P_UP*I_u_UP + v_x_P_DOWN*I_u_DOWN
        #w_x_P = w_x_P_UP*I_u_UP + w_x_P_DOWN*I_u_DOWN

        #u_y_P = u_y_P_UP*I_v_UP + u_y_P_DOWN*I_v_DOWN
        #v_y_P = v_y_P_UP*I_v_UP + v_y_P_DOWN*I_v_DOWN
        #w_y_P = w_y_P_UP*I_v_UP + w_y_P_DOWN*I_v_DOWN   
        
        #u_z_P = u_z_P_UP*I_w_UP + u_z_P_DOWN*I_w_DOWN
        #v_z_P = v_z_P_UP*I_w_UP + v_z_P_DOWN*I_w_DOWN
        #w_z_P = w_z_P_UP*I_w_UP + w_z_P_DOWN*I_w_DOWN           
 
        Nonlinear_term_u_P = u_x_P*u_P + u_y_P*v_P  + u_z_P*w_P

        Nonlinear_term_v_P = v_x_P*u_P + v_y_P*v_P  + v_z_P*w_P
        
        Nonlinear_term_w_P = w_x_P*u_P + w_y_P*v_P  + w_z_P*w_P
        
        
        #Diffusion term
        
        Diffusion_term_u = u_xx_P + u_yy_P + u_zz_P
        
        Diffusion_term_v = v_xx_P + v_yy_P + v_zz_P
        
        Diffusion_term_w = w_xx_P + w_yy_P + w_zz_P
     
        
        #model_output = model( u_P)
        
        #model_output2 = model_output.cpu()
        
        Closure_u_P, Closure_v_P, Closure_w_P = Smagorinsky(u_x_P, u_y_P, u_z_P, v_x_P, v_y_P, v_z_P, w_x_P, w_y_P, w_z_P, Closure_u_P, Closure_v_P, Closure_w_P)

        u_P = u_P + ( (mu/rho)*Diffusion_term_u -Nonlinear_term_u_P  - Closure_u_P  )*dt_LES
        
        v_P = v_P + ( (mu/rho)*Diffusion_term_v -Nonlinear_term_v_P  - Closure_v_P   )*dt_LES
        
        w_P = w_P + ( (mu/rho)*Diffusion_term_w -Nonlinear_term_w_P  - Closure_w_P   )*dt_LES
        
        
        
        #Update pressure using an iteration...
        
        Source_term = rho*dx*dx*(u_x_P + v_y_P + w_z_P)
        
        for j in range( Num_pressure_iterations ):
            
            p_OLD_P[:,:,:,:] = p_P[0:,0:,0:,0:]
            
            Pressure_update_j1 = p_OLD_P[2:,1:-1,1:-1,:] + p_OLD_P[0:-2,1:-1,1:-1,:]
                 
            
            Pressure_update_j2 = p_OLD_P[1:-1,2:,1:-1,:] + p_OLD_P[1:-1,0:-2,1:-1,:]
            
            Pressure_update_j3 = p_OLD_P[1:-1,1:-1,2:,:] + p_OLD_P[1:-1,1:-1,0:-2,:]

            #interior of domain
            p_P[1:-1, 1:-1,1:-1,:] = (1.0/6.0)*( Pressure_update_j1 + Pressure_update_j2  + Pressure_update_j3 - Source_term[1:-1,1:-1,1:-1,:])
            
            #Edges of domain
            p_P[0, 1:-1,1:-1,:] = (1.0/6.0)*( (p_OLD_P[1,1:-1,1:-1,:] + p_OLD_P[-1,1:-1,1:-1,:])  + (p_OLD_P[0,2:,1:-1,:] + p_OLD_P[0,0:-2,1:-1,:] )  +  (p_OLD_P[0,1:-1,2:,:] + p_OLD_P[0,1:-1,0:-2,:])      - Source_term[0,1:-1,1:-1,:])
            p_P[-1, 1:-1,1:-1,:] = (1.0/6.0)*( (p_OLD_P[0,1:-1,1:-1,:] + p_OLD_P[-2,1:-1,1:-1,:])  + (p_OLD_P[-1,2:,1:-1,:] + p_OLD_P[-1,0:-2,1:-1,:] )  +  (p_OLD_P[-1,1:-1,2:,:] + p_OLD_P[-1,1:-1,0:-2,:])      - Source_term[-1,1:-1,1:-1,:])  
            
            p_P[1:-1, 0,1:-1,:] = (1.0/6.0)*( p_OLD_P[2:,0,1:-1,:] + p_OLD_P[0:-2,0,1:-1,:] + p_OLD_P[1:-1,1,1:-1,:] + p_OLD_P[1:-1,-1,1:-1,:]  + p_OLD_P[1:-1,0,2:,:] + p_OLD_P[1:-1,0,0:-2,:]      - Source_term[1:-1,0,1:-1,:])
            p_P[1:-1, -1,1:-1,:] = (1.0/6.0)*( p_OLD_P[2:,-1,1:-1,:] + p_OLD_P[0:-2,-1,1:-1,:] + p_OLD_P[1:-1,0,1:-1,:] + p_OLD_P[1:-1,-2,1:-1,:]  + p_OLD_P[1:-1,-1,2:,:] + p_OLD_P[1:-1,-1,0:-2,:]      - Source_term[1:-1,-1,1:-1,:])            
            
            p_P[1:-1, 1:-1,0,:] = (1.0/6.0)*( p_OLD_P[2:,1:-1,0,:] + p_OLD_P[0:-2,1:-1,0,:]+ p_OLD_P[1:-1,2:,0,:] + p_OLD_P[1:-1,0:-2,0,:] + p_OLD_P[1:-1,1:-1,1,:] + p_OLD_P[1:-1,1:-1,-1,:]  - Source_term[1:-1,1:-1,0,:]) 
            p_P[1:-1, 1:-1,-1,:] = (1.0/6.0)*( p_OLD_P[2:,1:-1,-1,:] + p_OLD_P[0:-2,1:-1,-1,:]+ p_OLD_P[1:-1,2:,-1,:] + p_OLD_P[1:-1,0:-2,-1,:] + p_OLD_P[1:-1,1:-1,0,:] + p_OLD_P[1:-1,1:-1,-2,:]  - Source_term[1:-1,1:-1,-1,:])            
            
            #Corners of domain
            p_P[0, 0,0,:] = (1.0/6.0)*( (p_OLD_P[1,0,0,:] + p_OLD_P[-1,0,0,:])  + (p_OLD_P[0,1,0,:] + p_OLD_P[0,-1,0,:] )  +  (p_OLD_P[0,0,1,:] + p_OLD_P[0,0,-1,:]) - Source_term[0,0,0,:])
            p_P[0, 0,-1,:] = (1.0/6.0)*( (p_OLD_P[1,0,-1,:] + p_OLD_P[-1,0,-1,:])  + (p_OLD_P[0,1,-1,:] + p_OLD_P[0,-1,-1,:] )  +  (p_OLD_P[0,0,0,:] + p_OLD_P[0,0,-2,:]) - Source_term[0,0,-1,:])    
            
            p_P[0, -1,0,:] = (1.0/6.0)*( (p_OLD_P[1,-1,0,:] + p_OLD_P[-1,-1,0,:])  + (p_OLD_P[0,0,0,:] + p_OLD_P[0,-2,0,:] )  +  (p_OLD_P[0,-1,1,:] + p_OLD_P[0,-1,-1,:]) - Source_term[0,-1,0,:])     
            p_P[0, -1,-1,:] = (1.0/6.0)*( (p_OLD_P[1,-1,-1,:] + p_OLD_P[-1,-1,-1,:])  + (p_OLD_P[0,0,-1,:] + p_OLD_P[0,-2,-1,:] )  +  (p_OLD_P[0,-1,0,:] + p_OLD_P[0,-1,-2,:]) - Source_term[0,-1,-1,:])    
            
            
            p_P[-1, 0,0,:] = (1.0/6.0)*( (p_OLD_P[0,0,0,:] + p_OLD_P[-2,0,0,:])  + (p_OLD_P[-1,1,0,:] + p_OLD_P[-1,-1,0,:] )  +  (p_OLD_P[-1,0,1,:] + p_OLD_P[-1,0,-1,:]) - Source_term[-1,0,0,:])     
            p_P[-1, 0,-1,:] = (1.0/6.0)*( (p_OLD_P[0,0,-1,:] + p_OLD_P[-2,0,-1,:])  + (p_OLD_P[-1,1,-1,:] + p_OLD_P[-1,-1,-1,:] )  +  (p_OLD_P[-1,0,0,:] + p_OLD_P[-1,0,-2,:]) - Source_term[-1,0,-1,:])              
            
            #Need to still work on this...!
            p_P[-1, -1,0,:] = (1.0/6.0)*( (p_OLD_P[0,-1,0,:] + p_OLD_P[-2,-1,0,:])  + (p_OLD_P[-1,0,0,:] + p_OLD_P[-1,-2,0,:] )  +  (p_OLD_P[-1,-1,1,:] + p_OLD_P[-1,-1,-1,:]) - Source_term[-1,-1,0,:])     
            p_P[-1, -1,-1,:] = (1.0/6.0)*( (p_OLD_P[0,-1,-1,:] + p_OLD_P[-2,-1,-1,:])  + (p_OLD_P[-1,0,-1,:] + p_OLD_P[-1,-2,-1,:] )  +  (p_OLD_P[-1,-1,0,:] + p_OLD_P[-1,-1,-2,:]) - Source_term[-1,-1,-1,:])   
            

        
        #pressure gradients
        p_x_P[1:-1,:,:,:] = (p_P[2:,:, :,:] - p_P[0:-2,:,:,:])/(2*dx)
        p_x_P[-1,:,:,:] = (p_P[0,:, :,:] - p_P[-2,:,:,:])/(2*dx)
        p_x_P[0,:,:,:] = (p_P[1,:, :,:] - p_P[-1,:,:,:])/(2*dx)
        
        p_y_P[:,1:-1,:,:] = (p_P[:,2:,:,:] - p_P[:,0:-2,:,:])/(2*dx)
        p_y_P[:,-1,:,:] = (p_P[:,0,:,:] - p_P[:,-2,:,:])/(2*dx)
        p_y_P[:,0,:,:] = (p_P[:,1,:,:] - p_P[:,-1,:,:])/(2*dx)
        
        p_z_P[:,:,1:-1,:] = (p_P[:,:,2:,:] - p_P[:,:,0:-2,:])/(2*dx)
        p_z_P[:,:,-1,:] = (p_P[:,:,0,:] - p_P[:,:,-2,:])/(2*dx)
        p_z_P[:,:,0,:] = (p_P[:,:,1,:] - p_P[:,:,-1,:])/(2*dx)
       
        
        #Update velocities using updated pressure...
        u_P = u_P - p_x_P*(1.0/rho)
        v_P = v_P - p_y_P*(1.0/rho)
        w_P = w_P - p_z_P*(1.0/rho)
        

        #Diff = u_P - Variable( torch.FloatTensor( np.matrix( u_DNS_downsamples[T_factor*(i+1),:]).T ) )
    Diff = u_P - target_P
    Loss_i = torch.mean( torch.abs( Diff ) )
    Loss = Loss + Loss_i
    
    #Loss_np = Loss.cpu().numpy()
    
    time2 = time.time()
    time_elapsed = time2 - time1
    
    test = torch.mean( u_P)
    
    error = np.mean(np.abs( u_P.cpu().numpy()[:,:,:,0:1] -  target_P.cpu().numpy()[:,:,:,0:1] ) )
    
    print(iterations, test, error,  time_elapsed)
