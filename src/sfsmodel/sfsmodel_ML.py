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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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



# ----------------------------------------------------
# Machine learning model for the SFS residual stress
# ----------------------------------------------------
class residual_stress:
    def __init__(self,decomp,geo,metric,modelDictName):
        # Default precision
        prec = decomp.prec

        # External model type identifier
        self.modelType = 'tensor'

        # Data sizes
        nxo_ = decomp.nxo_
        nyo_ = decomp.nyo_
        nzo_ = decomp.nzo_

        # Allocate workspace arrays
        self.FXX = torch.zeros(nxo_,nyo_,nzo_,dtype=prec).to(decomp.device)
        self.FYY = torch.zeros(nxo_,nyo_,nzo_,dtype=prec).to(decomp.device)
        self.FZZ = torch.zeros(nxo_,nyo_,nzo_,dtype=prec).to(decomp.device)
        self.FXY = torch.zeros(nxo_,nyo_,nzo_,dtype=prec).to(decomp.device)
        self.FYZ = torch.zeros(nxo_,nyo_,nzo_,dtype=prec).to(decomp.device)
        self.FXZ = torch.zeros(nxo_,nyo_,nzo_,dtype=prec).to(decomp.device)
        self.tmp = torch.zeros(nxo_,nyo_,nzo_,dtype=prec).to(decomp.device)

        # Filter width
        if (geo.type=='uniform'):
            self.const = geo.dx*geo.dx/12.0
        else:
            raise Exception('sfsmodel_gradient: geometry type not implemented')

        
    # ----------------------------------------------------
    # Evaluate the model
    def update(self,state_u,state_v,state_w,metric):

        # Cell-centered velocity gradients
        metric.grad_vel_center(state_u,'u')
        metric.grad_vel_center(state_v,'v')
        metric.grad_vel_center(state_w,'w')

        # Update the residual stress tensor
        # xx
        self.tmp.copy_( state_u.grad_x )
        self.tmp.mul_ ( state_u.grad_x )
        self.FXX.copy_( self.tmp )
        self.tmp.copy_( state_u.grad_y )
        self.tmp.mul_ ( state_u.grad_y )
        self.FXX.add_ ( self.tmp )
        self.tmp.copy_( state_u.grad_z )
        self.tmp.mul_ ( state_u.grad_z )
        self.FXX.add_ ( self.tmp )
        self.FXX.mul_ ( self.const )
        
        # yy
        self.tmp.copy_( state_v.grad_x )
        self.tmp.mul_ ( state_v.grad_x )
        self.FYY.copy_( self.tmp )
        self.tmp.copy_( state_v.grad_y )
        self.tmp.mul_ ( state_v.grad_y )
        self.FYY.add_ ( self.tmp )
        self.tmp.copy_( state_v.grad_z )
        self.tmp.mul_ ( state_v.grad_z )
        self.FYY.add_ ( self.tmp )
        self.FYY.mul_ ( self.const )
        
        # zz
        self.tmp.copy_( state_w.grad_x )
        self.tmp.mul_ ( state_w.grad_x )
        self.FZZ.copy_( self.tmp )
        self.tmp.copy_( state_w.grad_y )
        self.tmp.mul_ ( state_w.grad_y )
        self.FZZ.add_ ( self.tmp )
        self.tmp.copy_( state_w.grad_z )
        self.tmp.mul_ ( state_w.grad_z )
        self.FZZ.add_ ( self.tmp )
        self.FZZ.mul_ ( self.const )
        
        # xy
        self.tmp.copy_( state_u.grad_x )
        self.tmp.mul_ ( state_v.grad_x )
        self.FXY.copy_( self.tmp )
        self.tmp.copy_( state_u.grad_y )
        self.tmp.mul_ ( state_v.grad_y )
        self.FXY.add_ ( self.tmp )
        self.tmp.copy_( state_u.grad_z )
        self.tmp.mul_ ( state_v.grad_z )
        self.FXY.add_ ( self.tmp )
        self.FXY.mul_ ( self.const )
        
        # xz
        self.tmp.copy_( state_u.grad_x )
        self.tmp.mul_ ( state_w.grad_x )
        self.FXZ.copy_( self.tmp )
        self.tmp.copy_( state_u.grad_y )
        self.tmp.mul_ ( state_w.grad_y )
        self.FXZ.add_ ( self.tmp )
        self.tmp.copy_( state_u.grad_z )
        self.tmp.mul_ ( state_w.grad_z )
        self.FXZ.add_ ( self.tmp )
        self.FXZ.mul_ ( self.const )
        
        # yz
        self.tmp.copy_( state_v.grad_x )
        self.tmp.mul_ ( state_w.grad_x )
        self.FYZ.copy_( self.tmp )
        self.tmp.copy_( state_v.grad_y )
        self.tmp.mul_ ( state_w.grad_y )
        self.FYZ.add_ ( self.tmp )
        self.tmp.copy_( state_v.grad_z )
        self.tmp.mul_ ( state_w.grad_z )
        self.FYZ.add_ ( self.tmp )
        self.FYZ.mul_ ( self.const )
        
