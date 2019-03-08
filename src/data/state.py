# ------------------------------------------------------------------------
#
# PyFlow: A GPU-accelerated CFD platform written in Python
#
# @file state.py
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

# ----------------------------------------------------
# Base class for state data
# ----------------------------------------------------
class data_P:
    def __init__(self,IC,IC_zeros):
        
        # Allocate arrays and set initial conditions
        if (torch.cuda.is_available()):
            # State data
            self.var     = torch.FloatTensor(IC).cuda()
            # First derivatives
            self.grad_x  = torch.FloatTensor(IC_zeros).cuda()
            self.grad_y  = torch.FloatTensor(IC_zeros).cuda()
            self.grad_z  = torch.FloatTensor(IC_zeros).cuda()
            # Second derivatives
            self.grad_xx = torch.FloatTensor(IC_zeros).cuda()
            self.grad_yy = torch.FloatTensor(IC_zeros).cuda()
            self.grad_zz = torch.FloatTensor(IC_zeros).cuda()
            self.grad_xz = torch.FloatTensor(IC_zeros).cuda()
            self.grad_xy = torch.FloatTensor(IC_zeros).cuda()
            self.grad_yz = torch.FloatTensor(IC_zeros).cuda()
            
        else:
            # State data
            self.var     = torch.FloatTensor(IC)
            # First derivatives
            self.grad_x  = torch.FloatTensor(IC_zeros)
            self.grad_y  = torch.FloatTensor(IC_zeros)
            self.grad_z  = torch.FloatTensor(IC_zeros)
            # Second derivatives
            self.grad_xx = torch.FloatTensor(IC_zeros)
            self.grad_yy = torch.FloatTensor(IC_zeros)
            self.grad_zz = torch.FloatTensor(IC_zeros)
            self.grad_xz = torch.FloatTensor(IC_zeros)
            self.grad_xy = torch.FloatTensor(IC_zeros)
            self.grad_yz = torch.FloatTensor(IC_zeros)


    def ZAXPY(self,A,X,Y):
        # Z = A*X + Y
        # Scalar A; vector X and Y
        self.var = A*X + Y
