# ------------------------------------------------------------------------
#
# PyFlow: A GPU-accelerated CFD platform written in Python
#
# @file sfsmodel_nn.py
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
