# ------------------------------------------------------------------------
#
# PyFlow: A GPU-accelerated CFD platform written in Python
#
# @file sfsmodel_smagorinsky.py
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
# Smagorinsky model for the SFS residual stress
# ----------------------------------------------------
class stress_constCs:
    def __init__(self,inputConfig,geo):
        #,CsIn=0.18,expFilterFacIn=1.0):

        # External model type identifier
        self.modelType = 'eddyVisc'

        # Model parameters
        # Assuming uniform grid for now
        try:
            self.Cs = inputConfig.Cs
        except AttributeError:
            self.Cs = 0.18
        try:
            expFilterFacIn = inputConfig.expFilterFac
        except AttributeError:
            expFilterFacIn = 1.0
            
        if (geo.type=='uniform'):
            # expFilterFac is the ratio between the filter width and LES grid size
            self.Delta = geo.dx * expFilterFacIn
        else:
            raise Exception('sfsmodel_smagorinsky: geometry type not implemented')

        
    # ----------------------------------------------------
    # Compute the eddy viscosity
    def eddyVisc(self,state_u,state_v,state_w,rho,metric):

        # Cell-centered velocity gradients
        metric.grad_vel_center(state_u,'u')
        metric.grad_vel_center(state_v,'v')
        metric.grad_vel_center(state_w,'w')

        # Filtered strain rate magnitude
        S_11 = state_u.grad_x
        S_22 = state_v.grad_y
        S_33 = state_w.grad_z
        S_12 = 0.5*(state_u.grad_y + state_v.grad_x)
        S_13 = 0.5*(state_u.grad_z + state_w.grad_x)
        S_23 = 0.5*(state_v.grad_z + state_w.grad_y)
        S_mag = torch.sqrt(2.0*( S_11*S_11 + S_22*S_22 + S_33*S_33 +
                                 2.0*(S_12*S_12 + S_13*S_13  + S_23*S_23) ))
        
        # Eddy viscosity
        mu_t = rho * (self.Cs*self.Delta)**2 * S_mag
        
        return mu_t
