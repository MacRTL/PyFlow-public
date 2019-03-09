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

#Smagorinsky Closure Model
def eval(dx, state_u, state_v, state_w, Closure_u_P, Closure_v_P, Closure_w_P, metric):

    # Compute velocity gradients
    metric.grad_vel_visc(state_u)
    metric.grad_vel_visc(state_v)
    metric.grad_vel_visc(state_w)

    # Local pointers to velocity gradients
    u_x_P = state_u.grad_x
    u_y_P = state_u.grad_y
    u_z_P = state_u.grad_z
    v_x_P = state_v.grad_x
    v_y_P = state_v.grad_y
    v_z_P = state_v.grad_x
    w_x_P = state_w.grad_x
    w_y_P = state_w.grad_y
    w_z_P = state_w.grad_x
    
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
    
    Closure_u_P[1:-1,1:-1,1:-1] =  Coeff[1:-1,1:-1, 1:-1]*( (S_11[2:,1:-1, 1:-1] - S_11[0:-2,1:-1, 1:-1])    +  (S_12[1:-1,2:, 1:-1] - S_12[1:-1,0:-2, 1:-1]) +  (S_13[1:-1,1:-1, 2:] - S_13[1:-1, 1:-1, 0:-2])  )/(2*dx)
    
    Closure_v_P[1:-1,1:-1,1:-1] =  Coeff[1:-1,1:-1, 1:-1]*( (S_12[2:,1:-1, 1:-1] - S_12[0:-2,1:-1, 1:-1])    +  (S_22[1:-1,2:, 1:-1] - S_22[1:-1,0:-2, 1:-1]) +  (S_23[1:-1,1:-1, 2:] - S_23[1:-1, 1:-1, 0:-2])  )/(2*dx)
    
    Closure_w_P[1:-1,1:-1,1:-1] =  Coeff[1:-1,1:-1, 1:-1]*( (S_13[2:,1:-1, 1:-1] - S_13[0:-2,1:-1, 1:-1])    +  (S_23[1:-1,2:, 1:-1] - S_23[1:-1,0:-2, 1:-1]) +  (S_33[1:-1,1:-1, 2:] - S_33[1:-1, 1:-1, 0:-2])  )/(2*dx)
    
    
    return Closure_u_P, Closure_v_P, Closure_w_P
    
