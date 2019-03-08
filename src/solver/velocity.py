# ------------------------------------------------------------------------
#
# PyFlow: A GPU-accelerated CFD platform written in Python
#
# @file velocity.py
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
import sys
#
sys.path.append("../metric")
import metric_collocated as metric

# ----------------------------------------------------
# Velocity predictor rhs
# ----------------------------------------------------
def rhs_predictor(state_u_P,state_v_P,state_w_P,uMax,vMax,wMax,mu,rho,genericOrder,dx):

    # Compute all derivatives
    metric.FiniteDifference_u(genericOrder,dx,state_u_P)
    metric.FiniteDifference_u(genericOrder,dx,state_v_P)
    metric.FiniteDifference_u(genericOrder,dx,state_w_P)
    
    # Advection-diffusion
    Nonlinear_term_u_P = state_u_P.grad_x*uMax + state_u_P.grad_y*vMax  + state_u_P.grad_z*wMax
    Nonlinear_term_v_P = state_v_P.grad_x*uMax + state_v_P.grad_y*vMax  + state_v_P.grad_z*wMax
    Nonlinear_term_w_P = state_w_P.grad_x*uMax + state_w_P.grad_y*vMax  + state_w_P.grad_z*wMax
        
    # Navier-Stokes
    #Nonlinear_term_u_P = u_x_P*u_P + u_y_P*v_P  + u_z_P*w_P
    #Nonlinear_term_v_P = v_x_P*u_P + v_y_P*v_P  + v_z_P*w_P
    #Nonlinear_term_w_P = w_x_P*u_P + w_y_P*v_P  + w_z_P*w_P
        
    #Diffusion term
    Diffusion_term_u = state_u_P.grad_xx + state_u_P.grad_yy + state_u_P.grad_zz
    Diffusion_term_v = state_v_P.grad_xx + state_v_P.grad_yy + state_v_P.grad_zz
    Diffusion_term_w = state_w_P.grad_xx + state_w_P.grad_yy + state_w_P.grad_zz

    # Accumulate to rhs
    rhs_u = (mu/rho)*Diffusion_term_u - Nonlinear_term_u_P
    rhs_v = (mu/rho)*Diffusion_term_v - Nonlinear_term_v_P
    rhs_w = (mu/rho)*Diffusion_term_w - Nonlinear_term_w_P

    # Return the rhs
    return rhs_u, rhs_v, rhs_w
