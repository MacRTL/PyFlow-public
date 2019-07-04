# ------------------------------------------------------------------------
#
# PyFlow: A GPU-accelerated CFD platform written in Python
#
# @file advancer.py
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

# The idea here is that rhsClass will have already been initialized
# with the correct number of RHS objects for the domain-specified
# advancer

# ----------------------------------------------------
# Explicit Euler advancer
# ----------------------------------------------------
class Euler:
    def __init__(self,state_u,state_v,state_w,rhsClass):
        # Get pointers to the state variables and rhs
        self.state_u  = state_u
        self.state_v  = state_v
        self.state_w  = state_w
        self.rhsClass = rhsClass

    def step(self,simDt):
        
        # Evaluate the rhs
        self.rhsClass.rhs1.evaluate(self.state_u,self.state_v,self.state_w)

        # Update the state
        self.state_u.update( self.state_u.interior + self.rhsClass.rhs1.rhs_u*simDt )
        self.state_v.update( self.state_v.interior + self.rhsClass.rhs1.rhs_v*simDt )
        self.state_w.update( self.state_w.interior + self.rhsClass.rhs1.rhs_w*simDt )


# ----------------------------------------------------
# RK4 advancer
# ----------------------------------------------------
class RK4:
    def __init__(self,state_u,state_uTmp,state_v,state_vTmp,
                 state_w,state_wTmp,rhsClass):
        # Get pointers to the state variables and rhs
        self.state_u    = state_u
        self.state_v    = state_v
        self.state_w    = state_w
        self.state_uTmp = state_uTmp
        self.state_vTmp = state_vTmp
        self.state_wTmp = state_wTmp
        self.rhsClass   = rhsClass

    def step(self,simDt):
        
        # Stage 1
        self.rhsClass.rhs1.evaluate(self.state_u,self.state_v,self.state_w)

        # Stage 2
        self.state_uTmp.ZAXPY(0.5*simDt,self.rhsClass.rhs1.rhs_u,self.state_u.interior())
        self.state_vTmp.ZAXPY(0.5*simDt,self.rhsClass.rhs1.rhs_v,self.state_v.interior())
        self.state_wTmp.ZAXPY(0.5*simDt,self.rhsClass.rhs1.rhs_w,self.state_w.interior())
        self.rhsClass.rhs2.evaluate(self.state_uTmp,self.state_vTmp,self.state_wTmp)

        # Stage 3
        self.state_uTmp.ZAXPY(0.5*simDt,self.rhsClass.rhs2.rhs_u,self.state_u.interior())
        self.state_vTmp.ZAXPY(0.5*simDt,self.rhsClass.rhs2.rhs_v,self.state_v.interior())
        self.state_wTmp.ZAXPY(0.5*simDt,self.rhsClass.rhs2.rhs_w,self.state_w.interior())
        self.rhsClass.rhs3.evaluate(self.state_uTmp,self.state_vTmp,self.state_wTmp)

        # Stage 4
        self.state_uTmp.ZAXPY(0.5*simDt,self.rhsClass.rhs3.rhs_u,self.state_u.interior())
        self.state_vTmp.ZAXPY(0.5*simDt,self.rhsClass.rhs3.rhs_v,self.state_v.interior())
        self.state_wTmp.ZAXPY(0.5*simDt,self.rhsClass.rhs3.rhs_w,self.state_w.interior())
        self.rhsClass.rhs4.evaluate(self.state_uTmp,self.state_vTmp,self.state_wTmp)
        
        # Update the state
        self.state_u.update( self.state_u.interior()
                             + simDt/6.0*( self.rhsClass.rhs1.rhs_u +
                                           2.0*self.rhsClass.rhs2.rhs_u +
                                           2.0*self.rhsClass.rhs3.rhs_u +
                                           self.rhsClass.rhs4.rhs_u ) )
        self.state_v.update( self.state_v.interior()
                             + simDt/6.0*( self.rhsClass.rhs1.rhs_v +
                                           2.0*self.rhsClass.rhs2.rhs_v +
                                           2.0*self.rhsClass.rhs3.rhs_v +
                                           self.rhsClass.rhs4.rhs_v ) )
        self.state_w.update( self.state_w.interior()
                             + simDt/6.0*( self.rhsClass.rhs1.rhs_w +
                                           2.0*self.rhsClass.rhs2.rhs_w +
                                           2.0*self.rhsClass.rhs3.rhs_w +
                                           self.rhsClass.rhs4.rhs_w ) )
