# ------------------------------------------------------------------------
#
# PyFlow: A GPU-accelerated CFD platform written in Python
#
# @file constants.py
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

# Normalization weights
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
