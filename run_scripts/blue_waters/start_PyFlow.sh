#!/bin/bash
# setup the environment for python within container
# $MY_PATH and $MY_LD_LIBRARY_PATH are passed in from the 
# job script

# Python script to execute
#PYFLOW_DRIVER=run_PyFlow_test_smagorinsky.py
PYFLOW_DRIVER=run_PyFlow_test_ML.py

export PATH=$MY_PATH
export LD_LIBRARY_PATH=$MY_LD_LIBRARY_PATH
# want this first in PYTHONPATH so that we get the mpi4py that was built
# here on blue waters inside the container
export PYTHONPATH=\
/u/staff/arnoldg/pytorch/lib/python:\
$PYTHONPATH

# Run it
python ${PYFLOW_DRIVER}
