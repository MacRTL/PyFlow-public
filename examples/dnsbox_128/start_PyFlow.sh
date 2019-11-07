#!/bin/bash

# ------------------------------------------------------------
# User-configurable options
# ------------------------------------------------------------

# Name of PyFlow driver script
PYFLOW_DRIVER=run_PyFlow_example_dnsbox128.py

# Number of tasks (NTASKS) & number of tasks per node (NTASKS_NODE)
#   --> For GPU runs, leave NTASKS_NODE=1
NTASKS=1
NTASKS_NODE=1

# Path to PyFlow source code
PYFLOW_PATH=../../


# ------------------------------------------------------------
# No user-configurable options below here
# ------------------------------------------------------------

# Set up PyFlow environment
cp -r ${PYFLOW_PATH}/src .
cd src/core
cp ../../${PYFLOW_DRIVER} .

# Call the PyFlow driver
#   Change this to your system's default (mpirun, srun, aprun, etc.)
mpirun -np $NTASKS python ${PYFLOW_DRIVER}

# Process results
cd ../..
