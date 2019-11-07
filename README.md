# Welcome

Welcome to the PyFlow README! This page contains basic installation instructions and usage tips for the PyFlow package.


## Generic Environment Setup

A generic environment is the best starting point for using PyFlow on a personal machine. For larger runs, machine-specific environments are described below.

Anaconda is the recommended Python package manager for generic PyFlow environments. You'll need to instal Anaconda first; full instructions for the platform of your choice may be found [on the web](https://www.anaconda.com/distribution/). Please configure Anaconda to use the latest Python 3.x distribution.


After installing Anaconda, create a PyFlow-specific environment:
```
conda create --name pyflow
```
In future login shells, the environment can be activated using `conda activate pyflow`.

PyFlow depends on several Python packages: `mpi4py`, `torch`, `scipy`, `numpy`, and `matplotlib` (the last is optional for visualization support). The following PyTorch install command will work for Linux and OS X machines without GPU/CUDA support. For CUDA support on your platform, refer to the [PyTorch Getting Started page](https://pytorch.org/get-started/locally).
```
conda install mpi4py
conda install pytorch-cpu torchvision-cpu -c pytorch
conda install matplotlib
conda install scipy
```



## Running PyFlow

PyFlow contains a heierarchy of modules structured around the main function `PyFlow.run()`. Python-based driver scripts configure and invoke each instance of PyFlow. In certain advanced cases, PyFlow calls back to the driver script during the simulation, *e.g.*, to retrieve the name of a target data file.

The basic tasks the driver script performs are:

1. Create an object containing input parameters (`inputConfig`)
2. Invoke PyFlow via `PyFlow.run(inputConfig)`

Several example driver scripts are available in the repository:

1. DNS of decaying isotropic turbulence on a 128^3 mesh (`run_PyFlow_example_dnsbox128.py`)
2. Adjoint training for LES of decaying isotropic turbulence on a 64^3 mesh, downsampled from 1024^3 DNS data (`run_PyFlow_adjoint_training.py`)

Each of these example cases contains a companion shell script `start_PyFlow.sh` that sets up an out-of-source run environment. Out-of-source runs are **highly** recommended -- at least until we set up PyFlow as a packaged Python project. ;)

The example shell scripts can be invoked using `source start_PyFlow.sh`.


Since PyFlow is designed to be scripted by Python, driver scripts can get as fancy as the user wants! In general, a driver script is executed on the command line in serial via
```
python <run_PyFlow_script.py>
```
or in parallel via
```
mpirun -np <NUM_PROC> python <run_PyFlow_script.py>
```

**NOTE 1:** Out-of-source runs are HIGHLY recommended. This means setting up your job submission script (or shell script, etc.) to copy the PyFlow source tree to the current working directory.

**NOTE 2:** Parallel execution commands might vary system-to-system. If the machine you're using has a site-specific environment configuration, please see the next section.



## Machine-specific Environment Setup

The following sub-sections contain site-specific environment setup details.


### Blue Waters (NCSA)

NCSA Blue Waters is a Cray XE/XK hybrid machine composed of AMD 6276 "Interlagos" CPUs and Nvidia GK110 (K20X) "Kepler" GPUs. XE nodes have 32 CPU cores and no GPU; XK nodes have 16 CPU cores and one K20X GPU.

A containerized Python environment has been created on Blue Waters for PyFlow using Shifter/Docker. The environment contains consistent versions of `mpi4py` and `torch` with CUDA support.

Example job submission scripts for Blue Waters may be found in
```
run_scripts/blue_waters/
```
The job submission script `mybatch_PyFlow.pbs` contains all PBS scheduler directives, configures the Shifter environment, copies the PyFlow source code to the PBS working directory, and invokes the bash script `start_PyFlow.sh` via the `aprun` command.

Note that the name of the PyFlow driver script, set by the environment variable `PYFLOW_DRIVER`, must be consistent between `mybatch_PyFlow.pbs` and `start_PyFlow.sh`.


### Lassen (LLNL)

LLNL Lassen is an 795-node IBM machine composed of Power9 CPUs and NVIDIA V100 "Volta" GPUs. Each node contains 44 CPU cores (40 available to the user) and four V100 GPUs. Please contact [Jon MacArt](mailto:jmacart@illinois.edu) for details about running PyFlow on this machine.

