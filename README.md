# reproduce_J1J2model_via_CNN
This repository aims to reproduce the results in Physical Review B 98 (10), 104426 (2018) and Physical Review B 103 (3), 035138 (2021) using FLAX and JAX

## dependency
Python 3.8\
JAX 0.2.9 and FLAX 0.3.0\
numba 0.52.0

## Motivation
There has been increasing interest in applying neural networks as variational ansatz for solving complex quantum many-particle system, such as solving the frustrated J1-J2 Heisenberg model. Based on previous successful trails with PyTorch, here I have re-written the MCMC program via JAX and FLAX. Therefore one can easily check the ground state energies for J2=0.5 case, within the competitive performance of JAX.

Until now I have verified the results for single layer CNN. Because of the special designed structure of the deep CNN, I am still confused on converting PyTorch weights to FLAX weights for deep CNN. 

## Usage
Change the model name in fire.py, then python fire.py to run.

Increasing sample number can lead to very long single thread time, for verifying energies on 10x10 square lattice, 10000 samples are enough.
